from functools import partial
import matplotlib.pyplot as plt

import cc3d
import numpy as np
import ot
import scipy
import scipy.sparse as sp
from joblib import Parallel, delayed

from .strip_utils import find_strip


def g_f(group_id, _ndx, _pos, _count):
    """Return indices in the original array that belong to group `group_id`."""
    return _ndx[_pos[group_id] : (_pos[group_id] + _count[group_id])]


def mean_sum_by_labs(X, labs):
    """
    Group rows of X according to integer labels `labs` and compute:

    - mean of all columns except the last,
    - sum of the last column.

    Parameters
    ----------
    X : ndarray, shape (num_points, dim)
        Features per point. The last column is typically an intensity.
    labs : ndarray, shape (num_points,)
        Integer labels defining clusters.

    Returns
    -------
    group_stats : ndarray, shape (num_groups, dim)
        For each group: mean of X[...,:-1] and sum of X[...,-1].
    group_index_fn : callable
        Function such that group_index_fn(g) returns the indices of points
        belonging to group g in the original X.
    """
    # NOTE: X can be multidimensional
    sorted_indices = np.argsort(labs)
    unique_ids, first_positions, counts = np.unique(
        labs[sorted_indices],
        return_index=True,
        return_counts=True,
    )

    group_sums = np.add.reduceat(X[sorted_indices], first_positions, axis=0)

    # Normalize all columns except the last by the group count (means)
    group_sums[:, :-1] = group_sums[:, :-1] / counts[:, None]

    group_index_fn = partial(
        g_f, _ndx=sorted_indices, _pos=first_positions, _count=counts
    )

    return group_sums, group_index_fn


def find_component(ds):
    """
    Find connected components in a 2D sparse dataset `ds`.

    Parameters
    ----------
    ds : scipy.sparse matrix
        2D sparse array of intensities.

    Returns
    -------
    cluster_stats : ndarray, shape (num_clusters, 3)
        For each cluster: [mean_row, mean_col, sum_intensity].
    cluster_index_fn : callable
        Function mapping a cluster id -> indices of its points in the
        ORIGINAL COO representation of `ds` (ds.tocoo()).
    """
    # Work in COO so we can relate clusters back to the sparse matrix
    coo = ds.tocoo()
    rows_all = coo.row
    cols_all = coo.col
    data_all = coo.data

    # Keep only strictly positive intensities: zeros are pure background
    pos_mask = data_all > 0.0
    if not np.any(pos_mask):
        # No positive mass at all -> no clusters
        empty_stats = np.zeros((0, 3), dtype=float)

        def empty_index_fn(_):
            return np.array([], dtype=int)

        return empty_stats, empty_index_fn

    # Subset of *positive* entries
    rows = rows_all[pos_mask]
    cols = cols_all[pos_mask]
    intensities = data_all[pos_mask]
    # Indices of these positives in the ORIGINAL COO arrays
    orig_indices = np.nonzero(pos_mask)[0]

    # Build a boolean image only on positive entries for cc3d
    A_bool = np.zeros(ds.shape, dtype=bool)
    A_bool[rows, cols] = True

    labels, _ = cc3d.connected_components(
        A_bool,
        binary_image=True,
        connectivity=8,
        return_N=True,
    )
    labs = labels[rows, cols]  # component labels for positive entries

    # Sanity: labs should all be > 0 here, but clip just in case
    valid = labs > 0
    if not np.any(valid):
        empty_stats = np.zeros((0, 3), dtype=float)

        def empty_index_fn(_):
            return np.array([], dtype=int)

        return empty_stats, empty_index_fn

    rows = rows[valid]
    cols = cols[valid]
    intensities = intensities[valid]
    labs = labs[valid]
    orig_indices = orig_indices[valid]

    # Group by component label: mean(row, col), sum(intensity)
    cluster_stats, cluster_index_fn_inner = mean_sum_by_labs(
        np.stack([rows, cols, intensities], axis=1),
        labs,
    )

    # Adapt the index function so that it returns INDICES INTO THE
    # ORIGINAL COO ARRAYS (rows_all/cols_all/data_all), not into the
    # compressed positive-only subset.
    def cluster_index_fn(group_id, _orig=orig_indices, _inner=cluster_index_fn_inner):
        # indices in [0 .. len(rows)-1] for this cluster
        sub_idx = _inner(group_id)
        # map back to original COO indices
        return _orig[sub_idx]

    return cluster_stats, cluster_index_fn


def _balanced_ot_with_dustbin_2d(
    X,
    Y,
    a=None,
    b=None,
    dust_cost=1.0,
    cost="sqeuclidean",
    axis_weights=None,
):
    """
    Solve balanced OT between 2D point clouds with an explicit dustbin using
    Earth Mover's Distance (POT), restricted to active points (mass > 0).

    Conventions
    -----------
    - N = number of source points
    - M = number of target / reference points

    2D setting
    ----------
    - X: source points in R^2, shape (N, 2)
    - Y: target points in R^2, shape (M, 2)
    - a: source weights on X, shape (N,)
    - b: target weights on Y, shape (M,)

    If a or b is None, uniform weights are used on the corresponding cloud.

    Active support
    --------------
    We only keep points with strictly positive mass on each side:
        active_source = (a > 0)
        active_target = (b > 0)
    then work in the compressed index spaces:
        X_act, a_act  (size k_source)
        Y_act, b_act  (size k_target)

    Dustbin construction (unbalanced → balanced)
    -------------------------------------------
    Let:
        sa = sum(a_act),  sb = sum(b_act).

    We build augmented marginals:
        a_ext = [a_act, sb] in R^{k_source + 1}_+
        b_ext = [b_act, sa] in R^{k_target + 1}_+

    and an augmented cost matrix C_ext of shape (k_source + 1, k_target + 1):
    - For i < k_source, j < k_target:
        C_ext[i, j] = ||x_i - y_j||^2 (in normalized units)
    - For i < k_source, j = k_target:
        C_ext[i, k_target] = dust_cost_scaled   (real -> dustbin)
    - For i = k_source, j < k_target:
        C_ext[k_source, j] = dust_cost_scaled   (dustbin -> real)
    - For i = k_source, j = k_target:
        C_ext[k_source, k_target] = 0.0         (dustbin -> dustbin)

    Then:
        Gamma_ext = ot.emd(a_ext, b_ext, C_ext)
    and we keep the real-real block Gamma_ext[:k_source, :k_target], which we
    expand back to a full (N, M) plan with zeros on inactive rows/cols.

    Coordinate normalization & dust_cost scaling
    -------------------------------------------
    To improve numerical conditioning, we rescale coordinates so that the
    bounding-box diagonal of X_act ∪ Y_act is O(1):
        coords_all = stack(X_act, Y_act)
        scale = max(1, ||max(coords_all) - min(coords_all)||_2)
        X_scaled = X_act / scale
        Y_scaled = Y_act / scale
    so typical squared distances are <= 1.

    To keep the semantics of dust_cost unchanged (interpreted in the ORIGINAL
    coordinate^2 units), we scale it internally:
        dust_cost_scaled = dust_cost / scale**2

    Marginal normalization for POT
    ------------------------------
    As in the 1D version, we:
    - clean NaN / inf,
    - clip tiny negatives to 0,
    - normalize a_ext and b_ext to sum ≈ 1 before calling ot.emd,
    - then rescale Gamma_ext by the original total mass (mass_ext).

    Parameters
    ----------
    X : array_like, shape (N, 2)
        Source point cloud in R^2.
    Y : array_like, shape (M, 2)
        Target point cloud in R^2.
    a : array_like, shape (N,), optional
        Source weights (non-negative). If None, uses uniform weights.
    b : array_like, shape (M,), optional
        Target weights (non-negative). If None, uses uniform weights.
    dust_cost : float
        Dustbin cost in the ORIGINAL (coordinate^2) distance units.
    cost : {"sqeuclidean"}, optional
        Ground cost on R^2.

    Returns
    -------
    Gamma_full : ndarray, shape (N, M)
        Real-real OT plan; rows/cols corresponding to zero-mass points are
        identically zero.
    """
    # --- Inputs and basic checks ------------------------------------------------
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"X must have shape (N, 2), got {X.shape}")
    if Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError(f"Y must have shape (M, 2), got {Y.shape}")

    N = X.shape[0]  # number of source points
    M = Y.shape[0]  # number of target points

    # Weights
    if a is None:
        source_weights = np.ones(N, dtype=float)
    else:
        source_weights = np.asarray(a, dtype=float).ravel()
        if source_weights.shape[0] != N:
            raise ValueError(f"a must have length {N}, got {source_weights.shape[0]}")

    if b is None:
        target_weights = np.ones(M, dtype=float)
    else:
        target_weights = np.asarray(b, dtype=float).ravel()
        if target_weights.shape[0] != M:
            raise ValueError(f"b must have length {M}, got {target_weights.shape[0]}")

    # --- Active support on each side -------------------------------------------
    active_source = source_weights > 0.0
    active_target = target_weights > 0.0
    idx_source = np.nonzero(active_source)[0]
    idx_target = np.nonzero(active_target)[0]
    k_source = idx_source.size
    k_target = idx_target.size

    # If no active point on at least one side: nothing meaningful to transport
    if k_source == 0 or k_target == 0:
        return np.zeros((N, M), dtype=float)

    X_act = X[idx_source]
    Y_act = Y[idx_target]
    a_act = source_weights[idx_source]
    b_act = target_weights[idx_target]
    sa = float(a_act.sum())
    sb = float(b_act.sum())

    # If both are (numerically) empty, nothing to transport
    if sa <= 0.0 and sb <= 0.0:
        return np.zeros((N, M), dtype=float)

    # Augmented marginals (before cleaning)
    a_ext = np.concatenate([a_act, [sb]]).astype(float)
    b_ext = np.concatenate([b_act, [sa]]).astype(float)

    # --- Clean & normalize for POT (improves numerical stability) --------------
    a_ext[~np.isfinite(a_ext)] = 0.0
    b_ext[~np.isfinite(b_ext)] = 0.0
    a_ext = np.clip(a_ext, 0.0, None)
    b_ext = np.clip(b_ext, 0.0, None)

    mass_a = float(a_ext.sum())
    mass_b = float(b_ext.sum())

    if mass_a <= 0.0 and mass_b <= 0.0:
        return np.zeros((N, M), dtype=float)

    # If only one side is empty, also return zeros (pathological input)
    if mass_a <= 0.0 or mass_b <= 0.0:
        return np.zeros((N, M), dtype=float)

    # They should be equal by construction; use their mean as robust scale
    mass_ext = 0.5 * (mass_a + mass_b)

    # Normalize to a probability simplex (sum ≈ 1)
    a_ext /= mass_ext
    b_ext /= mass_ext

    # --- Cost matrix on active points (2D squared Euclidean) -------------------
    if cost == "sqeuclidean":
        # Axis weights: w_x, w_y (default = 1,1 --> isotropic)
        if axis_weights is None:
            w = np.ones(2, dtype=float)
        else:
            w = np.asarray(axis_weights, dtype=float).ravel()
        if w.shape[0] != 2:
            raise ValueError(f"axis_weights must have length 2, got {w.shape[0]}")
        if np.any(w < 0):
            raise ValueError("axis_weights must be non-negative")

        # Apply anisotropic scaling: A = diag(sqrt(w_x), sqrt(w_y))
        sqrt_w = np.sqrt(w)
        X_aniso = X_act * sqrt_w  # (k_source, 2)
        Y_aniso = Y_act * sqrt_w  # (k_target, 2)

        # Global scale for numerical stability (on the transformed coords)
        coords_all = np.vstack([X_aniso, Y_aniso])
        bb_min = coords_all.min(axis=0)
        bb_max = coords_all.max(axis=0)
        diag = np.linalg.norm(bb_max - bb_min)
        scale = max(1.0, float(diag))

        X_scaled = X_aniso / scale
        Y_scaled = Y_aniso / scale

        # Pairwise squared Euclidean distances in transformed+scaled space
        diff = X_scaled[:, None, :] - Y_scaled[None, :, :]
        C_real = np.sum(diff**2, axis=2)

        # dust_cost is assumed in the SAME anisotropic metric units
        # (i.e. in terms of c(x,y) above). We divide by scale**2 as before.
        dust_cost_scaled = dust_cost / (scale**2)
    else:
        raise ValueError(f"Unsupported cost: {cost!r}")

    # Augmented cost matrix with dustbin (k_source+1, k_target+1)
    C_ext = np.empty((k_source + 1, k_target + 1), dtype=float)
    C_ext[:k_source, :k_target] = C_real
    C_ext[:k_source, k_target] = dust_cost_scaled  # real -> dustbin
    C_ext[k_source, :k_target] = dust_cost_scaled  # dustbin -> real
    C_ext[k_source, k_target] = 0.0  # dustbin -> dustbin

    # --- Solve balanced EMD on the augmented problem ---------------------------
    Gamma_ext = ot.emd(a_ext, b_ext, C_ext)

    # Rescale transport plan back to original total mass
    Gamma_ext *= mass_ext

    # Real-real block on the compressed supports
    Gamma_act = np.asarray(Gamma_ext[:k_source, :k_target], dtype=float)

    # --- Expand back to full (N, M) plan ---------------------------------------
    Gamma_full = np.zeros((N, M), dtype=float)
    Gamma_full[np.ix_(idx_source, idx_target)] = Gamma_act

    return Gamma_full


def _balanced_ot_near_target_source_2d(
    X,
    Y,
    a=None,
    b=None,
    dust_cost=1.0,
    cost="sqeuclidean",
    axis_weights=None,
    mass_tol=0.0,
    merge_duplicates=True,
    return_plan=False,
):
    """
    Build the 'near target' transported source measure from OT-with-dustbin.

    This function DOES NOT modify `_balanced_ot_with_dustbin_2d`. It only:
      1) calls it to get the real-real plan Gamma_full,
      2) interprets the remaining mass as 'staying at the source',
      3) builds a measure on the UNION of coordinates from X and Y,
         merging duplicates and removing zero-mass points.

    Definitions
    -----------
    Let:
        Gamma = _balanced_ot_with_dustbin_2d(...), shape (N, M).

    For each source i (row of Gamma):
        mass_sent_to_real_targets[i] = sum_j Gamma[i, j]
        mass_stays_at_source[i]      = a[i] - mass_sent_to_real_targets[i]

      mass_stays_at_source[i] is exactly the part that went to the
      target-side dustbin in the augmented problem, which we now keep
      at X[i].

    For each target j (column of Gamma):
        mass_on_target[j] = sum_i Gamma[i, j]

      This is the mass from *real* sources that ends up on Y[j]
      (we do NOT include any mass coming from the source-side dustbin).

    Output measure
    --------------
    We construct a measure (Z, w) on the union of coordinates from X and Y:

      1) Start from concatenation:
             Z_raw = [X; Y]                  (shape (N+M, 2))
             w_raw = [mass_on_X, mass_on_Y] (shape (N+M,))

      2) Remove small-mass / zero-mass points:
             mask = w_raw > mass_tol

      3) Optionally merge duplicates (exact coordinate equality):
             coords_unique, inv = np.unique(Z_nz, axis=0, return_inverse=True)
             w_merged[k] = sum_{i: inv[i]==k} w_nz[i]

         The resulting support has size ≤ N+M.

    Parameters
    ----------
    X : ndarray, shape (N, 2)
        Source points.
    Y : ndarray, shape (M, 2)
        Target points.
    a : ndarray, shape (N,), optional
        Source weights. If None, uniform over X.
    b : ndarray, shape (M,), optional
        Target weights. Passed through to `_balanced_ot_with_dustbin_2d`.
    dust_cost, cost, axis_weights :
        Passed directly to `_balanced_ot_with_dustbin_2d`.
    mass_tol : float, default 0.0
        Threshold below which masses are considered zero and removed.
        Example: use 1e-12 to drop numerical noise.
    merge_duplicates : bool, default True
        If True, merge identical coordinates in the union X ∪ Y.
        If False, you get the raw concatenation with zeros removed
        but without merging identical coordinates.
    return_plan : bool, default False
        If True, also return Gamma_full (N, M).

    Returns
    -------
    Z : ndarray, shape (K, 2)
        Coordinates of the 'near target' transported source, with K ≤ N+M.
    w : ndarray, shape (K,)
        Corresponding masses (all > mass_tol).
    Gamma_full : ndarray, shape (N, M), optional
        The real-real plan from `_balanced_ot_with_dustbin_2d`, only if
        return_plan=True.
    """
    # I keep dtype=int because your code uses integer coords everywhere here.
    # If you prefer, you can switch to float to match POT's usual conventions.
    X = np.asarray(X, dtype=int)
    Y = np.asarray(Y, dtype=int)

    N = X.shape[0]  # number of source points
    M = Y.shape[0]  # number of target points

    # 1) Compute the real-real transport plan with your existing function
    Gamma_full = _balanced_ot_with_dustbin_2d(
        X,
        Y,
        a=a,
        b=b,
        dust_cost=dust_cost,
        cost=cost,
        axis_weights=axis_weights,
    )  # shape (N, M)

    # 2) Reconstruct source weights with the SAME convention:
    if a is None:
        source_weights = np.ones(N, dtype=float)
    else:
        source_weights = np.asarray(a, dtype=float)

    # 3) Mass sent from each source to real targets
    mass_sent_to_targets = Gamma_full.sum(axis=1)  # shape (N,)

    # 4) Mass that stays at each source (dustbin part reinterpreted)
    mass_on_X = source_weights - mass_sent_to_targets
    # Clamp small negative values due to numerical issues
    mass_on_X = np.clip(mass_on_X, 0.0, None)

    # 5) Mass that ends up at each target from real sources
    mass_on_Y = Gamma_full.sum(axis=0)  # shape (M,)

    # 6) Concatenate supports and weights
    Z_raw = np.vstack([X, Y])  # (N+M, 2)
    w_raw = np.concatenate([mass_on_X, mass_on_Y])  # (N+M,)

    # 7) Remove zero / tiny masses
    if mass_tol > 0.0:
        mask = w_raw > mass_tol
    else:
        mask = w_raw > 0.0

    Z_nz = Z_raw[mask]
    w_nz = w_raw[mask]

    # 8) Merge duplicates in the union (exact equality)
    if merge_duplicates and Z_nz.shape[0] > 0:
        coords_unique, inv = np.unique(Z_nz, axis=0, return_inverse=True)
        w_merged = np.zeros(coords_unique.shape[0], dtype=float)
        np.add.at(w_merged, inv, w_nz)
        Z_out, w_out = coords_unique, w_merged
    else:
        Z_out, w_out = Z_nz, w_nz

    if return_plan:
        return Z_out, w_out, Gamma_full
    return Z_out, w_out


def ot_component(los, strips, dust_cost, dust_cost_comp, axis_weights, n_jobs):
    """
    Hierarchical 2D OT on connected components, per strip, per sample.

    Implementation notes
    --------------------
    - We never do in-place sparse slice assignment.
    - For each *sample* (except reference), we accumulate all transported
      (row, col, weight) triplets in GLOBAL coordinates and rebuild
      `sample.grid.data` once.
    - We rely on the fact that *all* nonzeros lie in strips; outside strips
      the matrix is (and remains) all zeros.
    """

    assert los.all_grids_standard(ref=True)
    ref_grid = los.ref_grid

    # Common shape for all samples
    n_rows, n_cols = ref_grid.data.shape

    # ---------------------------------------------------------------
    # 1) Precompute reference (target) clusters once per strip
    # ---------------------------------------------------------------
    strip_infos = []  # list of (strip_start, strip_end, ref_points_coo,
    #          ref_cluster_df, ref_cluster_to_points)

    for strip_start, strip_end in strips:
        ref_points_coo = ref_grid.data[:, strip_start:strip_end].tocoo()
        ref_cluster_df, ref_cluster_to_points = find_component(ref_points_coo)
        strip_infos.append(
            (
                strip_start,
                strip_end,
                ref_points_coo,
                ref_cluster_df,
                ref_cluster_to_points,
            )
        )

    # ---------------------------------------------------------------
    # 2) Process each non-reference sample independently
    #    (natural unit for parallelization)
    # ---------------------------------------------------------------
    def _process_single_sample(sample):
        """Process one sample (non-reference) in place."""
        all_rows = []
        all_cols = []
        all_data = []

        # ---- Per-strip hierarchical OT ------------------------------------
        for (
            strip_start,
            strip_end,
            ref_points_coo,
            ref_cluster_df,
            ref_cluster_to_points,
        ) in strip_infos:
            # Sample clusters on this strip
            sample_points_coo = sample.grid.data[:, strip_start:strip_end].tocoo()
            sample_cluster_df, sample_cluster_to_points = find_component(
                sample_points_coo
            )
            N = len(sample_cluster_df)  # # source clusters
            M = len(ref_cluster_df)  # # target clusters

            if N == 0 or M == 0:
                continue

            # Cluster-level OT: source = sample clusters, target = ref clusters
            transport_plan = _balanced_ot_with_dustbin_2d(
                sample_cluster_df[:, :-1],  # (row_mean, col_mean)
                ref_cluster_df[:, :-1],
                sample_cluster_df[:, -1],  # masses
                ref_cluster_df[:, -1],
                dust_cost=dust_cost_comp,
                axis_weights=axis_weights,
            )  # shape (N, M)

            if np.allclose(transport_plan, 0.0):
                continue

            # Adjacency on (targets | sources) from the OT plan
            adj = np.zeros((N + M, N + M), dtype=float)
            adj[:M, M:] = transport_plan.T  # target -> source
            adj[M:, :M] = transport_plan  # source -> target

            n_comp, labels = scipy.sparse.csgraph.connected_components(
                adj, directed=False
            )

            # ---- Point-level 2D OT within each connected component -------
            for comp_id in range(n_comp):
                comp_nodes = np.nonzero(labels == comp_id)[0]
                source_cluster_indices = [c for c in comp_nodes if c >= M]
                target_cluster_indices = [c for c in comp_nodes if c < M]

                if not source_cluster_indices and not target_cluster_indices:
                    continue

                source_point_indices = [
                    pt
                    for c in source_cluster_indices
                    for pt in sample_cluster_to_points(c - M)
                ]
                target_point_indices = [
                    pt
                    for c in target_cluster_indices
                    for pt in ref_cluster_to_points(c)
                ]

                if not source_point_indices and not target_point_indices:
                    continue

                # (row, local_col, intensity) arrays for points in this component
                source = np.array(
                    [
                        sample_points_coo.row[source_point_indices],
                        sample_points_coo.col[source_point_indices],
                        sample_points_coo.data[source_point_indices],
                    ]
                ).T
                target = np.array(
                    [
                        ref_points_coo.row[target_point_indices],
                        ref_points_coo.col[target_point_indices],
                        ref_points_coo.data[target_point_indices],
                    ]
                ).T

                # 2D OT-with-dustbin at point level within the component
                tpt_coord, tpt_weights = _balanced_ot_near_target_source_2d(
                    source[:, :-1],  # (row, local_col)
                    target[:, :-1],
                    source[:, -1],  # source intensities
                    target[:, -1],  # target intensities
                    dust_cost=dust_cost,
                    axis_weights=[1, 15],
                )

                # tpt_coord[:, 0] -> row index
                # tpt_coord[:, 1] -> *local* col index in [0, strip_end-strip_start)
                if tpt_weights.size == 0:
                    continue

                # Accumulate in GLOBAL column coordinates
                all_rows.append(tpt_coord[:, 0])
                all_cols.append(tpt_coord[:, 1] + strip_start)
                all_data.append(tpt_weights)

        # -----------------------------------------------------------
        # 3) Rebuild this sample's grid from all transported triplets
        # -----------------------------------------------------------
        if all_rows:
            rows = np.concatenate(all_rows).astype(int, copy=False)
            cols = np.concatenate(all_cols).astype(int, copy=False)
            data = np.concatenate(all_data).astype(float, copy=False)
        else:
            # No mass anywhere -> all zeros
            rows = np.array([], dtype=int)
            cols = np.array([], dtype=int)
            data = np.array([], dtype=float)

        new_grid = sp.coo_array(
            (data, (rows, cols)),
            shape=(n_rows, n_cols),
        ).tocsr()
        sample.grid.data = new_grid

    samples = los

    if n_jobs == 1:
        # Serial fallback (original behavior)
        for sample in samples:
            _process_single_sample(sample)
    else:
        # Parallel over samples (threading backend to avoid pickling issues
        # and to keep in-place updates on `sample` visible)
        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_process_single_sample)(sample) for sample in samples
        )


def hierarchical_ot(los, **params):

    strips, strip_masses = find_strip(
        los,
        params["min_zero"],
        optimal=True,
    )
    if strips is None or len(strips) == 0:
        return None

    ot_component(
        los,
        strips,
        dust_cost=params["dust_cost"],
        dust_cost_comp=params["dust_cost_comp"],
        axis_weights=params["axis_weights"],
        n_jobs=params.get("n_jobs", 1),
    )
