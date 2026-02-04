import numpy as np
import ot
import scipy
import scipy.sparse as sp
from joblib import Parallel, delayed

from .strip_utils import find_strip

try:
    import connected_component  # pybind11 module built from connected_component.cpp
except ImportError as e:
    raise ImportError(
        "Failed to import the pybind11 module 'connected_component'. "
        "Build/install it so that `import connected_component` works."
    ) from e


def _clean_sorted_csc(block):
    """
    Prepare a CSC matrix that satisfies C++ preconditions:

    C++ requires:
      - CSC format
      - indptr monotone, indptr[0]=0, indptr[-1]=nnz (SciPy guarantees this)
      - within each column, indices must be STRICTLY increasing
        (=> we must remove duplicates and sort indices)
      - foreground is data > 0 (we drop <= 0 explicitly)

    Returns
    -------
    csc : scipy.sparse.csc_array/matrix
        Cleaned CSC with only positive data.
    """
    csc = block.tocsc(copy=True)

    # Merge duplicates (otherwise "strictly increasing indices" can fail)
    try:
        csc.sum_duplicates()
    except Exception:
        pass

    # Drop <= 0 because C++ considers foreground = data > 0
    if csc.nnz:
        pos = csc.data > 0
        if not np.all(pos):
            csc.data = csc.data * pos
            try:
                csc.eliminate_zeros()
            except Exception:
                # Older SciPy fallback
                csc = csc.tocsc()

    # Ensure indices are sorted per column (strictly increasing after duplicates merged)
    try:
        csc.sort_indices()
    except Exception:
        pass

    return csc


def _ccl_csc8(block):
    """
    Run C++ 8-connected components on a strip block (CSC).

    Returns a dict with:
      - stats: (K, 3) float64 [mean_row, mean_col, sum_intensity]
      - comp_indptr: (K+1,) int64
      - comp_indices: (nnz_fg,) int64 positions into CSC arrays
      - col_of_pos: (nnz,) int32 column index per CSC position
      - labels_nnz: (nnz,) int32 label per CSC position (optional)
    """
    csc = _clean_sorted_csc(block)
    n_rows = csc.shape[0]

    # Note: passing NumPy views of the SciPy arrays is fine; pybind sees them as 1D arrays.
    res = connected_component.ccl_csc8(
        csc.indptr,
        csc.indices,
        csc.data,
        n_rows,
    )
    return csc, res


def _gather_points_from_components(csc, res, comp_ids):
    """
    Convert a set of component ids into a point cloud (coords + weights).

    Parameters
    ----------
    csc : scipy.sparse.csc_*
    res : dict returned by ccl_csc8
    comp_ids : list[int]
        0-based component ids (Python convention).

    Returns
    -------
    coords : (P, 2) int array
        Columns are [row, col] in *local strip coordinates*.
    weights : (P,) float array
    """
    if len(comp_ids) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)

    comp_indptr = res["comp_indptr"]
    comp_indices = res["comp_indices"]
    col_of_pos = res["col_of_pos"]

    pos_chunks = []
    for k in comp_ids:
        start = comp_indptr[k]
        end = comp_indptr[k + 1]
        if end > start:
            pos_chunks.append(comp_indices[start:end])

    if not pos_chunks:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)

    pos = np.concatenate(pos_chunks)

    rows = csc.indices[pos].astype(int, copy=False)
    cols = col_of_pos[pos].astype(int, copy=False)
    w = csc.data[pos].astype(float, copy=False)

    coords = np.stack([rows, cols], axis=1)
    return coords, w


# ---------------------------------------------------------------------
# OT routines (unchanged)
# ---------------------------------------------------------------------
def _balanced_ot_with_dustbin_2d(
    X,
    Y,
    a=None,
    b=None,
    dust_cost=1.0,
    cost="sqeuclidean",
    axis_weights=None,
):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"X must have shape (N, 2), got {X.shape}")
    if Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError(f"Y must have shape (M, 2), got {Y.shape}")

    N = X.shape[0]
    M = Y.shape[0]

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

    active_source = source_weights > 0.0
    active_target = target_weights > 0.0

    idx_source = np.nonzero(active_source)[0]
    idx_target = np.nonzero(active_target)[0]

    k_source = idx_source.size
    k_target = idx_target.size
    if k_source == 0 or k_target == 0:
        return np.zeros((N, M), dtype=float)

    X_act = X[idx_source]
    Y_act = Y[idx_target]
    a_act = source_weights[idx_source]
    b_act = target_weights[idx_target]

    sa = float(a_act.sum())
    sb = float(b_act.sum())
    if sa <= 0.0 and sb <= 0.0:
        return np.zeros((N, M), dtype=float)

    a_ext = np.concatenate([a_act, [sb]]).astype(float)
    b_ext = np.concatenate([b_act, [sa]]).astype(float)

    a_ext[~np.isfinite(a_ext)] = 0.0
    b_ext[~np.isfinite(b_ext)] = 0.0
    a_ext = np.clip(a_ext, 0.0, None)
    b_ext = np.clip(b_ext, 0.0, None)

    mass_a = float(a_ext.sum())
    mass_b = float(b_ext.sum())
    if mass_a <= 0.0 and mass_b <= 0.0:
        return np.zeros((N, M), dtype=float)
    if mass_a <= 0.0 or mass_b <= 0.0:
        return np.zeros((N, M), dtype=float)

    mass_ext = 0.5 * (mass_a + mass_b)
    a_ext /= mass_ext
    b_ext /= mass_ext

    if cost != "sqeuclidean":
        raise ValueError(f"Unsupported cost: {cost!r}")

    if axis_weights is None:  # FIX: change axis_weights effect before looping ?
        w = np.ones(2, dtype=float)
    else:
        w = np.asarray(axis_weights, dtype=float).ravel()
        if w.shape[0] != 2:
            raise ValueError(f"axis_weights must have length 2, got {w.shape[0]}")
        if np.any(w < 0):
            raise ValueError("axis_weights must be non-negative")

    sqrt_w = np.sqrt(w)
    X_aniso = X_act * sqrt_w
    Y_aniso = Y_act * sqrt_w

    coords_all = np.vstack([X_aniso, Y_aniso])
    bb_min = coords_all.min(axis=0)
    bb_max = coords_all.max(axis=0)
    diag = np.linalg.norm(bb_max - bb_min)
    scale = max(1.0, float(diag))

    X_scaled = X_aniso / scale
    Y_scaled = Y_aniso / scale

    X = X_scaled.astype(np.float32, copy=False)
    Y = Y_scaled.astype(np.float32, copy=False)

    X2 = np.sum(X * X, axis=1, dtype=np.float32)[:, None]
    Y2 = np.sum(Y * Y, axis=1, dtype=np.float32)[None, :]
    C_real32 = X2 + Y2 - np.float32(2.0) * (X @ Y.T)

    dust_cost_scaled = dust_cost / (scale**2)

    C_ext = np.empty((k_source + 1, k_target + 1), dtype=float)
    C_ext[:k_source, :k_target] = C_real32
    C_ext[:k_source, k_target] = dust_cost_scaled
    C_ext[k_source, :k_target] = dust_cost_scaled
    C_ext[k_source, k_target] = 0.0

    Gamma_ext = ot.emd(a_ext, b_ext, C_ext)
    Gamma_ext *= mass_ext

    Gamma_act = np.asarray(Gamma_ext[:k_source, :k_target], dtype=float)

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
    X = np.asarray(X, dtype=int)
    Y = np.asarray(Y, dtype=int)

    N = X.shape[0]
    M = Y.shape[0]

    Gamma_full = _balanced_ot_with_dustbin_2d(
        X,
        Y,
        a=a,
        b=b,
        dust_cost=dust_cost,
        cost=cost,
        axis_weights=axis_weights,
    )

    if a is None:
        source_weights = np.ones(N, dtype=float)
    else:
        source_weights = np.asarray(a, dtype=float)

    mass_sent_to_targets = Gamma_full.sum(axis=1)
    mass_on_X = np.clip(source_weights - mass_sent_to_targets, 0.0, None)
    mass_on_Y = Gamma_full.sum(axis=0)

    Z_raw = np.vstack([X, Y])
    w_raw = np.concatenate([mass_on_X, mass_on_Y])

    if mass_tol > 0.0:
        mask = w_raw > mass_tol
    else:
        mask = w_raw > 0.0

    Z_nz = Z_raw[mask]
    w_nz = w_raw[mask]

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


# ---------------------------------------------------------------------
# Hierarchical OT using C++ connected components
# ---------------------------------------------------------------------
def ot_component(los, strips, dust_cost, dust_cost_comp, axis_weights, n_jobs):
    """
    Hierarchical 2D OT on connected components, per strip, per sample.

    High-level logic (same as your current hot.py):
      1) For each strip, compute connected components in the reference (once).
      2) For each sample, for each strip:
           a) compute sample components
           b) OT between *components* (using their centroids + masses)
           c) build a bipartite graph (sample comps <-> ref comps) from OT edges
           d) take graph connected-components = "groups to solve jointly"
           e) inside each group, do OT again at the *pixel/point* level
      3) Rebuild the sample sparse grid from transported triplets.

    Notes:
      - We use the C++ CCL module, so we never create a dense boolean image.
      - Component membership is read from (comp_indptr, comp_indices).
    """
    assert los.all_grids_standard(ref=True)

    ref_grid = los.ref_grid
    n_rows, n_cols = ref_grid.data.shape

    # ---------------------------------------------------------------
    # 1) Precompute reference connected components per strip
    # ---------------------------------------------------------------
    strip_infos = []
    for strip_start, strip_end in strips:
        ref_block = ref_grid.data[:, strip_start:strip_end]
        ref_csc, ref_res = _ccl_csc8(ref_block)
        ref_stats = np.asarray(ref_res["stats"])  # (M, 3)
        strip_infos.append((strip_start, strip_end, ref_csc, ref_res, ref_stats))

    # ---------------------------------------------------------------
    # 2) Process one sample (can be parallelized)
    # ---------------------------------------------------------------
    def _process_single_sample(sample):
        all_rows = []
        all_cols = []
        all_data = []

        for strip_start, strip_end, ref_csc, ref_res, ref_stats in strip_infos:
            # --- sample components on this strip
            sample_block = sample.grid.data[:, strip_start:strip_end]
            sample_csc, sample_res = _ccl_csc8(sample_block)
            sample_stats = np.asarray(sample_res["stats"])  # (N, 3)

            N = sample_stats.shape[0]  # source components
            M = ref_stats.shape[0]  # target components
            if N == 0:
                continue

            if M == 0:
                # Keep original sample strip mass (positive entries only)
                sample_coo = sample_block.tocoo()
                pos = sample_coo.data > 0
                if np.any(pos):
                    all_rows.append(sample_coo.row[pos].astype(int, copy=False))
                    all_cols.append(
                        (sample_coo.col[pos] + strip_start).astype(int, copy=False)
                    )
                    all_data.append(sample_coo.data[pos].astype(float, copy=False))
                continue

            # --- (a) Component-level OT (centroids + masses)
            transport_plan = _balanced_ot_with_dustbin_2d(
                sample_stats[:, :2],  # (row_mean, col_mean)
                ref_stats[:, :2],
                sample_stats[:, 2],  # masses
                ref_stats[:, 2],
                dust_cost=dust_cost_comp,
                axis_weights=axis_weights,
            )  # (N, M)

            if np.allclose(transport_plan, 0.0):
                continue

            # --- (b) Build a graph linking ref comps <-> sample comps using OT edges
            # Nodes: [0..M-1] are ref comps, [M..M+N-1] are sample comps
            adj = np.zeros((N + M, N + M), dtype=float)
            adj[:M, M:] = transport_plan.T
            adj[M:, :M] = transport_plan

            n_groups, group_labels = scipy.sparse.csgraph.connected_components(
                adj, directed=False
            )

            # --- (c) For each group, run point-level OT on the union of pixels
            for gid in range(n_groups):
                nodes = np.nonzero(group_labels == gid)[0]

                ref_comp_ids = [int(x) for x in nodes if x < M]  # 0..M-1
                src_comp_ids = [int(x - M) for x in nodes if x >= M]  # 0..N-1

                if len(src_comp_ids) == 0:
                    continue
                # Gather pixels for this group (local strip coords)
                src_xy, src_w = _gather_points_from_components(
                    sample_csc, sample_res, src_comp_ids
                )
                ref_xy, ref_w = _gather_points_from_components(
                    ref_csc, ref_res, ref_comp_ids
                )

                tpt_coord, tpt_weights = _balanced_ot_near_target_source_2d(
                    src_xy,
                    ref_xy,
                    src_w,
                    ref_w,
                    dust_cost=dust_cost,
                    axis_weights=axis_weights,
                )

                if tpt_weights.size == 0:
                    continue

                # Convert local strip cols -> global cols
                all_rows.append(tpt_coord[:, 0])
                all_cols.append(tpt_coord[:, 1] + strip_start)
                all_data.append(tpt_weights)

        # -----------------------------------------------------------
        # 3) Rebuild sample grid once from transported triplets
        # -----------------------------------------------------------
        if all_rows:
            rows = np.concatenate(all_rows).astype(int, copy=False)
            cols = np.concatenate(all_cols).astype(int, copy=False)
            data = np.concatenate(all_data).astype(float, copy=False)
        else:
            rows = np.array([], dtype=int)
            cols = np.array([], dtype=int)
            data = np.array([], dtype=float)

        new_grid = sp.coo_array((data, (rows, cols)), shape=(n_rows, n_cols)).tocsr()

        # Optional cleanup (robustness)
        try:
            new_grid.sum_duplicates()
        except Exception:
            pass
        try:
            new_grid.eliminate_zeros()
        except Exception:
            pass

        sample.grid.data = new_grid

    # ---------------------------------------------------------------
    # 4) Run (serial or parallel)
    # ---------------------------------------------------------------
    samples = list(los)  # keep same behavior as your current file

    if n_jobs == 1:
        for sample in samples:
            _process_single_sample(sample)
    else:
        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_process_single_sample)(sample) for sample in samples
        )


def hierarchical_ot(los, **params):
    strips, _ = find_strip(los, params["min_zero"], optimal=True)
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
