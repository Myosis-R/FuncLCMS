from functools import partial

import cc3d
import matplotlib.pyplot as plt
import numpy as np
import ot
import scipy
import scipy.sparse as sp
from ptw import ptw
from scipy.optimize import minimize
from scipy.sparse import csr_array

# from SWGG import swgg


def swgg(los):  # NOTE: explain name and ref
    ...


def translation_f(t, ref, s):  # TODO: optim roll, exact result ?
    t, c = np.divmod(t, 1)
    s = c * np.roll(s, -t - 1) + (1 - c) * np.roll(s, -t)
    return np.linalg.norm(ref - s)


def translation_grad_tmz(los):  # TODO: change ref, add axis
    assert los.all_grids_standard()
    tics = np.array([s.grid.sum_along_axis(0, boolean=True)[1] for s in los])
    ref = tics[0]
    for i, s in enumerate(los[1:]):
        translation = minimize(translation_f, 0, args=(ref, tics[i]))[
            "x"
        ]  # TODO: bound
        s.grid.interpolate_axis(
            axis=1, out_coord=s.ds_coord[1], in_coord=(s.ds_coord[1] - translation)
        )
    assert los.all_grids_standard()


def zero_runs(a, min_zero):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    ranges = np.ravel(ranges[(ranges[:, 1] - ranges[:, 0]) > min_zero, :])
    assert len(ranges) > 0  # TODO: better test empty array
    ranges = np.concatenate(([0], ranges)) if ranges[0] != 0 else ranges[1:]
    ranges = np.concatenate((ranges, [len(a)])) if ranges[-1] != len(a) else ranges[:-1]
    return ranges.reshape((-1, 2))


def find_strip(los, min_zero, optimal=False, min_points=40_000):
    assert los.all_grids_standard()
    mean_tics = np.array([s.grid.sum_along_axis(1, boolean=True)[1] for s in los]).mean(
        axis=0
    )
    strips = zero_runs(mean_tics, min_zero=min_zero)
    sum_strips = np.array(
        [np.sum(mean_tics[strips[i, 0] : strips[i, 1]]) for i in range(len(strips))]
    )
    if optimal:  # FIX: avoid empty interval
        idx = np.searchsorted(
            np.cumsum(sum_strips), np.arange(0, sum_strips.sum(), min_points)
        )  # WARN: case of >40_000 strip
        strips = np.roll(np.roll(strips, 1)[idx, :], -1)  # HACK: need check

    return (strips, sum_strips)


#################### strip        #######################


def strip_ot(los, **params):
    """
    Align each strip independently along the first axis using 1D balanced OT
    with a dustbin cost (cost-threshold OT).

    Parameters in **params
    ----------------------
    min_zero : int
        Minimum run length of zeros to define strips (passed to find_strip).

    dust_cost : float
        Dustbin cost C_dust. Real-to-dustbin and dustbin-to-real moves cost
        C_dust, so going through the dustbin instead of direct transport
        costs 2 * C_dust. This acts as a distance threshold.

    binarize : bool, optional (default: False)
        If True, build the 1D histograms from presence/absence (X > 0)
        instead of intensities. If False, use the actual intensities.

    cost : {"sqeuclidean"}, optional (default: "sqeuclidean")
        Ground cost on the 1D row grid. Currently only "sqeuclidean"
        is implemented: C[i, j] = (i - j)**2.
    """

    # Required parameters
    strips, _ = find_strip(los, params["min_zero"], optimal=True)
    dust_cost = params["dust_cost"]

    # Optional parameters
    binarize = params.get("binarize", False)
    cost = params.get("cost", "sqeuclidean")

    n_strips = len(strips)

    for k, strip in enumerate(strips):
        print(f"strip {k + 1}/{n_strips}")

        # Reference histogram from the first sample on this strip
        ref_block = los[0].grid.data[:, strip[0] : strip[1]]

        if binarize:
            ref = ref_block.astype(bool).sum(axis=1)
        else:
            ref = ref_block.sum(axis=1)

        # Make sure this is a 1D float array
        ref = np.asarray(ref).ravel().astype(float)

        # Align each sample to that reference
        for s in los:
            block = s.grid.data[:, strip[0] : strip[1]]

            _, aligned_block = ot_align_1d(
                block,
                ref=ref,
                dust_cost=dust_cost,
                cost=cost,
                binarize=binarize,
            )

            # In-place replacement of the strip data
            s.grid.data[:, strip[0] : strip[1]] = aligned_block

    # TODO: check that inplace works for your grid storage format (csc/csr/etc.)


def ot_align_1d(
    X,
    ref,
    dust_cost,
    cost="sqeuclidean",
    binarize=False,
):
    """
    Align a 2D matrix along rows using 1D *balanced* OT with a dustbin
    (cost-threshold OT) and warp X using the full OT plan.

    The OT is solved on the 1D row histograms:
      - source: a[i] = row sum of X[i, :]
      - target: b[i] = ref[i]

    We construct an augmented problem with one dustbin index d:

      - a_ext = [a, sum(b)]
      - b_ext = [b, sum(a)]

      - For real i, real j:     C[i, j]   = (i - j)**2  (if cost == "sqeuclidean")
      - For real i, dustbin d:  C[i, d]   = dust_cost
      - For dustbin d, real j:  C[d, j]   = dust_cost
      - For dustbin d, d:       C[d, d]   = 0

    Solving balanced EMD on (a_ext, b_ext, C_ext) yields a plan Gamma_ext.
    We then restrict to the real-real block Gamma_real and warp X as follows:

      - matched_mass_i = sum_j Gamma_real[i, j]
      - alpha_i = matched_mass_i / a[i]   (fraction of row i to move)
      - matched part (alpha_i * row_i) is redistributed to targets j
        with weights Gamma_real[i, j] / matched_mass_i
      - unmatched part ((1 - alpha_i) * row_i) stays at row i.

    Parameters
    ----------
    X : array-like or scipy.sparse matrix, shape (n_rows, n_cols)
        Source matrix to be warped along rows.
    ref : array_like, shape (n_rows,)
        Target 1D histogram along the row axis (non-negative).
    dust_cost : float
        Dustbin cost C_dust used in the augmented cost matrix.
    cost : {"sqeuclidean"}, optional
        Ground cost on the 1D row grid. Currently only "sqeuclidean"
        is implemented: C[i, j] = (i - j)**2.
    binarize : bool, optional (default: False)
        If True, build histograms from presence/absence (X > 0)
        instead of intensities.

    Returns
    -------
    Gamma_real : ndarray, shape (n_rows, n_rows)
        OT plan restricted to real rows/cols (zeros on dustbin).
    X_aligned : csr_array, shape (n_rows, n_cols)
        Row-warped matrix. Matched mass moves according to Gamma_real;
        unmatched mass stays at its original row.
    """

    # Ensure CSR sparse array
    if isinstance(X, csr_array):
        X_csr = X.copy()
    elif sp.issparse(X):
        X_csr = csr_array(X)
    else:
        X_csr = csr_array(X)

    n_rows, _ = X_csr.shape

    # Source histogram a
    if binarize:
        X_mask = X_csr.copy()
        if X_mask.nnz > 0:
            X_mask.data[:] = 1.0
        a = np.asarray(X_mask.sum(axis=1)).ravel().astype(float)
    else:
        a = np.asarray(X_csr.sum(axis=1)).ravel().astype(float)

    if np.any(a < 0):
        raise ValueError("Row sums of X must be non-negative")

    # Target histogram b
    b = np.asarray(ref, dtype=float).ravel()
    if b.shape[0] != n_rows:
        raise ValueError(f"ref must have length {n_rows}, got {b.shape[0]}")
    if np.any(b < 0):
        raise ValueError("ref must be non-negative")

    # Solve balanced OT with an explicit dustbin
    Gamma_real = _balanced_ot_with_dustbin_1d(a, b, dust_cost=dust_cost, cost=cost)

    # If there is effectively no transport, return X as-is
    if np.allclose(Gamma_real, 0.0):
        return Gamma_real, X_csr

    # Warp rows using the OT plan; unmatched mass stays in place
    X_aligned = _warp_rows_by_plan_csr(X_csr, a, Gamma_real)

    # TEST:
    fig, ax = plt.subplots()
    x = np.arange(len(a))
    ax.plot(x, a, "b-", label="source")
    ax.plot(x, b, "r-", label="target")
    ax.plot(x, X_aligned.sum(axis=1), "k-", label="aligned")
    plt.legend()
    plt.show()

    return Gamma_real, X_aligned


def _balanced_ot_with_dustbin_1d(a, b, dust_cost, cost="sqeuclidean"):
    """
    Solve 1D balanced OT with an explicit dustbin using Earth Mover's Distance,
    restricted to active indices (a[i] > 0 or b[i] > 0).

    Implementation details
    ----------------------
    - We only build the OT problem on the active support:
        active = (a > 0) | (b > 0)
      and work in that compressed index space.

    - Ground cost is squared distance along the *row index* axis, but we
      normalize indices to avoid very large values:

          coords = idx / scale,   with scale = max(1, n - 1)

      so that coords lie roughly in [0, 1] and (coords_i - coords_j)^2 <= 1.

    - To keep the semantics of dust_cost unchanged (it is interpreted in the
      original index^2 units), we scale it internally:

          dust_cost_scaled = dust_cost / scale**2

      The entire augmented cost matrix is therefore the original one divided
      by scale**2, which does NOT change the optimal OT plan.

    - To improve numerical stability of POT's LP solver, we normalize the
      augmented marginals a_ext, b_ext to the simplex (sum ≈ 1) before
      calling ot.emd, then rescale the resulting plan back to the original
      total mass.

    Augmented problem on the active support
    ---------------------------------------
    Let k = number of active indices.

    a_act, b_act in R^k_+ (compressed versions of a, b),
    sa = sum(a_act), sb = sum(b_act).

    a_ext = [a_act, sb]
    b_ext = [b_act, sa]

    Cost matrix C_ext (size (k+1, k+1)):

      - For p < k, q < k:
            C_ext[p, q] = (coords_p - coords_q)^2  (in normalized units)

      - For p < k, q = k:
            C_ext[p, k] = dust_cost_scaled         (real -> dustbin)
      - For p = k, q < k:
            C_ext[k, q] = dust_cost_scaled         (dustbin -> real)
      - For p = k, q = k:
            C_ext[k, k] = 0.0                      (dustbin -> dustbin)

    Then:

        Gamma_ext = ot.emd(a_ext, b_ext, C_ext)

    and we set Gamma_full[i, j] = Gamma_ext[p, q] for active indices
    i = idx[p], j = idx[q], and 0 elsewhere.

    Parameters
    ----------
    a : array_like, shape (n,)
        Source histogram (non-negative).
    b : array_like, shape (n,)
        Target histogram (non-negative).
    dust_cost : float
        Dustbin cost in the ORIGINAL (index^2) distance units.
    cost : {"sqeuclidean"}, optional
        Ground cost on the 1D row grid.

    Returns
    -------
    Gamma_full : ndarray, shape (n, n)
        Real-real OT plan on the full index set; rows/cols where a[i] = b[i] = 0
        are identically zero.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    n = a.shape[0]
    if b.shape[0] != n:
        raise ValueError(f"a and b must have the same length, got {n} and {b.shape[0]}")

    # Active support: indices where there is some mass in a or b
    active = (a > 0.0) | (b > 0.0)
    idx = np.nonzero(active)[0]
    k = idx.size

    # If no active point, nothing to transport
    if k == 0:
        return np.zeros((n, n), dtype=float)

    # Compressed histograms on the active support
    a_act = a[active]
    b_act = b[active]

    sa = float(a_act.sum())
    sb = float(b_act.sum())

    # If both are (numerically) empty, nothing to transport
    if sa <= 0.0 and sb <= 0.0:
        return np.zeros((n, n), dtype=float)

    # Augmented marginals on the active support
    a_ext = np.concatenate([a_act, [sb]]).astype(float)
    b_ext = np.concatenate([b_act, [sa]]).astype(float)

    # ---- Clean & normalize for POT (improves numerical stability) ----
    # Replace NaN / inf by 0
    a_ext[~np.isfinite(a_ext)] = 0.0
    b_ext[~np.isfinite(b_ext)] = 0.0

    # Clip tiny negative values (from numerical noise) to 0
    a_ext = np.clip(a_ext, 0.0, None)
    b_ext = np.clip(b_ext, 0.0, None)

    mass_a = float(a_ext.sum())
    mass_b = float(b_ext.sum())

    # If both sides are empty after cleaning: nothing to transport
    if mass_a <= 0.0 and mass_b <= 0.0:
        return np.zeros((n, n), dtype=float)

    # If only one side is empty, also return zeros (pathological input)
    if mass_a <= 0.0 or mass_b <= 0.0:
        return np.zeros((n, n), dtype=float)

    # They should be equal by construction; use their mean as robust scale
    mass_ext = 0.5 * (mass_a + mass_b)

    # Normalize to a probability simplex (sum ≈ 1)
    a_ext /= mass_ext
    b_ext /= mass_ext

    # Cost matrix on the active indices: use normalized original positions
    if cost == "sqeuclidean":
        # Normalize indices to keep squared distances ~ O(1)
        # This scaling is compensated in dust_cost so the OT plan is unchanged.
        scale = max(1.0, float(n - 1))
        coords = idx.astype(float) / scale
        C_real = (coords[:, None] - coords[None, :]) ** 2

        dust_cost_scaled = dust_cost / (scale**2)
    else:
        raise ValueError(f"Unsupported cost: {cost!r}")

    C_ext = np.empty((k + 1, k + 1), dtype=float)
    C_ext[:k, :k] = C_real
    C_ext[:k, k] = dust_cost_scaled  # real -> dustbin
    C_ext[k, :k] = dust_cost_scaled  # dustbin -> real
    C_ext[k, k] = 0.0  # dustbin -> dustbin

    # Balanced Earth Mover's Distance on the compressed + dustbin problem
    Gamma_ext = ot.emd(a_ext, b_ext, C_ext)

    # Rescale transport plan back to original total mass
    Gamma_ext *= mass_ext

    # Real-real block on the compressed support
    Gamma_act = np.asarray(Gamma_ext[:k, :k], dtype=float)

    # Expand back to full (n, n), filling inactive rows/cols with zeros
    Gamma_full = np.zeros((n, n), dtype=float)
    Gamma_full[np.ix_(idx, idx)] = Gamma_act

    return Gamma_full


def _warp_rows_by_plan_csr(X, a, Gamma, tol=1e-12):
    """
    Warp a CSR matrix X along rows according to transport plan Gamma.

    For each row i:
      - matched_mass_i = sum_j Gamma[i, j]
      - alpha_i = matched_mass_i / a[i]   (fraction of the row to move)
      - matched part (alpha_i * row_i) is redistributed to targets j
        with weights Gamma[i, j] / matched_mass_i
      - unmatched part ((1 - alpha_i) * row_i) stays at row i.

    Parameters
    ----------
    X : csr_array, shape (n_rows, n_cols)
        Source matrix.
    a : array_like, shape (n_rows,)
        Row sums of X used to compute the plan (same as in _balanced_ot_with_dustbin_1d).
    Gamma : array_like, shape (n_rows, n_rows)
        Real-real part of the OT plan.
    tol : float, optional
        Numerical tolerance for discarding tiny weights.

    Returns
    -------
    X_warped : csr_array, shape (n_rows, n_cols)
        Row-warped matrix.
    """
    X = csr_array(X)  # ensure CSR array
    n_rows, n_cols = X.shape

    a = np.asarray(a, dtype=float).ravel()
    Gamma = np.asarray(Gamma, dtype=float)

    indptr = X.indptr
    indices = X.indices
    data = X.data

    # Matched mass per row, and fraction alpha_i to move
    matched = Gamma.sum(axis=1)
    matched = np.asarray(matched).ravel()

    alpha = np.zeros(n_rows, dtype=float)
    valid = (a > 0.0) & (matched > 0.0)
    alpha[valid] = np.minimum(matched[valid] / a[valid], 1.0)

    # Split rows into matched and unmatched parts
    data_matched = data.copy()
    for i in range(n_rows):
        row_start, row_end = indptr[i], indptr[i + 1]
        if row_start == row_end:
            continue
        ai = alpha[i]
        if ai == 0.0:
            data_matched[row_start:row_end] = 0.0
        elif ai != 1.0:
            data_matched[row_start:row_end] *= ai

    data_unmatched = data - data_matched

    # Unmatched part: stays at original rows
    X_unmatched = csr_array(
        (data_unmatched, indices.copy(), indptr.copy()), shape=X.shape
    )

    # Matched part: transported according to Gamma
    rows_list = []
    cols_list = []
    vals_list = []

    for i in range(n_rows):
        ai = alpha[i]
        if ai <= 0.0:
            continue

        row_start, row_end = indptr[i], indptr[i + 1]
        if row_start == row_end:
            continue

        gi = Gamma[i, :]
        mass_i = gi.sum()
        if mass_i <= tol:
            continue

        w = gi / mass_i
        dest = np.nonzero(w > tol)[0]
        if dest.size == 0:
            continue

        row_cols = indices[row_start:row_end]
        row_vals = data_matched[row_start:row_end]

        for j in dest:
            wij = w[j]
            if wij <= 0.0:
                continue
            rows_list.append(np.full(row_cols.shape, j, dtype=int))
            cols_list.append(row_cols)
            vals_list.append(row_vals * wij)

    if rows_list:
        rows_cat = np.concatenate(rows_list)
        cols_cat = np.concatenate(cols_list)
        vals_cat = np.concatenate(vals_list)
        X_moved = csr_array((vals_cat, (rows_cat, cols_cat)), shape=X.shape)
        X_moved.sum_duplicates()
    else:
        # No matched mass transported
        X_moved = csr_array((n_rows, n_cols))

    return X_unmatched + X_moved


#################### hierarchical #######################


def g_f(id, _ndx, _pos):
    return _ndx[_pos[id] : _pos[id + 1]]


def mean_sum_by_labs(X, labs):  # NOTE: X multidim
    _ndx = np.argsort(labs)
    _id, _pos, g_count = np.unique(labs[_ndx], return_index=True, return_counts=True)

    g_sum = np.add.reduceat(X[_ndx], _pos, axis=0)
    g_sum[:, :-1] = g_sum[:, :-1] / g_count[:, None]
    g_fp = partial(g_f, _ndx=_ndx, _pos=_pos)
    return g_sum, g_fp


def find_component(ds):
    clust = scipy.sparse.coo_array(
        cc3d.connected_components(
            ds.todense(),
            binary_image=True,
            connectivity=8,
            return_N=True,
        )[0],
    )
    clust_df, clust_func = mean_sum_by_labs(
        np.array([clust.row, clust.col, ds.tocoo().data]).T,
        clust.data,
    )
    return clust_df, clust_func


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

    2D setting
    ----------
    - X: source points in R^2, shape (M, 2)
    - Y: target points in R^2, shape (N, 2)
    - a: source weights on X, shape (M,)
    - b: target weights on Y, shape (N,)

      If a or b is None, we use uniform weights on the corresponding cloud.

    - Cost between two real points is squared Euclidean distance in R^2
      (after a coordinate renormalization for numerical stability).

    Active support
    --------------
    We only keep points with strictly positive mass on each side:

        active_a = (a > 0)
        active_b = (b > 0)

    then work in the compressed index spaces:

        X_act, a_act  (size k_a)
        Y_act, b_act  (size k_b)

    Dustbin construction (unbalanced → balanced)
    -------------------------------------------
    Let sa = sum(a_act), sb = sum(b_act).

    We build augmented marginals:

        a_ext = [a_act, sb]   in R^{k_a + 1}_+
        b_ext = [b_act, sa]   in R^{k_b + 1}_+

    and an augmented cost matrix C_ext of shape (k_a + 1, k_b + 1):

      - For i < k_a, j < k_b:   C_ext[i, j] = ||x_i - y_j||^2  (in normalized units)
      - For i < k_a, j = k_b:  C_ext[i, k_b] = dust_cost_scaled   (real -> dustbin)
      - For i = k_a, j < k_b:  C_ext[k_a, j] = dust_cost_scaled   (dustbin -> real)
      - For i = k_a, j = k_b:  C_ext[k_a, k_b] = 0.0              (dustbin -> dustbin)

    Then:

        Gamma_ext = ot.emd(a_ext, b_ext, C_ext)

    and we keep the real-real block Gamma_ext[:k_a, :k_b], which we expand
    back to a full (M, N) plan with zeros on inactive rows/cols.

    Coordinate normalization & dust_cost scaling
    --------------------------------------------
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

    The entire augmented cost matrix is therefore the original one divided
    by scale**2, which does NOT change the optimal OT plan.

    Marginal normalization for POT
    ------------------------------
    As in the 1D version, we:

      - clean NaN / inf,
      - clip tiny negatives to 0,
      - normalize a_ext and b_ext to sum ≈ 1 before calling ot.emd,
      - then rescale Gamma_ext by the original total mass (mass_ext).

    Parameters
    ----------
    X : array_like, shape (M, 2)
        Source point cloud in R^2.
    Y : array_like, shape (N, 2)
        Target point cloud in R^2.
    a : array_like, shape (M,), optional
        Source weights (non-negative). If None, uses uniform weights.
    b : array_like, shape (N,), optional
        Target weights (non-negative). If None, uses uniform weights.
    dust_cost : float
        Dustbin cost in the ORIGINAL (coordinate^2) distance units.
    cost : {"sqeuclidean"}, optional
        Ground cost on R^2.

    Returns
    -------
    Gamma_full : ndarray, shape (M, N)
        Real-real OT plan; rows/cols corresponding to zero-mass points
        are identically zero.
    """
    # --- Inputs and basic checks ------------------------------------------------
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"X must have shape (M, 2), got {X.shape}")
    if Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError(f"Y must have shape (N, 2), got {Y.shape}")

    M = X.shape[0]
    N = Y.shape[0]

    if a is None:
        a = np.ones(M, dtype=float)
    else:
        a = np.asarray(a, dtype=float).ravel()
        if a.shape[0] != M:
            raise ValueError(f"a must have length {M}, got {a.shape[0]}")

    if b is None:
        b = np.ones(N, dtype=float)
    else:
        b = np.asarray(b, dtype=float).ravel()
        if b.shape[0] != N:
            raise ValueError(f"b must have length {N}, got {b.shape[0]}")

    # --- Active support on each side -------------------------------------------
    active_a = a > 0.0
    active_b = b > 0.0

    idx_a = np.nonzero(active_a)[0]
    idx_b = np.nonzero(active_b)[0]

    k_a = idx_a.size
    k_b = idx_b.size

    # If no active point on at least one side: nothing meaningful to transport
    if k_a == 0 or k_b == 0:
        return np.zeros((M, N), dtype=float)

    X_act = X[idx_a]
    Y_act = Y[idx_b]
    a_act = a[idx_a]
    b_act = b[idx_b]

    sa = float(a_act.sum())
    sb = float(b_act.sum())

    # If both are (numerically) empty, nothing to transport
    if sa <= 0.0 and sb <= 0.0:
        return np.zeros((M, N), dtype=float)

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
        return np.zeros((M, N), dtype=float)

    # If only one side is empty, also return zeros (pathological input)
    if mass_a <= 0.0 or mass_b <= 0.0:
        return np.zeros((M, N), dtype=float)

    # They should be equal by construction; use their mean as robust scale
    mass_ext = 0.5 * (mass_a + mass_b)

    # Normalize to a probability simplex (sum ≈ 1)
    a_ext /= mass_ext
    b_ext /= mass_ext

    # --- Cost matrix on active points (2D squared Euclidean) -------------------
    if cost == "sqeuclidean":
        # Axis weights: w_x, w_y  (default = 1,1 --> isotropic)
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
        X_aniso = X_act * sqrt_w  # (k_a, 2)
        Y_aniso = Y_act * sqrt_w  # (k_b, 2)

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

    # Augmented cost matrix with dustbin (k_a+1, k_b+1)
    C_ext = np.empty((k_a + 1, k_b + 1), dtype=float)
    C_ext[:k_a, :k_b] = C_real
    C_ext[:k_a, k_b] = dust_cost_scaled  # real -> dustbin
    C_ext[k_a, :k_b] = dust_cost_scaled  # dustbin -> real
    C_ext[k_a, k_b] = 0.0  # dustbin -> dustbin

    # --- Solve balanced EMD on the augmented problem ---------------------------
    Gamma_ext = ot.emd(a_ext, b_ext, C_ext)

    # Rescale transport plan back to original total mass
    Gamma_ext *= mass_ext

    # Real-real block on the compressed supports
    Gamma_act = np.asarray(Gamma_ext[:k_a, :k_b], dtype=float)

    # --- Expand back to full (M, N) plan ---------------------------------------
    Gamma_full = np.zeros((M, N), dtype=float)
    Gamma_full[np.ix_(idx_a, idx_b)] = Gamma_act

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
        Gamma = _balanced_ot_with_dustbin_2d(...), shape (M, N).

    For each source i:
        mass_sent_to_real_targets[i] = sum_j Gamma[i, j]
        mass_stays_at_source[i]      = a[i] - mass_sent_to_real_targets[i]

      mass_stays_at_source[i] is exactly the part that went to the
      target-side dustbin in the augmented problem, which we now keep
      at X[i].

    For each target j:
        mass_on_target[j] = sum_i Gamma[i, j]

      This is the mass from *real* sources that ends up on Y[j]
      (we do NOT include any mass coming from the source-side dustbin).

    Output measure
    --------------
    We construct a measure (Z, w) on the union of coordinates from X and Y:

      1) Start from concatenation:
             Z_raw = [X; Y]                (shape (M+N, 2))
             w_raw = [mass_on_X, mass_on_Y]  (shape (M+N,))

      2) Remove small-mass / zero-mass points:
             mask = w_raw > mass_tol

      3) Optionally merge duplicates (exact coordinate equality):
             coords_unique, inv = np.unique(Z_nz, axis=0, return_inverse=True)
             w_merged[k] = sum_{i: inv[i]==k} w_nz[i]

         The resulting support has size <= M+N.

    Parameters
    ----------
    X : ndarray, shape (M, 2)
        Source points.
    Y : ndarray, shape (N, 2)
        Target points.
    a : ndarray, shape (M,), optional
        Source weights. If None, uniform over X.
    b : ndarray, shape (N,), optional
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
        If True, also return Gamma_full (M, N).

    Returns
    -------
    Z : ndarray, shape (K, 2)
        Coordinates of the 'near target' transported source, with K ≤ M+N.
    w : ndarray, shape (K,)
        Corresponding masses (all > mass_tol).
    Gamma_full : ndarray, shape (M, N), optional
        The real-real plan from `_balanced_ot_with_dustbin_2d`, only if
        return_plan=True.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    M = X.shape[0]
    N = Y.shape[0]

    # 1) Compute the real-real transport plan with your existing function
    Gamma_full = _balanced_ot_with_dustbin_2d(
        X,
        Y,
        a=a,
        b=b,
        dust_cost=dust_cost,
        cost=cost,
        axis_weights=axis_weights,
    )  # shape (M, N)

    # 2) Reconstruct source weights a_vec with the SAME convention:
    if a is None:
        if M > 0:
            a_vec = np.ones(M, dtype=float) / M
        else:
            a_vec = np.zeros(0, dtype=float)
    else:
        a_vec = np.asarray(a, dtype=float)
        if a_vec.shape != (M,):
            raise ValueError(f"a must have shape ({M},), got {a_vec.shape}")

    # 3) Mass sent from each source to real targets
    mass_to_Y_per_source = Gamma_full.sum(axis=1)  # shape (M,)

    # 4) Mass that stays at each source (dustbin part reinterpreted)
    mass_on_X = a_vec - mass_to_Y_per_source
    # Clamp small negative values due to numerical issues
    mass_on_X = np.clip(mass_on_X, 0.0, None)

    # 5) Mass that ends up at each target from real sources
    mass_on_Y = Gamma_full.sum(axis=0)  # shape (N,)

    # 6) Concatenate supports and weights
    Z_raw = np.vstack([X, Y])  # (M+N, 2)
    w_raw = np.concatenate([mass_on_X, mass_on_Y])  # (M+N,)

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


def ot_component(los, strips, dust_cost, dust_cost_comp):
    assert los.all_grids_standard()

    for strip in strips:
        points_ref_ds = los[0].grid.data[:, strip[0] : strip[1]].tocoo()
        clust_ref_df, clust_ref_func = find_component(points_ref_ds)
        M = len(clust_ref_df)
        for s in los:
            points_ds = s.grid.data[:, strip[0] : strip[1]].tocoo()
            clust_df, clust_func = find_component(points_ds)
            N = len(clust_df)
            sol = _balanced_ot_with_dustbin_2d(
                clust_df[:, :-1],
                clust_ref_df[:, :-1],
                clust_df[:, -1],
                clust_ref_df[:, -1],
                dust_cost=dust_cost_comp,
                axis_weights=[1, 15],  # FIX:
            )
            adj = np.zeros((N + M, N + M))
            adj[:M, M:] = sol
            adj[M:, :M] = adj[:M, M:].T
            n_comp, labs = scipy.sparse.csgraph.connected_components(
                adj, directed=False
            )

            for i in range(n_comp):
                clust_i = np.nonzero(labs == i)  # TODO: optim
                # TODO: vectorize
                pt_idx = [pt for c in clust_i if c >= M for pt in clust_func(c)]
                pt_ref_idx = [pt for c in clust_i if c < M for pt in clust_ref_func(c)]
                # TODO: sparse type?
                source = np.array(
                    [
                        points_ds.tocoo().row[pt_idx],
                        points_ds.tocoo().col[pt_idx],
                        points_ds.tocoo().data[pt_idx],
                    ]
                ).T
                target = np.array(
                    [
                        points_ref_ds.tocoo().row[pt_ref_idx],
                        points_ref_ds.tocoo().col[pt_ref_idx],
                        points_ref_ds.tocoo().data[pt_ref_idx],
                    ]
                ).T
                tpt_coord, tpt_weights = _balanced_ot_near_target_source_2d(
                    source[:, :-1],
                    target[:, :-1],
                    source[:, -1],
                    target[:, -1],
                    dust_cost=dust_cost,
                    axis_weights=[1, 15],
                )
                s.grid.data[:, strip[0] : strip[1]] = sp.coo_array(
                    (tpt_weights, tpt_coord)
                ).tocsr()


def hierarchical_ot(los, **params):

    strips, sum_strips = find_strip(los, params["min_zero"], optimal=True)
    ot_component(los, strips, params["dust_cost"], params["dust_cost_comp"])

    # scale_strip
    # scale_component
    # scale_point
    pass
