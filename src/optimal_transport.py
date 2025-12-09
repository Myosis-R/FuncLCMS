import cc3d
import matplotlib.pyplot as plt
import numpy as np
import ot
import scipy
import scipy.sparse as sp
from ot.unbalanced import sinkhorn_unbalanced
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


def plan_to_kernel(Gamma):
    """
    Convert a transport plan Gamma (shape: [n_source, n_target])
    into a kernel K(y|x) by normalizing each row.

    Parameters
    ----------
    Gamma : array_like, shape (n_source, n_target)
        Transport plan (non-negative entries).

    Returns
    -------
    K : ndarray, shape (n_source, n_target)
        Row-stochastic kernel: each row sums to 1, except rows that were all-zero
        in Gamma, which remain all-zero.
    """

    Gamma = np.asarray(Gamma, dtype=float)
    row_sums = Gamma.sum(axis=1, keepdims=True)  # shape (n_source, 1)
    # Initialize K as zeros and safely divide where row_sums > 0
    K = np.zeros_like(Gamma)
    np.divide(Gamma, row_sums, out=K, where=(row_sums > 0))
    return K


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


def mean_sum_by_labs(X, labs):  # NOTE: X multidim
    _ndx = np.argsort(labs)
    _id, _pos, g_count = np.unique(labs[_ndx], return_index=True, return_counts=True)

    g_sum = np.add.reduceat(X[_ndx], _pos, axis=0)
    g_sum[:, :-1] = g_sum[:, :-1] / g_count[:, None]
    return g_sum


def labs_to_dic(labs):  # NOTE: to avoid
    dic_labs = {}
    for i, l in enumerate(labs):
        dic_labs.setdefault(l, []).append(i)
    return dic_labs


def find_component(los, strips):
    # FIX: strip relative start
    # HACK: use window_ds for common axis

    for strip in strips:
        ref = ...
        for s in los:
            # FIX: zeros can be a component ? start at one
            clust = scipy.sparse.coo_array(
                cc3d.connected_components(
                    s.ds[:, strip[0] : strip[1]].todense(),
                    binary_image=True,
                    connectivity=8,
                ),
                return_N=True,
            )
            clust_df = mean_sum_by_labs(
                np.array(
                    [clust.row, clust.col, s.ds[:, strip[0] : strip[1]].tocoo().data]
                ),
                clust.data,
            )
            sol = ot.solve_sample(
                clust_df[:, :-1],
                coord_ref,
                clust_df[:, -1],
                int_ref,
                reg=1e-1,
                lazy=True,
            )
            adj = np.zeros((N + M, N + M))
            adj[M:, N:] = sol.lazy_plan[:]
            adj[N:, M:] = adj[M:, N:].T
            n_comp, labs = scipy.sparse.scgraph.connected_components(
                adj, directed=False
            )

            for i in range(n_comp):
                # TODO: convert clusts to points
                # to =
                # from = clust.row[clust.data == i], clust.col[clust.data == i], s.ds.
                pass

        pass


def hierarchical_ot(los, **params):

    strips, sum_strips = find_strip(los, params["length"])
    find_component(los, strips)

    # scale_strip
    # scale_component
    # scale_point
    pass
