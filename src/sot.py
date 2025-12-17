import numpy as np
import ot
import scipy.sparse as sp

from strip_utils import find_strip


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
        C_dust, so going through the dustbin instead of direct transport costs
        2 * C_dust. This acts as a distance threshold.
    binarize : bool, optional (default: False)
        If True, build the 1D histograms from presence/absence (X > 0)
        instead of intensities. If False, use the actual intensities.
    cost : {"sqeuclidean"}, optional (default: "sqeuclidean")
        Ground cost on the 1D row grid. Currently only "sqeuclidean"
        is implemented: C[i, j] = (i - j)**2.
    """
    assert los.all_grids_standard()

    # Required parameters
    strips, _ = find_strip(los, params["min_zero"], optimal=True)
    dust_cost = params["dust_cost"]

    # Optional parameters
    binarize = params.get("binarize", False)
    cost = params.get("cost", "sqeuclidean")

    num_strips = len(strips)

    for strip_index, (start, end) in enumerate(strips):
        print(f"strip {strip_index + 1}/{num_strips}")

        # Reference histogram from the first sample on this strip
        ref_block = los[0].grid.data[:, start:end]
        if binarize:
            ref_hist = ref_block.astype(bool).sum(axis=1)
        else:
            ref_hist = ref_block.sum(axis=1)

        # Make sure this is a 1D float array
        ref_hist = np.asarray(ref_hist).ravel().astype(float)

        # Align each sample to that reference
        for sample in los:
            block = sample.grid.data[:, start:end]
            _, aligned_block = ot_align_1d(
                block,
                ref=ref_hist,
                dust_cost=dust_cost,
                cost=cost,
                binarize=binarize,
            )
            # In-place replacement of the strip data
            sample.grid.data[:, start:end] = aligned_block

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
      - For real i, real j: C[i, j] = (i - j)**2 (if cost == "sqeuclidean")
      - For real i, dustbin d: C[i, d] = dust_cost
      - For dustbin d, real j: C[d, j] = dust_cost
      - For dustbin d, d:      C[d, d] = 0

    Solving balanced EMD on (a_ext, b_ext, C_ext) yields a plan Gamma_ext.
    We then restrict to the real-real block Gamma_real and warp X as follows:
      - matched_mass_i = sum_j Gamma_real[i, j]
      - alpha_i = matched_mass_i / a[i] (fraction of row i to move)
      - matched part (alpha_i * row_i) is redistributed to targets j with
        weights Gamma_real[i, j] / matched_mass_i
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
    X_aligned :sp.csr_array, shape (n_rows, n_cols)
        Row-warped matrix. Matched mass moves according to Gamma_real;
        unmatched mass stays at its original row.
    """
    # Ensure CSR sparse array
    if isinstance(X, sp.csr_array):
        source_matrix = X.copy()
    elif sp.issparse(X):
        source_matrix = sp.csr_array(X)
    else:
        source_matrix = sp.csr_array(X)

    n_rows, _ = source_matrix.shape

    # Source histogram (row sums of X)
    if binarize:
        presence_mask = source_matrix.copy()
        if presence_mask.nnz > 0:
            presence_mask.data[:] = 1.0
        source_hist = np.asarray(presence_mask.sum(axis=1)).ravel().astype(float)
    else:
        source_hist = np.asarray(source_matrix.sum(axis=1)).ravel().astype(float)

    if np.any(source_hist < 0):
        raise ValueError("Row sums of X must be non-negative")

    # Target histogram
    target_hist = np.asarray(ref, dtype=float).ravel()
    if target_hist.shape[0] != n_rows:
        raise ValueError(f"ref must have length {n_rows}, got {target_hist.shape[0]}")
    if np.any(target_hist < 0):
        raise ValueError("ref must be non-negative")

    # Solve balanced OT with an explicit dustbin
    transport_plan = _balanced_ot_with_dustbin_1d(
        source_hist, target_hist, dust_cost=dust_cost, cost=cost
    )

    # If there is effectively no transport, return X as-is
    if np.allclose(transport_plan, 0.0):
        return transport_plan, source_matrix

    # Warp rows using the OT plan; unmatched mass stays in place
    aligned_matrix = _warp_rows_by_plan_csr(source_matrix, source_hist, transport_plan)

    # Optional debug plot
    # fig, ax = plt.subplots()
    # x = np.arange(len(source_hist))
    # ax.plot(x, source_hist, "b-", label="source")
    # ax.plot(x, target_hist, "r-", label="target")
    # ax.plot(x, aligned_matrix.sum(axis=1), "k-", label="aligned")
    # plt.legend()
    # plt.show()

    return transport_plan, aligned_matrix


def _balanced_ot_with_dustbin_1d(
    source_hist, target_hist, dust_cost, cost="sqeuclidean"
):
    """
    Solve 1D balanced OT with an explicit dustbin using Earth Mover's Distance,
    restricted to active indices (source_hist[i] > 0 or target_hist[i] > 0).

    Implementation details
    ----------------------
    - We only build the OT problem on the active support:
          active = (source_hist > 0) | (target_hist > 0)
      and work in that compressed index space.
    - Ground cost is squared distance along the *row index* axis, but we
      normalize indices to avoid very large values:
          coords = active_indices / scale, with
          scale = max(1, n - 1)
      so that coords lie roughly in [0, 1] and
          (coords_i - coords_j)^2 <= 1.
    - To keep the semantics of dust_cost unchanged (interpreted in the
      original index^2 units), we scale it internally:
          dust_cost_scaled = dust_cost / scale**2
      The entire augmented cost matrix is therefore the original one divided
      by scale**2, which does NOT change the optimal OT plan.
    - To improve numerical stability of POT's LP solver, we normalize the
      augmented marginals to the simplex (sum ≈ 1) before calling ot.emd,
      then rescale the resulting plan back to the original total mass.

    Augmented problem on the active support
    ---------------------------------------
    Let n = len(source_hist) = len(target_hist).
        active_indices = {i : source_hist[i] > 0 or target_hist[i] > 0}
        k = |active_indices|

        source_active, target_active in R^k_+  (compressed histograms)
        source_mass = sum(source_active)
        target_mass = sum(target_active)

        source_aug = [source_active, target_mass]
        target_aug = [target_active, source_mass]

    Cost matrix C_ext (size (k+1, k+1)):
      - For p < k, q < k:
            C_ext[p, q] = (coords_p - coords_q)^2  (in normalized units)
      - For p < k, q = k:
            C_ext[p, k] = dust_cost_scaled        (real -> dustbin)
      - For p = k, q < k:
            C_ext[k, q] = dust_cost_scaled        (dustbin -> real)
      - For p = k, q = k:
            C_ext[k, k] = 0.0                     (dustbin -> dustbin)

    Then:
        Gamma_ext = ot.emd(source_aug, target_aug, C_ext)

    and we set Gamma_full[i, j] = Gamma_ext[p, q] for active indices
    i = active_indices[p], j = active_indices[q], and 0 elsewhere.

    Parameters
    ----------
    source_hist : array_like, shape (n,)
        Source histogram (non-negative).
    target_hist : array_like, shape (n,)
        Target histogram (non-negative).
    dust_cost : float
        Dustbin cost in the ORIGINAL (index^2) distance units.
    cost : {"sqeuclidean"}, optional
        Ground cost on the 1D row grid.

    Returns
    -------
    transport_full : ndarray, shape (n, n)
        Real-real OT plan on the full index set; rows/cols where
        source_hist[i] = target_hist[i] = 0 are identically zero.
    """
    source_hist = np.asarray(source_hist, dtype=float).ravel()
    target_hist = np.asarray(target_hist, dtype=float).ravel()

    num_bins = source_hist.shape[0]
    if target_hist.shape[0] != num_bins:
        raise ValueError(
            f"source_hist and target_hist must have the same length, "
            f"got {num_bins} and {target_hist.shape[0]}"
        )

    # Active support: indices where there is some mass in source or target
    active_bins_mask = (source_hist > 0.0) | (target_hist > 0.0)
    active_bins_idx = np.nonzero(active_bins_mask)[0]
    num_active_bins = active_bins_idx.size

    # If no active point, nothing to transport
    if num_active_bins == 0:
        return np.zeros((num_bins, num_bins), dtype=float)

    # Compressed histograms on the active support
    source_active = source_hist[active_bins_mask]
    target_active = target_hist[active_bins_mask]
    source_mass = float(source_active.sum())
    target_mass = float(target_active.sum())

    # If both are (numerically) empty, nothing to transport
    if source_mass <= 0.0 and target_mass <= 0.0:
        return np.zeros((num_bins, num_bins), dtype=float)

    # Augmented marginals on the active support
    source_aug = np.concatenate([source_active, [target_mass]]).astype(float)
    target_aug = np.concatenate([target_active, [source_mass]]).astype(float)

    # ---- Clean & normalize for POT (improves numerical stability) ----
    # Replace NaN / inf by 0
    source_aug[~np.isfinite(source_aug)] = 0.0
    target_aug[~np.isfinite(target_aug)] = 0.0

    # Clip tiny negative values (from numerical noise) to 0
    source_aug = np.clip(source_aug, 0.0, None)
    target_aug = np.clip(target_aug, 0.0, None)

    mass_source_aug = float(source_aug.sum())
    mass_target_aug = float(target_aug.sum())

    # If both sides are empty after cleaning: nothing to transport
    if mass_source_aug <= 0.0 and mass_target_aug <= 0.0:
        return np.zeros((num_bins, num_bins), dtype=float)

    # If only one side is empty, also return zeros (pathological input)
    if mass_source_aug <= 0.0 or mass_target_aug <= 0.0:
        return np.zeros((num_bins, num_bins), dtype=float)

    # They should be equal by construction; use their mean as robust scale
    mass_aug = 0.5 * (mass_source_aug + mass_target_aug)

    # Normalize to a probability simplex (sum ≈ 1)
    source_aug /= mass_aug
    target_aug /= mass_aug

    # Cost matrix on the active indices: use normalized original positions
    if cost == "sqeuclidean":
        # Normalize indices to keep squared distances ~ O(1)
        # This scaling is compensated in dust_cost so the OT plan is unchanged.
        scale = max(1.0, float(num_bins - 1))
        coords = active_bins_idx.astype(float) / scale
        cost_real = (coords[:, None] - coords[None, :]) ** 2
        dust_cost_scaled = dust_cost / (scale**2)
    else:
        raise ValueError(f"Unsupported cost: {cost!r}")

    cost_aug = np.empty((num_active_bins + 1, num_active_bins + 1), dtype=float)
    cost_aug[:num_active_bins, :num_active_bins] = cost_real
    cost_aug[:num_active_bins, num_active_bins] = dust_cost_scaled  # real -> dustbin
    cost_aug[num_active_bins, :num_active_bins] = dust_cost_scaled  # dustbin -> real
    cost_aug[num_active_bins, num_active_bins] = 0.0  # dustbin -> dustbin

    # Balanced Earth Mover's Distance on the compressed + dustbin problem
    transport_aug = ot.emd(source_aug, target_aug, cost_aug)

    # Rescale transport plan back to original total mass
    transport_aug *= mass_aug

    # Real-real block on the compressed support
    transport_active = np.asarray(
        transport_aug[:num_active_bins, :num_active_bins], dtype=float
    )

    # Expand back to full (n, n), filling inactive rows/cols with zeros
    transport_full = np.zeros((num_bins, num_bins), dtype=float)
    transport_full[np.ix_(active_bins_idx, active_bins_idx)] = transport_active

    return transport_full


def _warp_rows_by_plan_csr(X, a, Gamma, tol=1e-12):
    """
    Warp a CSR matrix X along rows according to transport plan Gamma.

    For each row i:
      - matched_mass_i = sum_j Gamma[i, j]
      - alpha_i = matched_mass_i / a[i] (fraction of the row to move)
      - matched part (alpha_i * row_i) is redistributed to targets j with
        weights Gamma[i, j] / matched_mass_i
      - unmatched part ((1 - alpha_i) * row_i) stays at row i.

    Parameters
    ----------
    X :sp.csr_array, shape (n_rows, n_cols)
        Source matrix.
    a : array_like, shape (n_rows,)
        Row sums of X used to compute the plan (same as in
        _balanced_ot_with_dustbin_1d).
    Gamma : array_like, shape (n_rows, n_rows)
        Real-real part of the OT plan.
    tol : float, optional
        Numerical tolerance for discarding tiny weights.

    Returns
    -------
    X_warped :sp.csr_array, shape (n_rows, n_cols)
        Row-warped matrix.
    """
    source_matrix = sp.csr_array(X)  # ensure CSR array
    n_rows, n_cols = source_matrix.shape

    row_sums = np.asarray(a, dtype=float).ravel()
    transport_plan = np.asarray(Gamma, dtype=float)

    indptr = source_matrix.indptr
    indices = source_matrix.indices
    data = source_matrix.data

    # Matched mass per row, and fraction alpha_i to move
    matched_mass = transport_plan.sum(axis=1)
    matched_mass = np.asarray(matched_mass).ravel()

    move_fraction = np.zeros(n_rows, dtype=float)
    valid_rows = (row_sums > 0.0) & (matched_mass > 0.0)
    move_fraction[valid_rows] = np.minimum(
        matched_mass[valid_rows] / row_sums[valid_rows], 1.0
    )

    # Split rows into matched and unmatched parts
    data_matched = data.copy()
    for i in range(n_rows):
        row_start, row_end = indptr[i], indptr[i + 1]
        if row_start == row_end:
            continue

        frac_i = move_fraction[i]
        if frac_i == 0.0:
            data_matched[row_start:row_end] = 0.0
        elif frac_i != 1.0:
            data_matched[row_start:row_end] *= frac_i

    data_unmatched = data - data_matched

    # Unmatched part: stays at original rows
    X_unmatched = sp.csr_array(
        (data_unmatched, indices.copy(), indptr.copy()),
        shape=(n_rows, n_cols),
    )

    # Matched part: transported according to Gamma
    rows_list = []
    cols_list = []
    vals_list = []

    for i in range(n_rows):
        frac_i = move_fraction[i]
        if frac_i <= 0.0:
            continue

        row_start, row_end = indptr[i], indptr[i + 1]
        if row_start == row_end:
            continue

        row_plan = transport_plan[i, :]
        total_row_plan = row_plan.sum()
        if total_row_plan <= tol:
            continue

        weights = row_plan / total_row_plan
        dest_rows = np.nonzero(weights > tol)[0]
        if dest_rows.size == 0:
            continue

        row_cols = indices[row_start:row_end]
        row_vals = data_matched[row_start:row_end]

        for j in dest_rows:
            weight_ij = weights[j]
            if weight_ij <= 0.0:
                continue
            rows_list.append(np.full(row_cols.shape, j, dtype=int))
            cols_list.append(row_cols)
            vals_list.append(row_vals * weight_ij)

    if rows_list:
        rows_cat = np.concatenate(rows_list)
        cols_cat = np.concatenate(cols_list)
        vals_cat = np.concatenate(vals_list)

        X_moved = sp.csr_array(
            (vals_cat, (rows_cat, cols_cat)),
            shape=(n_rows, n_cols),
        )
        X_moved.sum_duplicates()
    else:
        # No matched mass transported
        X_moved = sp.csr_array((n_rows, n_cols))

    return X_unmatched + X_moved
