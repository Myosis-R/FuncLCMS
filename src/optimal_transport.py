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
    strips, _ = find_strip(los, params["min_zero"], optimal=True)
    for i, strip in enumerate(strips):
        print(i / len(strips))
        ref = los[0].grid.data[:, strip[0] : strip[1]].astype(bool).sum(axis=1)
        for s in los:
            _, s.grid.data[:, strip[0] : strip[1]] = ot_align_1d(
                s.grid.data[:, strip[0] : strip[1]].astype(bool),
                ref=ref,
                reg=params["reg"],
                reg_m=params["reg_m"],
                mode=params["mode"],
            )  # TODO: check that inplace works, csc?


def ot_align_1d(
    X,
    ref,
    reg=1e-1,
    reg_m=1.0,
    cost="sqeuclidean",
    mode="barycentric",  # or "argmax"
):
    """
    Align a 2D sparse matrix along rows using 1D unbalanced OT, reducing
    the OT plan to an index-valued map m.

    Functional style: X is NOT modified in place. A new csr_array is returned.

    Parameters
    ----------
    X : scipy.sparse.csr_array, shape (n_rows, n_cols)
        Source matrix.

    ref : array_like, shape (n_rows,)
        Reference 1D distribution along the row axis.
        Must satisfy len(ref) == n_rows. Non-negative.

    reg : float, default=1e-1
        Entropic regularization parameter for unbalanced Sinkhorn.

    reg_m : float, default=1.0
        Mass regularization parameter for unbalanced OT.

    cost : {"sqeuclidean"}, default="sqeuclidean"
        Ground cost on the 1D row grid: currently (i - j)**2 in the
        original row index space.

    mode : {"barycentric", "argmax"}, default="barycentric"
        How to extract a discrete row index map from the OT plan:
        - "barycentric": barycentric projection in index space + rounding.
        - "argmax":      m[i] = argmax_j gamma[i, j], with identity
                         fallback on empty rows.

    Returns
    -------
    m : ndarray, shape (n_rows,)
        Integer index map. m[i] = k means row i is transported to row k.

    X_aligned : csr_array, shape (n_rows, n_cols)
        New matrix with rows permuted/merged according to m.
    """
    if not isinstance(X, csr_array):
        raise TypeError("X must be a scipy.sparse.csr_array")

    n_rows, _ = X.shape

    # row sums (source)
    a = np.asarray(X.sum(axis=1)).ravel().astype(float)
    if np.any(a < 0):
        raise ValueError("Row sums of X must be non-negative")

    # reference (target)
    b = np.asarray(ref, dtype=float).ravel()
    if b.shape[0] != n_rows:
        raise ValueError(f"ref must have length {n_rows}, got {b.shape[0]}")
    if np.any(b < 0):
        raise ValueError("ref must be non-negative")

    # compute discrete row map via OT (support reduction inside)
    m = _ot_row_map_1d_full(a, b, reg=reg, reg_m=reg_m, cost=cost, mode=mode)

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(a)), a, "r-")
    ax.plot(np.arange(len(a)), b, "b-")
    ax.plot(np.arange(len(a)), m, "g-")

    # apply map to CSR, return new matrix
    X_aligned = _apply_row_map_csr(X, m)

    c = np.asarray(X_aligned.sum(axis=1)).ravel().astype(float)
    ax.plot(np.arange(len(a)), c, "k-")
    plt.show()
    return m, X_aligned


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


def _ot_row_map_1d(a, b, reg, reg_m, cost, mode):
    """
    Compute a discrete row map m from 1D unbalanced OT between histograms a, b.
    a, b are 1D non-negative arrays of same length n_rows.
    """
    n_rows = a.shape[0]
    m = np.arange(n_rows, dtype=np.int64)

    # detect active support (rows where either source or target has mass)
    active = (a > 0) | (b > 0)
    active_idx = np.flatnonzero(active)
    n_active = active_idx.size

    if n_active == 0:
        # nothing to transport: identity map
        return m

    a_active = a[active_idx]
    b_active = b[active_idx]

    # cost matrix on active support
    if cost == "sqeuclidean":
        # normalize indices to [0, 1] to keep the cost scale bounded
        coords = active_idx.astype(float) / max(n_rows - 1, 1)
        i_coords = coords[:, None]
        j_coords = coords[None, :]
        C = (i_coords - j_coords) ** 2
    else:
        raise ValueError(f"Unsupported cost: {cost!r}")

    # unbalanced OT plan on the reduced support
    gamma_active, log = sinkhorn_unbalanced(
        a_active,
        b_active,
        C,
        reg=reg,
        reg_m=reg_m,
        method="sinkhorn_stabilized",
        verbose=True,
        log=True,
    )  # FIX: clean
    # gamma_active = ot.partial.partial_wasserstein(
    #     a_active, b_active, C, m=0.8,nb_dummies=400)

    print(log)

    row_sums_active = gamma_active.sum(axis=1)

    # rows with some transported mass in the plan
    has_mass = row_sums_active > 0

    if mode == "barycentric":
        # barycentric projection of indices, then round to nearest integer index
        j_coords = active_idx[None, :].astype(float)  # shape (1, n_active)
        # avoid division by zero by masking has_mass
        bary = np.zeros_like(row_sums_active, dtype=float)
        bary[has_mass] = (gamma_active[has_mass, :] * j_coords).sum(
            axis=1
        ) / row_sums_active[has_mass]
        k_active = np.rint(bary).astype(np.int64)
        k_active = np.clip(k_active, 0, n_rows - 1)

        # fallback: identity for rows with no mass in the plan
        k_active[~has_mass] = active_idx[~has_mass]
        m_active = k_active

    elif mode == "argmax":
        # send each active row to the target with max transported mass
        j_active = np.asarray(gamma_active.argmax(axis=1)).ravel()
        m_active = active_idx[j_active]

        # fallback: identity for rows with no mass in the plan
        m_active[~has_mass] = active_idx[~has_mass]

    else:
        raise ValueError(f"Unknown mode {mode!r} (use 'barycentric' or 'argmax')")

    m[active_idx] = m_active
    return m


def _apply_row_map_csr(X, m):
    """
    Given a CSR matrix X and a row map m (1D int array, m[i] = target row),
    build a new CSR matrix whose rows are the result of transporting each
    row i of X to row m[i]. Row contents are merged (with duplicate (row,col)
    entries summed by CSR mechanics).
    """
    if not isinstance(X, csr_array):
        raise TypeError("X must be a scipy.sparse.csr_array")

    n_rows, n_cols = X.shape
    if m.shape[0] != n_rows:
        raise ValueError(f"m must have length {n_rows}, got {m.shape[0]}")

    indptr = X.indptr
    indices = X.indices
    data = X.data

    # 1st pass: count nnz going to each target row
    counts = np.zeros(n_rows, dtype=np.int64)
    for i in range(n_rows):
        k = m[i]
        if k < 0 or k >= n_rows:
            raise ValueError(f"m[{i}] = {k} is out of bounds for n_rows={n_rows}")
        row_len = indptr[i + 1] - indptr[i]
        counts[k] += row_len

    # prefix sum -> new indptr
    new_indptr = np.empty(n_rows + 1, dtype=np.int64)
    new_indptr[0] = 0
    np.cumsum(counts, out=new_indptr[1:])

    total_nnz = int(new_indptr[-1])
    new_indices = np.empty(total_nnz, dtype=indices.dtype)
    new_data = np.empty(total_nnz, dtype=data.dtype)

    # 2nd pass: copy row blocks into their target rows
    write_pos = new_indptr[:-1].copy()
    for i in range(n_rows):
        k = m[i]
        start = indptr[i]
        end = indptr[i + 1]
        length = end - start
        if length == 0:
            continue

        dst_start = write_pos[k]
        dst_end = dst_start + length

        new_indices[dst_start:dst_end] = indices[start:end]
        new_data[dst_start:dst_end] = data[start:end]

        write_pos[k] = dst_end

    X_new = csr_array((new_data, new_indices, new_indptr), shape=(n_rows, n_cols))
    X_new.sum_duplicates()

    return X_new


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
