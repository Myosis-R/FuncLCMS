import numpy as np
from .spectrum import List_of_Spectrum, Spectrum
import matplotlib.pyplot as plt
import scipy.sparse as sp
from .sot import ot_align_1d
from .hot import _balanced_ot_near_target_source_2d


from grid2d import Grid2D


def ensure_common_toy_grid(los, size=16):
    """
    For toy data:
    - Enforce a common size `size x size` grid for all spectra in `los`,
      with coordinates 0..size-1 on both axes.
    - Build Grid2D directly from df (non-zero entries), no interpolation.
    - Overwrites each Spectrum.grid with this new Grid2D.

    Assumptions:
    - df.rt and df.tmz are integer-like and in [0, size-1].
    - df contains only non-zero intensities.
    """

    # 1) Define the global toy axes
    rt_axis = np.arange(size, dtype=int)
    tmz_axis = np.arange(size, dtype=int)

    for spec in los:
        df = spec.df  # ensure it is loaded
        if df is None or df.empty:
            # Completely empty: all-zero grid
            sparse_2d = sp.csr_array((size, size))
            spec.grid = Grid2D(
                sparse_2d, coord0=rt_axis, coord1=tmz_axis, axis_names=("rt", "tmz")
            )
            continue

        rt_vals = np.asarray(df["rt"].values)
        tmz_vals = np.asarray(df["tmz"].values)
        ints = np.asarray(df["int"].values, dtype=float)

        # 2) Sanity checks: integer-like, within bounds
        if not np.allclose(rt_vals, np.round(rt_vals)):
            raise ValueError(f"{spec.name}: df.rt must be integer-like for toy grid.")
        if not np.allclose(tmz_vals, np.round(tmz_vals)):
            raise ValueError(f"{spec.name}: df.tmz must be integer-like for toy grid.")

        rt_idx = rt_vals.astype(int)
        tmz_idx = tmz_vals.astype(int)

        if (rt_idx.min() < 0) or (rt_idx.max() >= size):
            raise ValueError(
                f"{spec.name}: rt indices out of [0, {size-1}] range: "
                f"[{rt_idx.min()}, {rt_idx.max()}]"
            )
        if (tmz_idx.min() < 0) or (tmz_idx.max() >= size):
            raise ValueError(
                f"{spec.name}: tmz indices out of [0, {size-1}] range: "
                f"[{tmz_idx.min()}, {tmz_idx.max()}]"
            )

        # 3) Build a sparse matrix on the *global* index grid
        sparse_2d = sp.coo_array(
            (ints, (rt_idx, tmz_idx)),
            shape=(size, size),
        ).tocsr()

        # 4) Overwrite the spectrum's grid with a common-coordinate Grid2D
        spec.grid = Grid2D(
            sparse_2d, coord0=rt_axis, coord1=tmz_axis, axis_names=("rt", "tmz")
        )

    # Store axes at List_of_Spectrum level (optional, mirrors standardize_all)
    los.rt_axis = rt_axis
    los.tmz_axis = tmz_axis


def plot_cases(los):
    N = len(los)
    ncols = int(np.ceil(np.sqrt(N)))
    nrows = int(np.ceil(N / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows))
    for i, s in enumerate(los):
        axes.flat[i].imshow(s.grid.data.todense())
    plt.show()


def align_pair_all_methods_16x16(
    target: Spectrum,
    source: Spectrum,
    intention: Spectrum,
    *,
    dust_cost_1d: float = 500.0,
    dust_cost_2d: float = 500.0,
    axis_weights_2d=(1.0, 15.0),
    mass_tol_2d: float = 0.0,
    cmap: str = "viridis",
):
    """
    Visual demo for pairwise alignment on *toy* 16×16 spectra.

    Parameters
    ----------
    target : Spectrum
        Reference spectrum (remains unchanged).
    source : Spectrum
        Spectrum to align onto `target`.
    dust_cost_1d : float
        Dustbin cost for 1D OT along rows.
    dust_cost_2d : float
        Dustbin cost for 2D OT over (row, col) point clouds.
    axis_weights_2d : (float, float)
        Anisotropic weights (w_row, w_col) for 2D cost.
    mass_tol_2d : float
        Minimum mass to keep for 2D OT output points (see
        `_balanced_ot_near_target_source_2d`).
    cmap : str
        Matplotlib colormap for imshow.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes array
        The 1×4 panel: [source, target, aligned_1d, aligned_2d].
    """

    # Ensure grids exist and are 16×16
    tgt_grid = target.grid
    src_grid = source.grid

    if tgt_grid.data.shape != src_grid.data.shape:
        raise ValueError(
            f"target and source must have the same shape, got "
            f"{tgt_grid.data.shape} and {src_grid.data.shape}"
        )

    if tgt_grid.data.shape != (16, 16):
        raise ValueError(
            f"This demo assumes toy grids of shape (16,16), "
            f"got {tgt_grid.data.shape}"
        )

    n_rows, n_cols = tgt_grid.data.shape  # should be (16, 16)

    # ------------------------------------------------------------------
    # 1) 1D OT along rows (using target row-sum histogram as reference)
    # ------------------------------------------------------------------
    target_hist = np.asarray(tgt_grid.data.sum(axis=1)).ravel().astype(float)

    _, src_aligned_1d = ot_align_1d(
        src_grid.data,
        ref=target_hist,
        dust_cost=dust_cost_1d,
        cost="sqeuclidean",
        binarize=False,
    )
    # src_aligned_1d is a sp.csr_array with shape (16, 16)

    # ------------------------------------------------------------------
    # 2) 2D OT with `_balanced_ot_near_target_source_2d`
    # ------------------------------------------------------------------
    # Extract point clouds (row, col) with intensities as weights
    src_coo = src_grid.data.tocoo()
    tgt_coo = tgt_grid.data.tocoo()

    X = np.vstack([src_coo.row, src_coo.col]).T  # (N, 2)
    a = src_coo.data.astype(float)  # (N,)

    Y = np.vstack([tgt_coo.row, tgt_coo.col]).T  # (M, 2)
    b = tgt_coo.data.astype(float)  # (M,)

    # Build "near target" transported source measure
    Z, w = _balanced_ot_near_target_source_2d(
        X,
        Y,
        a=a,
        b=b,
        dust_cost=dust_cost_2d,
        axis_weights=axis_weights_2d,
        mass_tol=mass_tol_2d,
        merge_duplicates=True,
        return_plan=False,
    )
    # Rebuild a 16×16 sparse grid from (Z, w)
    if w.size > 0:
        rows_2d = Z[:, 0].astype(int)
        cols_2d = Z[:, 1].astype(int)
        aligned_2d = sp.coo_array(
            (w, (rows_2d, cols_2d)),
            shape=(n_rows, n_cols),
        ).tocsr()
    else:
        aligned_2d = sp.csr_array((n_rows, n_cols))

    # ------------------------------------------------------------------
    # 3) Convert to dense for plotting
    # ------------------------------------------------------------------
    source_dense = src_grid.data.toarray()
    target_dense = tgt_grid.data.toarray()
    intention_dense = intention.grid.data.toarray()
    aligned_1d_dense = src_aligned_1d.toarray()
    aligned_2d_dense = aligned_2d.toarray()

    # Shared color scale
    vmax = max(
        source_dense.max(initial=0),
        target_dense.max(initial=0),
        aligned_1d_dense.max(initial=0),
        aligned_2d_dense.max(initial=0),
    )
    vmin = 0.0

    # ------------------------------------------------------------------
    # 4) Plot: [source, target, aligned_1d, aligned_2d]
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    ax0, ax1, ax4, ax2, ax3, ax6 = axes.flat

    im0 = ax0.imshow(
        source_dense,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax0.set_title("Source (original)")
    ax0.set_xlabel("col")
    ax0.set_ylabel("row")

    ax1.imshow(
        target_dense,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title("Target (reference)")
    ax1.set_xlabel("col")
    ax1.set_ylabel("row")

    ax2.imshow(
        aligned_1d_dense,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_title("Aligned source (1D OT)")
    ax2.set_xlabel("col")
    ax2.set_ylabel("row")

    im3 = ax3.imshow(
        aligned_2d_dense,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax3.set_title("Aligned source (2D OT)")
    ax3.set_xlabel("col")
    ax3.set_ylabel("row")

    im4 = ax4.imshow(
        intention_dense,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax4.set_title("Alignement proposal")
    ax4.set_xlabel("col")
    ax4.set_ylabel("row")

    # Single colorbar for all panels
    cbar = fig.colorbar(im4, ax=axes, shrink=0.8)
    cbar.set_label("intensity")

    return fig, axes


def main():
    params_data = {
        "analyser": "synthetic",
        "folder": "Data/Toy",
        "format": "toy",
        "name_specification": "name",
        "name_tweak": False,
    }

    los = List_of_Spectrum(params_data)
    los.sort("name")

    # For toy: check that all grids are already 16×16
    ensure_common_toy_grid(los)

    trios = [[0, 1, 0], [2, 3, 4], [5, 6, 7]]

    for trio in trios:
        source = los[trio[0]]
        target = los[trio[1]]
        intention = los[trio[2]]

        fig, axes = align_pair_all_methods_16x16(
            target,
            source,
            intention,
            dust_cost_1d=500.0,
            dust_cost_2d=500.0,
            axis_weights_2d=(1.0, 20.0),
            mass_tol_2d=0.0,
        )
    plt.show()


if __name__ == "__main__":
    main()
