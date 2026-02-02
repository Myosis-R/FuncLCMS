from pathlib import Path
import h5py

import numpy as np
import scipy.sparse as sp


def cache_spectrums(s, with_grid=True):
    """
    Build a cache for a Spectrum `s`, independent of original source.
    Uses s.df, s.frames_rt, s.date_time, and optionally s.grid.
    """
    # Ensure df is materialized (this triggers loader no matter the source)
    df = s.df
    frames_rt = s.frames_rt
    reg_a = s.reg_a
    reg_b = s.reg_b
    date_time = s.date_time

    cache_dir = Path(s.path).with_suffix(".cache")
    cache_dir.mkdir(exist_ok=True)

    # ---- df ----
    df.to_feather(cache_dir / "df.feather")

    # ---- convert ----
    np.savez(
        cache_dir / "convert.npz",
        frames_rt=frames_rt,
        tof_to_mz=np.array([reg_a, reg_b]),
    )

    # ---- meta ----
    np.savez_compressed(
        cache_dir / "meta.npz",
        date_time=date_time.timestamp(),
        analyser=s.analyser,
        # coefs=coefs_tmz_to_mz, TODO:
    )

    # ---- grid (optional) ----
    if with_grid:
        grid = s.grid  # will build local grid if not already done
        sp.save_npz(cache_dir / "grid_data.npz", grid._data)
        np.save(cache_dir / "grid_coord0.npy", grid.coord0)
        np.save(cache_dir / "grid_coord1.npy", grid.coord1)

    s.format = "cache"
    s.path = cache_dir

    return cache_dir


def projections(los):
    assert los.all_grids_standard()
    dir = Path(los[0].path).parent.joinpath("Proj")
    dir.mkdir(exist_ok=True)

    with h5py.File(dir / "proj_rt.h5", "w") as f:
        f.create_dataset(
            "proj_rt",
            data=np.array([s.grid.sum_along_axis(axis=0, boolean=False) for s in los]),
        )
    with h5py.File(dir / "proj_tof.h5", "w") as f:
        f.create_dataset(
            "proj_tof",
            data=np.array([s.grid.sum_along_axis(axis=1, boolean=False) for s in los]),
        )
    with h5py.File(dir / "proj_rt_bool.h5", "w") as f:
        f.create_dataset(
            "proj_rt_bool",
            data=np.array([s.grid.sum_along_axis(axis=0, boolean=True) for s in los]),
        )
    with h5py.File(dir / "proj_tof_bool.h5", "w") as f:
        f.create_dataset(
            "proj_tof_bool",
            data=np.array([s.grid.sum_along_axis(axis=1, boolean=True) for s in los]),
        )
