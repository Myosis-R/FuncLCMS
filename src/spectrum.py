import re
import xml.etree.ElementTree as ET
from datetime import datetime as dt
from pathlib import Path

import numpy as np
import opentims_bruker_bridge
import pandas as pd
import scipy.sparse as sp
from opentimspy.opentims import OpenTIMS
from pyopenms import MSExperiment, MzMLFile

from . import save, load
from .grid2d import Grid2D


class List_of_Spectrum(list):
    """
    List of spectrum objects with added methods

    ...

    Attributes
    ----------
    list_paths : list
        list of file path containing each spectrum

    Methods
    -------
    """

    def __init__(self, params_data):
        self.params_data = params_data
        self.ref_grid = None
        list_paths = list(
            Path()
            .absolute()
            .parent.joinpath(params_data["folder"])
            .glob("*.{}".format(params_data["format"]))
        )
        print(str(Path().absolute()))
        if params_data.get("pattern", False):
            list_paths = [
                path
                for path in list_paths
                if re.search(params_data["pattern"], path.stem)
            ]
        print(list_paths)
        list.__init__(self, (Spectrum(path, params_data) for path in list_paths))

    def subset(self, indexes):
        """Keep only a subset of the spectrums selected by their indexes"""
        list.__init__(self, [self[idx] for idx in indexes])

    def write(self, with_grid=True):
        """Write all spectrums to pickle files, useful in order to load spectrums faster next times"""
        for s in self:
            save.cache_spectrums(s, with_grid)

    def specification(self):
        """Return a pandas dataframe with all the specifications of all spectrums"""
        return pd.concat(
            [pd.DataFrame.from_dict(spec.specification) for spec in self]
        ).reset_index(drop=True)

    def sort(self, by="time"):
        """Sort spectrums by time of analysis"""
        values = self.specification()[by].to_list()
        order = sorted(range(len(values)), key=values.__getitem__)
        list.__init__(self, [self[o] for o in order])

    def transform_mz(self, analyzer):
        # TODO: and inverse
        pass

    def build_global_tmz_axis(self):
        """
        Build a global tmz axis for all spectra in `self`.

        Current design:
        - Use raw df.tmz values from each Spectrum.
        - Take global integer min / max across all spectra.
        - Return a contiguous integer axis np.arange(min_tmz, max_tmz + 1).

        This gives a *uniform* tmz grid. Later, you can switch to a more
        sophisticated strategy (e.g. union of non‑contiguous values) or
        a true reindex in Grid2D.reindex without changing callers.
        """
        mins = []
        maxs = []

        for s in self:
            df = s.df
            if df is None or df.empty:
                continue
            tmz_vals = df.tmz.values
            mins.append(int(np.floor(tmz_vals.min())))
            maxs.append(int(np.ceil(tmz_vals.max())))

        if not mins:
            raise ValueError("No non‑empty spectra to build global tmz axis")

        global_min = min(mins)
        global_max = max(maxs)

        # Integer, contiguous, uniform tmz grid
        return np.arange(global_min, global_max + 1, dtype=int)

    def build_global_rt_axis(self):
        """
        Build a global rt axis for all spectra in `self`.

        Strategy:
        - Collect all distinct rt values across all spectra from df.rt.
        - Sort and unique them.
        - Estimate a typical step as the median of positive diffs.
        - Build a uniform grid from global min to global max with that step.

        This matches your "uniform grid" idea while using real rt values.
        """
        all_rt_unique = []

        for s in self:
            df = s.df
            if df is None or df.empty:
                continue
            # Use unique RT values from this spectrum
            rt_vals = np.unique(df.rt.values)
            if rt_vals.size > 0:
                all_rt_unique.append(rt_vals)

        if not all_rt_unique:
            raise ValueError("No non‑empty spectra to build global rt axis")

        all_rt = np.unique(np.concatenate(all_rt_unique))
        if all_rt.size == 1:
            # Degenerate case: only one rt across all spectra
            return all_rt

        diffs = np.diff(all_rt)
        # Keep only positive steps to estimate a typical spacing
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            # All rt values equal or pathological; just return sorted unique
            return all_rt

        delta = np.median(diffs)

        rt_min = all_rt.min()
        rt_max = all_rt.max()

        # Number of points in the uniform grid
        n_steps = int(np.floor((rt_max - rt_min) / delta)) + 1
        rt_axis = rt_min + delta * np.arange(n_steps)

        return rt_axis

    def standardize_all(
        self,
        standardize_rt: bool = False,
        standardize_tmz: bool = False,
        rt_axis=None,
        tmz_axis=None,
    ):
        """
        Apply standardization to all spectra, per axis.

        Only touch the axes you ask for:
        - If standardize_tmz: build/receive tmz_axis, call s.standardize_tmz(...)
        - If standardize_rt:  build/receive rt_axis,  call s.standardize_rt(...)

        Parameters
        ----------
        standardize_rt : bool
            If True, all spectra are interpolated onto a common rt_axis.
        standardize_tmz : bool
            If True, all spectra are interpolated onto a common tmz_axis.
        rt_axis : array-like or None
            If provided, use it as the common rt grid.
            If None and standardize_rt is True, a global uniform rt_axis
            is built from the spectra (see build_global_rt_axis).
        tmz_axis : array-like or None
            If provided, use it as the common tmz grid.
            If None and standardize_tmz is True, a global integer tmz_axis
            is built from the spectra (see build_global_tmz_axis).

        Returns
        -------
        (rt_axis, tmz_axis) : tuple
            The actual axes used (may be None if not standardized).
        """
        # Build missing global axes if needed
        if standardize_tmz and tmz_axis is None:
            tmz_axis = self.build_global_tmz_axis()

        if standardize_rt and rt_axis is None:
            rt_axis = self.build_global_rt_axis()

        # Apply standardization on each Spectrum
        for s in self:
            if standardize_tmz:
                s.standardize_tmz(tmz_axis)
            if standardize_rt:
                s.standardize_rt(rt_axis)

        self.rt_axis = rt_axis
        self.tmz_axis = tmz_axis

    def all_grids_standard(self, ref=False, rtol=1e-6, atol=1e-12):
        """
        Return True iff all spectra have a Grid2D compatible with the first one.

        - Uses Grid2D.is_compatible_with.
        - Does NOT build grids: if any spectrum has no grid, returns False.
        """
        if len(self) <= 1:
            return True

        # Reference grid from first spectrum or reference
        if ref:
            assert self.ref_grid is not None
            ref_grid = self.ref_grid
        else:
            assert self[0]._grid is not None
            ref_grid = self[0]._grid

        for spec in self:
            grid = spec._grid
            if grid is None:
                return False
            if not grid.is_compatible_with(ref_grid, rtol=rtol, atol=atol):
                return False

        return True


class Spectrum:

    def __init__(self, path, params_data):
        self.path = path
        self.name = path.stem
        self.format = path.suffix[1:]
        self.analyser = params_data["analyser"]

        # 2D representation
        self._grid = None

        # convert and meta coming from loader
        self.date_time = getattr(load, self.format + "_meta")(self.path)["date_time"]
        self.frames_rt, self.reg_a, self.reg_b = getattr(
            load, self.format + "_convert"
        )(self.path)

        # tabular representation
        self._df = None

        self.extract_information(
            params_data["name_specification"], tweak=params_data["name_tweak"]
        )
        print(self.specification)

    def extract_information(self, name_specification, tweak):
        """extract information from the name of spectrums and create an attribute with all information"""
        keys = re.split("-", name_specification)
        values = re.split("-", self.name)
        if tweak:  # Specific to Jade naming of spectrums
            if values[2][:2] in ("Bl", "Mi", "FI"):  # TODO pas beau ∞
                keys.pop(3)
            elif values[2][:2] in ("QC"):
                ...
            else:
                values = values[:2] + ["Std"] + values[2:]

        self.specification = {keys[i]: [values[i]] for i in range(len(keys))}
        self.specification["time"] = self.date_time

    # converter
    def tmz_to_mz(self, tmz):
        assert hasattr(self, "reg_a") and hasattr(self, "reg_b")
        return (self.reg_a + self.reg_b * tmz) ** 2

    def mz_to_tmz(self, mz):
        assert hasattr(self, "reg_a") and hasattr(self, "reg_b")
        return np.round((np.sqrt(mz) - self.reg_a) / self.reg_b)

    # ------------------------------------------------------------------
    # df property (unchanged, with your comments)
    # ------------------------------------------------------------------
    def _get_df(self):
        if self._df is None:
            self._df = getattr(load, self.format)(self.path)
            self._df.columns = ["rt", "tmz", "int"]  # TODO:
        return self._df

    def _set_df(self, df):
        self._df = df

    def _del_df(self):
        self._df = None

    df = property(
        fget=_get_df, fset=_set_df, fdel=_del_df, doc="Pandas dataframe rt,t(mz),int"
    )

    # ------------------------------------------------------------------
    # Internal builder: df -> *local* Grid2D (no standardization here)
    # ------------------------------------------------------------------
    def _build_local_grid(self):
        """
        Build a Grid2D from self.df using *local* axes derived from this spectrum only.

        - axis 0: rt (subset of self.frames_rt, contiguous)
        - axis 1: tmz (integer indices, from min_tmz to max_tmz inclusive)
        """
        df = self.df  # ensures loader has run
        points_rt = df.rt.values
        assert np.all(points_rt[1:] >= points_rt[:-1]), "df must be sorted by rt"

        tmz_vals = df.tmz.values
        assert np.all(
            np.isclose(tmz_vals, np.round(tmz_vals))
        ), "tmz must be integer-binned"
        tmz_vals = tmz_vals.astype(int)
        if points_rt.size == 0:
            raise ValueError("Empty df, cannot build grid")

        # Group by constant RT (as in your original _get_ds logic)
        diffs = points_rt[1:] - points_rt[:-1]
        change_idx = np.nonzero(diffs)[0] + 1
        indptr = np.concatenate(([0], change_idx, [len(points_rt)]))

        # map each RT block to a frame index
        frame = np.searchsorted(self.frames_rt, points_rt[indptr[:-1]], side="left")

        # repeated frame index per row of df
        cat_rt = np.concatenate(
            [np.repeat(frame[i], indptr[i + 1] - indptr[i]) for i in range(len(frame))]
        )

        # tmz range
        min_tmz = int(np.amin(tmz_vals))
        max_tmz = int(np.amax(tmz_vals))

        coord_rt = self.frames_rt[frame[0] : frame[-1] + 1]
        coord_tmz = np.arange(min_tmz, max_tmz + 1)

        # row indices: shift so that frame[0] -> row 0
        rows = cat_rt - frame[0]
        # column indices: shift so that min_tmz -> col 0
        cols = tmz_vals - min_tmz
        cols = cols.astype(int, copy=False)
        rows = rows.astype(int, copy=False)
        data = df.int.values

        sparse_2d = sp.coo_array(
            (data, (rows, cols)), shape=(len(coord_rt), len(coord_tmz))
        )

        self._grid = Grid2D(
            sparse_2d,
            coord0=coord_rt,
            coord1=coord_tmz,
            axis_names=("rt", "tmz"),
        )

        return self._grid

    def standardize_rt(self, rt_axis):
        """
        Interpolate this spectrum's Grid2D onto a given rt_axis (axis=0).

        - Ensures self._grid exists (builds it if necessary).
        - Uses Grid2D.interpolate_axis(axis=0, out_coord=rt_axis).
        - Replaces self._grid with the interpolated Grid2D.

        Parameters
        ----------
        rt_axis : array-like
            Target retention time coordinate for axis 0.

        Returns
        -------
        Grid2D
            The updated grid on the new rt axis.
        """
        # Lazy construction of the grid from df if not done yet
        if self._grid is None:
            self._build_local_grid()

        # Interpolate along axis 0 (rt) onto the new coordinate
        # NOTE: we let interpolate_axis return a new Grid2D and assign it.
        self._grid = self._grid.interpolate_axis(axis=0, out_coord=rt_axis)

        return self._grid

    def standardize_tmz(self, tmz_axis):
        """
        Reindex this spectrum's Grid2D onto a given tmz_axis (axis=1).

        For tmz we do *discrete* reindexing (no intensity interpolation):
        - Each existing tmz column is moved to the position of the same
          tmz value in `tmz_axis`.
        - Columns in `tmz_axis` that do not exist in this spectrum
          become all-zero columns.

        Parameters
        ----------
        tmz_axis : array-like
            Target tmz coordinate for axis 1. Must contain all existing
            tmz coordinates (within Grid2D.reindex tolerances).

        Returns
        -------
        Grid2D
            The updated grid on the new tmz axis.
        """
        # Ensure the 2D grid exists
        if self._grid is None:
            self._build_local_grid()

        # Discrete reindex along axis 1 (tmz) onto the new coordinate
        self._grid = self._grid.reindex(axis=1, out_coord=tmz_axis)

        return self._grid

    # ------------------------------------------------------------------
    # grid property: no heavy logic here, just return the stored Grid2D
    # ------------------------------------------------------------------
    def _get_grid(self):
        if self._grid is None:
            self._build_local_grid()  # FIX: load grid
        return self._grid

    def _set_grid(self, grid):
        if not isinstance(grid, Grid2D):
            raise TypeError("grid must be a Grid2D instance")
        self._grid = grid

    def _del_grid(self):
        self._grid = None

    grid = property(
        fget=_get_grid,
        fset=_set_grid,
        fdel=_del_grid,
        doc="2D sparse representation (Grid2D) with rt and tmz coordinates",
    )
