import numpy as np
import scipy.sparse as sp


class Grid2D:
    """
    2D sparse grid with physical coordinates for each axis.

    Designed for your spectra use case:
      - axis 0: typically retention time (rt / frames)
      - axis 1: typically tmz / mz

    Attributes
    ----------
    data : scipy.sparse.spmatrix-like (2D)
        The underlying sparse array (csr_array by default; methods may
        internally convert to csr/csc/coo as needed).
    coords : tuple of np.ndarray
        (coord0, coord1), one coordinate value per row/column.
    axis_names : tuple of str
        Optional semantic names for each axis, e.g. ("rt", "tmz").
    """

    # mappings similar to your tools.py
    _sparse_methods = {
        -1: "tocoo",
        0: "tocsr",
        1: "tocsc",
    }
    _sparse_constructors = {
        -1: "coo_array",
        0: "csr_array",
        1: "csc_array",
    }

    def __init__(self, data, coord0, coord1, axis_names=("axis0", "axis1"), copy=False):
        """
        Parameters
        ----------
        data : array-like or scipy.sparse array
            2D data array. Will be converted to scipy.sparse.csr_array.
        coord0 : array-like
            Coordinates along axis 0 (length must match data.shape[0]).
        coord1 : array-like
            Coordinates along axis 1 (length must match data.shape[1]).
        axis_names : tuple of str, optional
            Semantic names for axes, e.g. ("rt", "tmz").
        copy : bool, optional
            If True, copy the input data and coordinates.
        """
        coord0 = np.asarray(coord0, copy=copy)
        coord1 = np.asarray(coord1, copy=copy)

        # Normalize data to a sparse array (csr by default)
        if sp.isspmatrix(data):
            data = sp.csr_array(data) if copy else data.tocsr()
        elif isinstance(data, (sp.coo_array, sp.csr_array, sp.csc_array)):
            data = data.copy() if copy else data
            # convert to csr for a consistent base representation
            data = data.tocsr()
        else:
            # dense -> sparse
            data = sp.csr_array(np.asarray(data, copy=copy))

        if data.ndim != 2:
            raise ValueError("Grid2D data must be 2D")

        if data.shape[0] != len(coord0) or data.shape[1] != len(coord1):
            raise ValueError(
                f"Shape {data.shape} is inconsistent with coord lengths "
                f"{len(coord0)}, {len(coord1)}"
            )

        self._data = data
        self._coords = [coord0, coord1]
        self._axis_names = tuple(axis_names)

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def data(self):
        """Underlying 2D sparse array (SciPy)."""
        return self._data

    @data.setter
    def data(self, new_data):
        new_data = self._ensure_sparse_2d(new_data)
        if new_data.shape != self.shape:
            raise ValueError(
                f"New data shape {new_data.shape} does not match "
                f"current shape {self.shape}"
            )
        self._data = new_data

    @property
    def coords(self):
        """Tuple (coord0, coord1) of 1D numpy arrays."""
        return tuple(self._coords)

    @property
    def coord0(self):
        return self._coords[0]

    @property
    def coord1(self):
        return self._coords[1]

    @property
    def axis_names(self):
        return self._axis_names

    @property
    def shape(self):
        return self._data.shape

    @property
    def nnz(self):
        return self._data.nnz

    @property
    def ndim(self):
        return 2

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_sparse_2d(data):
        """Ensure data is a 2D SciPy sparse array."""
        if sp.isspmatrix(data):
            return data.tocsr()
        if isinstance(data, (sp.coo_array, sp.csr_array, sp.csc_array)):
            if data.ndim != 2:
                raise ValueError("Sparse array must be 2D")
            return data
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValueError("Data must be 2D")
        return sp.csr_array(arr)

    @staticmethod
    def _axis_slice(shape, axis, start, end, step=1):
        """Create a slicing tuple for slicing along a single axis."""
        sl = [slice(None)] * len(shape)
        sl[axis] = slice(start, end, step)
        return tuple(sl)

    # ------------------------------------------------------------------
    # General utilities
    # ------------------------------------------------------------------
    def copy(self, deep=True):
        """Return a copy of the grid."""
        data = self._data.copy() if deep else self._data
        coord0 = self.coord0.copy() if deep else self.coord0
        coord1 = self.coord1.copy() if deep else self.coord1
        return Grid2D(data, coord0, coord1, axis_names=self.axis_names, copy=False)

    def as_format(self, fmt="csr"):
        """
        Return the underlying data in a specific sparse format.

        Parameters
        ----------
        fmt : {"csr", "csc", "coo"}
        """
        if fmt == "csr":
            return self._data.tocsr()
        if fmt == "csc":
            return self._data.tocsc()
        if fmt == "coo":
            return self._data.tocoo()
        raise ValueError(f"Unknown sparse format '{fmt}'")

    def to_dense(self):
        """Return a dense numpy.ndarray view of the data."""
        return self._data.toarray()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def set_axis_names(self, name0, name1):
        """Set semantic axis names (e.g. ('rt','tmz'))."""
        self._axis_names = (str(name0), str(name1))

    def coord(self, axis):
        """Return coordinate array for a given axis (0 or 1)."""
        return self._coords[axis]

    # ------------------------------------------------------------------
    # Window / cropping
    # ------------------------------------------------------------------
    def window(self, axis, min_val, max_val, *, inplace=False):
        """
        Restrict the grid to [min_val, max_val] along a given axis.

        Parameters
        ----------
        axis : {0, 1}
            Axis along which to apply the window.
        min_val, max_val : float
            Bounds in coordinate space.
        inplace : bool, optional
            If True, modify self. Else, return a new Grid2D.

        Returns
        -------
        Grid2D or None
        """
        if axis not in (0, 1):
            raise ValueError("axis must be 0 or 1")

        coord = self._coords[axis]
        # Assuming coords are sorted increasing
        start = np.searchsorted(coord, min_val, side="left")
        end = np.searchsorted(coord, max_val, side="right")

        if start >= end:
            # Empty window
            new_data = (
                sp.csr_array((0, self.shape[1 - axis]))
                if axis == 0
                else sp.csr_array((self.shape[0], 0))
            )
            new_coord = coord[:0]
        else:
            sl = self._axis_slice(self.shape, axis, start, end)
            if axis == 0:
                new_data = self._data[sl[0], :]
            else:
                # slicing CSR by columns is not super efficient but OK
                new_data = self._data[:, sl[1]]
            new_coord = coord[start:end]

        if inplace:
            self._data = new_data
            self._coords[axis] = new_coord
            return None

        new_coords = list(self._coords)
        new_coords[axis] = new_coord
        return Grid2D(
            new_data, new_coords[0], new_coords[1], axis_names=self.axis_names
        )

    # ------------------------------------------------------------------
    # Interpolation along an axis (ported from tools.interpolation)
    # ------------------------------------------------------------------
    def interpolate_axis(self, axis, out_coord, in_coord=None, *, inplace=False):
        """
        Interpolate the grid along one axis to new coordinates (1D linear).

        This is the Grid2D version of your tools.interpolation(s, axis, ...):

        - Avoids extrapolating out of bounds:
          values outside [min(in_coord), max(in_coord)] are set to 0.
        - Uses linear interpolation between neighboring bins.

        Parameters
        ----------
        axis : {0, 1}
            Axis along which to interpolate.
        out_coord : array-like
            New coordinate values along that axis.
        in_coord : array-like or None, optional
            Existing coordinates along that axis (defaults to self.coord(axis)).
        inplace : bool, optional
            If True, modify this Grid2D. Else, return a new Grid2D.

        Returns
        -------
        Grid2D or None
        """
        if axis not in (0, 1):
            raise ValueError("axis must be 0 or 1")

        out_coord = np.asarray(out_coord)
        if in_coord is None:
            in_coord = self._coords[axis]
        else:
            in_coord = np.asarray(in_coord)

        if self.shape[axis] != len(in_coord):
            raise ValueError(
                f"data.shape[{axis}] ({self.shape[axis]}) "
                f"!= len(in_coord) ({len(in_coord)})"
            )

        # Choose a format adapted to slicing along 'axis'
        # axis=0 -> efficient row indexing: use CSR
        # axis=1 -> efficient col indexing: use CSC
        ds = getattr(self._data, self._sparse_methods[axis])()

        # Avoid extrapolation out of bounds
        mask = np.logical_and(
            out_coord < np.max(in_coord), out_coord > np.min(in_coord)
        )

        idxs = np.ones(len(out_coord), dtype=int)
        idxs[mask] = np.searchsorted(in_coord, out_coord[mask])

        # Linear interpolation coefficients
        # coefs has shape (len(out_coord), 1)
        coefs = (out_coord - in_coord[idxs - 1]) / (in_coord[idxs] - in_coord[idxs - 1])
        coefs = coefs.reshape((-1, 1))

        # New shape: only the interpolated axis length changes
        new_shape = list(self.shape)
        new_shape[axis] = len(out_coord)

        # Build new sparse data, similar logic to your tools.interpolation
        if axis == 0:
            # interpolate rows: ds[idxs-1, :] and ds[idxs, :]
            ds0 = ds[idxs - 1, :]
            ds1 = ds[idxs, :]
            # mask[row] == 0 -> zero row
            new_ds = (1 - coefs) * ds0 + coefs * ds1
            new_ds = sp.csc_array(mask.reshape((-1, 1)) * new_ds)
        else:
            # axis == 1: interpolate columns
            ds0 = ds[:, idxs - 1]
            ds1 = ds[:, idxs]
            new_ds = ds0 * (1 - coefs.T) + ds1 * coefs.T
            new_ds = sp.csr_array(new_ds * mask.reshape((1, -1)))

        new_ds.eliminate_zeros()
        new_ds.resize(tuple(new_shape))

        if inplace:
            self._data = new_ds
            self._coords[axis] = out_coord
            return None

        new_coords = list(self._coords)
        new_coords[axis] = out_coord
        return Grid2D(new_ds, new_coords[0], new_coords[1], axis_names=self.axis_names)

    # ------------------------------------------------------------------
    # Reindex along an axis (no interpolation, only remapping)
    # ------------------------------------------------------------------
    def reindex(self, axis, out_coord, *, inplace=False, rtol=0.0, atol=0.0):
        """
        Reindex the grid along a given axis to new coordinates, *without*
        intensity interpolation.

        - Each existing row/column is moved to the position where its
          coordinate appears in `out_coord`.
        - Positions in `out_coord` that do not exist in the current grid
          become all-zero rows/columns.
        - If `rtol`/`atol` are 0, coordinates must match exactly; otherwise
          they must match within the tolerances.

        Parameters
        ----------
        axis : {0, 1}
            Axis to reindex: 0 for rows (rt), 1 for columns (tmz).
        out_coord : array-like
            Target coordinate axis, sorted and 1D.
        inplace : bool, optional
            If True, modify self in place and return None.
            If False (default), return a new Grid2D.
        rtol, atol : float, optional
            Relative/absolute tolerances for matching coordinates.

        Returns
        -------
        Grid2D or None
            New reindexed grid if inplace is False, else None.

        Raises
        ------
        ValueError
            If `axis` is not 0/1, or if `out_coord` does not contain all
            existing coordinates (within given tolerances).
        """
        if axis not in (0, 1):
            raise ValueError("axis must be 0 or 1")

        in_coord = self._coords[axis]
        out_coord = np.asarray(out_coord)

        if out_coord.ndim != 1:
            raise ValueError("out_coord must be 1D")

        # We require out_coord to be sorted increasing
        if np.any(np.diff(out_coord) < 0):
            raise ValueError("out_coord must be sorted in non-decreasing order")

        # For each existing coordinate, find its index in out_coord
        idx = np.searchsorted(out_coord, in_coord)

        # Any index outside [0, len(out_coord)-1] is invalid
        if np.any(idx < 0) or np.any(idx >= len(out_coord)):
            raise ValueError(
                "out_coord does not span the full range of existing coordinates"
            )

        # Check that matched coordinates really correspond (within tolerance)
        matched = out_coord[idx]
        if not np.allclose(matched, in_coord, rtol=rtol, atol=atol):
            raise ValueError(
                "out_coord must contain all existing coordinates along the axis "
                "within the specified tolerances"
            )

        # Now remap non-zeros using COO representation
        coo = self._data.tocoo()
        if axis == 0:
            # rows are reindexed
            new_rows = idx[coo.row]
            new_cols = coo.col
        else:
            # columns are reindexed
            new_rows = coo.row
            new_cols = idx[coo.col]

        new_shape = list(self.shape)
        new_shape[axis] = len(out_coord)
        new_shape = tuple(new_shape)

        new_data = sp.coo_array((coo.data, (new_rows, new_cols)), shape=new_shape)
        # Use CSR as a standard internal format (consistent with __init__)
        new_data = new_data.tocsr()

        if inplace:
            self._data = new_data
            self._coords[axis] = out_coord
            return None

        new_coords = list(self._coords)
        new_coords[axis] = out_coord
        return Grid2D(
            new_data,
            new_coords[0],
            new_coords[1],
            axis_names=self.axis_names,
        )

    # ------------------------------------------------------------------
    # Reductions and simple analysis
    # ------------------------------------------------------------------
    def sum(self):
        """
        Sum over both axes.
        """
        return self._data.sum()

    def sum_along_axis(self, axis=0, boolean=False):
        """
        Sum along one axis.

        Parameters
        ----------
        axis : {0, 1}
            Axis to keep (i.e. sum over 1-axis).
        boolean : bool, optional
            If True, treat non-zero entries as 1 (presence/absence).

        Returns
        -------
        coord : np.ndarray
            Coordinate values along the chosen axis.
        trace : np.ndarray
            1D array of summed intensities or counts.
        """
        if axis not in (0, 1):
            raise ValueError("axis must be 0 or 1")

        if boolean:
            # convert data to boolean (presence / absence)
            data_bool = self._data.astype(bool)
            if axis == 0:
                trace = np.asarray(data_bool.sum(axis=1)).ravel()  # NOTE: convert csr?
            else:
                trace = np.asarray(data_bool.sum(axis=0)).ravel()
        else:
            if axis == 0:
                trace = np.asarray(self._data.sum(axis=1)).ravel()
            else:
                trace = np.asarray(self._data.sum(axis=0)).ravel()

        return self._coords[axis], trace

    # ------------------------------------------------------------------
    # Compatibility / checks
    # ------------------------------------------------------------------
    def is_compatible_with(self, other, rtol=0.0, atol=0.0):
        """
        Check if another Grid2D has the same coordinates (within tolerances).

        Parameters
        ----------
        other : Grid2D
        rtol, atol : float
            Relative/absolute tolerances for np.allclose on coordinates.

        Returns
        -------
        bool
        """
        if not isinstance(other, Grid2D):
            return False
        if self.shape != other.shape:
            return False
        if not np.allclose(self.coord0, other.coord0, rtol=rtol, atol=atol):
            return False
        if not np.allclose(self.coord1, other.coord1, rtol=rtol, atol=atol):
            return False
        return True

    # ------------------------------------------------------------------
    # Arithmetic between compatible grids
    # ------------------------------------------------------------------
    def _binary_op(self, other, op):
        if isinstance(other, Grid2D):
            if not self.is_compatible_with(other):
                raise ValueError(
                    "Grid2D objects have different coordinates "
                    "or shapes; cannot combine safely."
                )
            new_data = op(self._data, other._data)
            return Grid2D(
                new_data, self.coord0, self.coord1, axis_names=self.axis_names
            )
        else:
            # scalar
            new_data = op(self._data, other)
            return Grid2D(
                new_data, self.coord0, self.coord1, axis_names=self.axis_names
            )

    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b)

    def __truediv__(self, other):
        return self._binary_op(other, lambda a, b: a / b)

    def __repr__(self):
        return (
            f"Grid2D(shape={self.shape}, nnz={self.nnz}, "
            f"axis_names={self.axis_names})"
        )
