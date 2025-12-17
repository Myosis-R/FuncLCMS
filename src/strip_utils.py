import numpy as np


def zero_runs(a, min_zero):
    """
    Return strips (start, end) of contiguous non-zero regions of `a`,
    using runs of zeros longer than `min_zero` as separators.

    If there is no such zero-run, returns a single strip (0, len(a)).
    """
    values = np.asarray(a)
    num_samples = len(values)

    if num_samples == 0:
        return np.zeros((0, 2), dtype=int)

    # 1 where values is 0, padded with a leading and trailing 0
    is_zero = np.concatenate(([0], np.equal(values, 0).view(np.int8), [0]))
    is_zero_diff = np.abs(np.diff(is_zero))

    # zero_runs: [start, end) indices of zero segments (in the padded index)
    zero_bounds = np.where(is_zero_diff == 1)[0].reshape(-1, 2)

    # keep only zero runs longer than min_zero
    long_zero_bounds = zero_bounds[(zero_bounds[:, 1] - zero_bounds[:, 0]) > min_zero]

    # No separator: everything is one strip
    if long_zero_bounds.size == 0:
        return np.array([[0, num_samples]], dtype=int)

    # Convert zero-run boundaries into strip boundaries
    strip_boundaries = np.ravel(long_zero_bounds)

    # Ensure 0 and num_samples are included as strip boundaries
    if strip_boundaries[0] != 0:
        strip_boundaries = np.concatenate(([0], strip_boundaries))
    else:
        strip_boundaries = strip_boundaries[1:]

    if strip_boundaries[-1] != num_samples:
        strip_boundaries = np.concatenate((strip_boundaries, [num_samples]))
    else:
        strip_boundaries = strip_boundaries[:-1]

    return strip_boundaries.reshape(-1, 2)


def find_strip(los, min_zero, optimal=False, min_points=40_000):
    """
    Find contiguous strips along axis 1 where the mean projection is non-zero.

    Parameters
    ----------
    los : sequence
        List of samples with `.grid`.
    min_zero : int
        Minimum run length of zeros to define strips (passed to zero_runs).
    optimal : bool, default False
        If True, sub-select strips to get roughly `min_points` per strip.
    min_points : int, default 40_000
        Target total mass per strip when `optimal=True`.

    Returns
    -------
    strips : ndarray, shape (n_strips, 2)
        Start and end indices of each strip along axis 1.
    sum_strips : ndarray, shape (n_strips,)
        Total mass per strip.
    """
    assert los.all_grids_standard()

    # Mean "tic" profile along axis 1 across all samples
    mean_profile = np.array(
        [s.grid.sum_along_axis(1, boolean=True)[1] for s in los]
    ).mean(axis=0)

    strips = zero_runs(mean_profile, min_zero=min_zero)

    mass_per_strip = np.array(
        [
            np.sum(mean_profile[strip_start:strip_end])
            for strip_start, strip_end in strips
        ]
    )

    if optimal:
        # Avoid empty intervals
        cumulative_mass = np.cumsum(mass_per_strip)
        target_masses = np.arange(0, mass_per_strip.sum(), min_points)

        selected_indices = np.searchsorted(cumulative_mass, target_masses)

        # WARN: case of >40_000 strip
        strips = np.roll(np.roll(strips, 1)[selected_indices, :], -1)

    return strips, mass_per_strip
