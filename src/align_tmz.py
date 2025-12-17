import numpy as np
from scipy.optimize import minimize


def translation_f(t, ref, s):  # TODO: optim roll, exact result ?
    t = float(t)
    t, c = np.divmod(t, 1)
    s = c * np.roll(s, -t - 1) + (1 - c) * np.roll(s, -t)
    return np.linalg.norm(ref - s)


def translation_grad_tmz(los):  # TODO: change ref, add axis
    assert los.all_grids_standard()
    tics = np.array([s.grid.sum_along_axis(0, boolean=True)[1] for s in los])
    ref = tics[0]
    for i, s in enumerate(los[1:], start=1):
        translation = minimize(translation_f, 0, args=(ref, tics[i]))[
            "x"
        ]  # TODO: bound
        s.grid.interpolate_axis(
            axis=1, out_coord=s.ds_coord[1], in_coord=(s.ds_coord[1] - translation)
        )
    assert los.all_grids_standard()
