import fdasrsf as fs
import matplotlib.pyplot as plt
import numpy as np


def alignment_FR(los, **params):
    assert los.all_grids_standard()
    boolean = params["binarize"]
    f = np.array(
        [s.grid.sum_along_axis(axis=0, boolean=boolean)[1] for s in los]
    )  # HACK: bool
    time = los[0].grid.coord0
    fdawarp = fs.fdawarp(f.T, time)
    fdawarp.srsf_align(parallel=True)  # FIX: error with scalene ?
    for i in range(len(los)):
        warp = (time[-1] - time[0]) * fdawarp.gam[:, i] + time[
            0
        ]  # pas une fonc, pas sur tout le spec
        assert len(warp) == len(time)
        los[i].grid = los[i].grid.interpolate_axis(0, out_coord=warp, in_coord=time)
        los[i].grid._coords[0] = time

    # fdawarp.plot()
