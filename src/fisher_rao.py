import fdasrsf as fs
import matplotlib.pyplot as plt
import numpy as np


def alignment_FR(los):  # TODO: check if FR without same time values works
    f = np.array(
        [s.grid.sum_along_axis(axis=0, boolean=True)[1] for s in los]
    )  # HACK: bool
    time = los[0].grid.coord0
    fdawarp = fs.fdawarp(f.T, time)
    fdawarp.srsf_align(parallel=True)
    for i in range(len(los)):
        warp = (time[-1] - time[0]) * fdawarp.gam[:, i] + time[
            0
        ]  # pas une fonc, pas sur tout le spec
        print(warp.shape, time.shape)
        los[i].grid.interpolate_axis(los[i], 0, out_coord=warp, in_coord=time)

    fdawarp.plot()
