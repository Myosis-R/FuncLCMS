from functools import reduce

import cc3d
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from sklearn import manifold, neighbors


def query_max(s):
    peak = s.df.query("int==int.max()").values[0, :]
    del s.df
    return peak


def rt_EDTA(los):  # NOTE: EDTA <=> higher pic

    pos_EDTA = np.array([query_max(s) for s in los])  # HACK:
    print(pos_EDTA)
    times = los.specification().time.values
    levels, categories = pd.factorize(los.specification().type)
    fig, ax = plt.subplots()
    for i in range(len(categories)):
        ax.scatter(times[levels == i], pos_EDTA[levels == i, 1], label=categories[i])
    ax.set_ylabel("rt")
    ax.set_xlabel("temps des échantillons")
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()
    print(pos_EDTA.shape)


def zero_runs(a, min_zero):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    ranges = ranges[(ranges[:, 1] - ranges[:, 0]) > min_zero, :]
    return ranges


def strip_tmz(los):
    assert los.all_grids_standard()
    tics = np.array([s.grid.sum_along_axis(axis=1, boolean=True)[1] for s in los])
    print(tics.shape)
    sum_tics = tics.sum(axis=0) / len(tics)
    zeros = zero_runs(sum_tics, 15)
    fig, ax = plt.subplots()
    # ax.bar(zeros[:, 0], height=10, width=(zeros[:, 1]-zeros[:, 0]),align="edge")
    sum_nonzeros = np.array(
        [np.sum(sum_tics[zeros[i, 1] : zeros[i + 1, 0]]) for i in range(len(zeros) - 1)]
    )
    ax.bar(
        zeros[:-1, 1],
        height=sum_nonzeros,
        width=(zeros[:-1, 1] - zeros[1:, 0]),
        align="edge",
        color="red",
    )
    plt.show()


def matrix(los):
    # for s in los:
    #     fig,ax = plt.subplots()
    #     ax.imshow(np.log(s.ds.todense()+1),interpolation=None)
    # plt.show()

    fps = 4
    snapshots = [np.log(s.ds.todense() + 1) for s in los]

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(8, 8))

    a = snapshots[0]
    im = plt.imshow(a, interpolation="none", aspect="auto", vmin=0, vmax=10)

    def animate_func(i):
        im.set_array(snapshots[i])
        return [im]

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=len(los),
        interval=1000 / fps,  # in ms
    )

    anim.save("test_anim.mp4", fps=fps, extra_args=["-vcodec", "libx264"])

    print("Done!")


def TICs(los, axis=0, boolean=False):
    assert los.all_grids_standard()
    fig, ax = plt.subplots()
    for i, s in enumerate(los):
        coord, tic = s.grid.sum_along_axis(
            axis=axis, boolean=boolean
        )  # TODO: check standard
        ax.plot(coord, tic, label=i)
    plt.legend()
    plt.show()


def dist_l2(los):
    # FIX: los.is_standard
    l = len(los)
    dist = np.zeros((l, l))
    for i in range(l):
        for j in range(i + 1, l):
            diff = (los[i].grid - los[j].grid) ** 2
            dist[i, j] = np.sqrt(
                diff._data.sum()
            )  # or diff._data.power(2).sum() if L2^2
    dist = dist + dist.T
    return dist


def plot_dist(los, dist):
    levels, categories = pd.factorize(los.specification().batch)
    mds = manifold.MDS(
        n_components=2,
        max_iter=3000,
        eps=1e-9,
        n_init=1,
        random_state=42,
        dissimilarity="precomputed",
        n_jobs=1,
    )
    X_mds = mds.fit(dist).embedding_

    fig, ax = plt.subplots()
    ax.set_xlabel("coordonnée 1")
    ax.set_ylabel("coordonnée 2")
    points = ax.scatter(X_mds[:, 0], X_mds[:, 1], c=levels)
    handles, labels = points.legend_elements(prop="colors", alpha=0.6)
    legend = ax.legend(handles, categories, loc="upper right", title="Lot")
    txt = los.specification().order.values
    texts = []
    for i in range(len(txt)):
        texts.append(ax.text(X_mds[i, 0], X_mds[i, 1], "QC-{}".format(txt[i])))
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="->", color="red"),
        avoid_self=True,
        ensure_inside_axes=True,
    )
    plt.show()


def cluster_bool(los):
    dist = dist_l2(los)
    plot_dist(los, dist)
