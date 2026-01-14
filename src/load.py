import sys
import xml.etree.ElementTree as ET
from datetime import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy_indexed as npi
import opentims_bruker_bridge
import pandas as pd
import pyopenms as oms
import scipy.sparse as sp
from opentimspy.opentims import OpenTIMS

from grid2d import Grid2D

transform = {
    "tof": lambda m, e, l, v: np.sqrt(m * l**2 / (2 * e * v)),
    "quadrupole": None,
    "fticr": lambda m, e, b: e * b / m,
    "orbitrap": lambda m, k: np.sqrt(k / m),
}

# TODO:


def tmz(mz, analyser):
    transform[analyser](mz)


def fit_tof_to_mz_all(
    tof_indices, mz_values
):  # NOTE: Theorical fit (not perfect) + error stonks with squaring
    """
    Fit sqrt(mz) = a + b * tof  (so mz = (a + b * tof)**2)
    using all data via closed-form linear regression.

    Parameters
    ----------
    tof_indices : array-like
        1D array of TOF indices (x).
    mz_values : array-like
        1D array of m/z values (same length as tof_indices).

    Returns
    -------
    a : float
        Intercept in sqrt(mz) space.
    b : float
        Slope in sqrt(mz) space.
    """
    x = np.asarray(tof_indices, dtype=np.float64)
    y = np.sqrt(np.asarray(mz_values, dtype=np.float64))

    n = x.size
    if n < 3:
        raise ValueError(
            "Need at least 3 points to estimate sigma and parameters reliably."
        )

    mean_x = x.mean()
    mean_y = y.mean()

    # Centered sums
    Sxx = np.dot(x - mean_x, x - mean_x)
    Sxy = np.dot(x - mean_x, y - mean_y)

    # Regression coefficients
    b = Sxy / Sxx
    a = mean_y - b * mean_x

    return a, b


def make_tof_to_mz_converter(tof_indices, mz_values):
    """
    Fit calibration and return a function f(tof) -> mz.
    """
    a, b = fit_tof_to_mz_all(
        tof_indices,
        mz_values,
    )

    return a, b


def d(path):
    D = OpenTIMS(path)
    df = pd.DataFrame.from_dict(
        D.query(
            frames=D.ms1_frames,
            columns=("retention_time", "tof", "intensity", "mz"),  # TODO: optional mz
        )
    )

    df = df[["retention_time", "tof", "intensity"]]
    # TODO: in case of more dimension
    df = (
        df.groupby(by=["retention_time", "tof"], as_index=True)[["intensity"]]
        .sum()
        .reset_index()
    )

    return df


def d_convert(path):
    D = OpenTIMS(path)
    frames_rt = D.retention_times[D.ms1_frames - 1]
    df = pd.DataFrame.from_dict(
        D.query(
            frames=D.ms1_frames,
            columns=("tof", "mz"),  # TODO: optional mz
        )
    )
    reg_a, reg_b = make_tof_to_mz_converter(
        df.tof.values, df.mz.values
    )  # TODO: add this property to list_of_spectrum or spectrum?
    return frames_rt, reg_a, reg_b


def d_meta(path):

    path = Path(path)
    path_to_read = path.parent.joinpath(f"{path.stem}.d/SampleInfo.xml")

    if not path_to_read.is_file():
        raise FileNotFoundError("no access to bruker datafile")

    root = ET.parse(path_to_read).getroot()
    date_time_str = root.find("AnalysisHeader").attrib["CreationDateTime"]
    date_time = dt.fromisoformat(date_time_str)
    # print(D.globalmetadata)

    return {"date_time": date_time}


def cache(path):
    """
    load df a cache directory.
    returns df
    """
    path = Path(path)

    df = pd.read_feather(path / "df.feather")
    return df


def cache_convert(path):
    """
    load convert from a cache directory.
    returns (frames_rt)
    """
    path = Path(path)

    convert = np.load(path / "convert.npz", allow_pickle=False)
    frames_rt = convert["frames_rt"]
    reg_a, reg_b = convert["tof_to_mz"]

    return frames_rt, reg_a, reg_b


def cache_meta(path):
    """
    load meta from a cache directory.
    returns date_time
    """
    path = Path(path)
    meta = np.load(path / "meta.npz", allow_pickle=True)
    date_time = dt.fromtimestamp(float(meta["date_time"]))

    return {"date_time": date_time}


def cache_grid(path):
    path = Path(path)
    data = sp.load_npz(path / "grid_data.npz")
    coord0 = np.load(path / "grid_coord0.npy")
    coord1 = np.load(path / "grid_coord1.npy")
    return Grid2D(data, coord0, coord1, axis_names=("rt", "tmz"))


def toy(path):
    """
    Load a tiny synthetic spectrum stored as CSV with columns:
    rt, tmz, int
    (e.g. generated from PNGs by tools/png_to_toy.py)
    """
    path = Path(path)
    df = pd.read_csv(path)

    expected_cols = {"rt", "tmz", "int"}
    if set(df.columns) != expected_cols:
        raise ValueError(
            f"toy file {path} must have exactly columns {expected_cols}, "
            f"got {set(df.columns)}"
        )

    # Ensure types and ordering
    df["rt"] = df["rt"].astype(float)
    df["tmz"] = df["tmz"].astype(float)
    df["int"] = df["int"].astype(float)

    df = df.sort_values(["rt", "tmz"]).reset_index(drop=True)
    return df


def toy_convert(path):
    """
    Build frames_rt axis for toy spectra.

    Convention:
    - rt values are integer indices in [0, H-1].
    - We reconstruct frames_rt as np.arange(0, max_rt + 1).
    """
    df = toy(path)
    if df.empty:
        raise ValueError(f"Empty toy spectrum: {path}")

    max_rt = int(df["rt"].max())
    frames_rt = np.arange(0, max_rt + 1, dtype=float)
    return frames_rt, 0, 0  # FIX: reg a,b


def toy_meta(path):
    """
    Minimal metadata for toy spectra.
    Use file mtime as a pseudo acquisition date.
    """
    path = Path(path)
    date_time = dt.fromtimestamp(path.stat().st_mtime)
    return {"date_time": date_time}


def mzml(path):
    reader = pymzml.run.reader(path)
    # todo check case of ionic mob,date_time

    data = []
    frames_rt = []
    # iterate through each spectrum in the mzml file
    for spectrum in reader:
        if spectrum["ms level"] == 1:  # only processing ms1 spectra
            # extracting m/z and intensities
            mass_array = spectrum.mz  # m/z values
            intensity_array = spectrum.i  # intensity values
            scan_id = spectrum.id  # scan id
            scan_time = spectrum.scan_time

            frames_rt.append(scan_time)
            if len(mass_array) == 0:
                print("empty frame")
            for mz, intensity in zip(mass_array, intensity_array):
                data.append(
                    {"retention_time": scan_time, "mz": mz, "intensity": intensity}
                )

    # create a dataframe
    df = pd.DataFrame(data)
    tmz(df.mz.values, analyser)
    return (df, tof_mz, frames_rt, date_time)
