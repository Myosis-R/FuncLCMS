import numpy as np
import pandas as pd
from pathlib import Path


def load_compounds_table(csv_path):  # TODO: transform mz
    path = Path().absolute().parent.joinpath(csv_path)
    compounds_table = pd.read_csv(str(path))
    compounds_table.columns = ["analyte", "rt", "ccs", "mz", "charge"]
    return compounds_table


def extract_compound(los, target, delta):
    idx = np.searchsorted(los[0].grid.coord1, target[1])
    error = np.zeros((len(los), 2))
    for i, s in enumerate(los):
        data = s.grid.data[:, (idx - delta) : (idx + delta)]
        xic = data.sum(axis=1)
        xmc = data.sum(axis=0)
        rt_error = (xic * s.grid.coord0).sum() / xic.sum() - target[0]
        tmz_error = (
            xmc * s.grid.coord1[(idx - delta) : (idx + delta)]
        ).sum() / xmc.sum() - target[1]
        error[i, :] = [rt_error, tmz_error]
    return error


def validate_compounds(los, **params):
    assert los.all_grids_standard()
    csv_path = params.get("csv_path")
    compounds_table = load_compounds_table(csv_path)
    compounds_table = compounds_table[compounds_table.charge.values == params["charge"]]
    compounds_table["tmz"] = los[0].mz_to_tmz(compounds_table.mz.values)
    error_table = pd.DataFrame()
    for compound in compounds_table[["rt", "tmz"]].values:
        error = extract_compound(los, compound, delta=params["delta"])
        error_table[[f"{compound}_rt", f"{compound}_mz"]] = error
    if params["plot"]:
        error_table.boxplot()
    error_rt = np.nanmedian(np.abs(error_table.values[:, ::2]))
    error_tmz = np.nanmedian(np.abs(error_table.values[:, 1::2]))
    print(
        f"proportion of nan : {np.sum(np.isnan(error_table.values))/error_table.size}"
    )
    print(
        f"measure of consistency in doped compounds. rt : {error_rt}, tmz : {error_tmz}"
    )
