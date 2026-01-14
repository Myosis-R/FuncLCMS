import numpy as np

import fisher_rao as fr
import hot
import validation
import plot
import save
import sot
from pipeline import Pipeline, PipelineStep
from spectrum import List_of_Spectrum


def main():

    # params_data = {
    #     "analyser": "tof",
    #     "folder": "Data/Eglantine",
    #     "format": "cache",  # TODO: both format at the same time
    #     "name_specification": "date-batch-type-order-drop",
    #     "name_tweak": True,
    #     "pattern": "[F]-QC-1[12]",  # TODO: check if one
    # }

    params_data = {
        "analyser": "synthetic",
        "folder": "Data/Toy",  # "Data/Eglantine"
        "format": "toy",  # TODO: both format at the same time
        "name_specification": "name",
        "name_tweak": False,
    }

    steps = [
        # Per-spectrum step
        # PipelineStep(
        #     obj=List_of_Spectrum,
        #     attr="write",
        #     args={"with_grid": False},
        #     mode="per_list",
        #     name="save_spectrum",
        # ),
        # Per-list step: standardize_all on List_of_Spectrum
        PipelineStep(
            obj=List_of_Spectrum,
            attr="standardize_all",
            args={
                "standardize_rt": True,
                "standardize_tmz": True,
                "rt_axis": np.arange(-1, 17),  # np.linspace(19, 1200, 2000),
                "tmz_axis": None,
            },
            mode="per_list",
            name="standardize_all",
        ),
        PipelineStep(
            obj=hot,
            attr="hierarchical_ot",
            args={
                "min_zero": 15,
                "min_points": 40000,
                "dust_cost": 500,
                "dust_cost_comp": 1000,
                "cost": "sqeuclidean",
            },
            mode="per_list",
            name="strip_ot",
        ),
        # PipelineStep(
        #     obj=sot,
        #     attr="strip_ot",
        #     args={
        #         "min_zero": 15,
        #         "min_points": 40000,
        #         "dust_cost": 500,  # ~20 index ~ 10 seconds !nop
        #         "cost": "sqeuclidean",
        #         "binarize": False,
        #     },
        #     mode="per_list",
        #     name="strip_ot",
        # ),
        # PipelineStep(
        #     obj=fr,
        #     attr="alignment_FR",
        #     args={},
        #     mode="per_list",
        #     name="Fisher_Rao",
        # ),
        # Per-list plotting
        PipelineStep(
            obj=plot,
            attr="TICs",
            args={"axis": 0, "boolean": False},
            mode="per_list",
            name="TICs_false",
        ),
        PipelineStep(
            obj=validation,
            attr="validate_compounds",
            args={
                "csv_path": "/Data/Eglantine/List-QC-compounds.csv",
                "delta": 5,
                "plot": True,
            },
            mode="per_list",
            name="TICs_false",
        ),
    ]

    # TODO: check pipeline consistency and optim sparse

    los = List_of_Spectrum(params_data)
    los.sort()
    breakpoint()

    pipeline = Pipeline(steps)
    pipeline.run(los)
    pipeline.print_timings()


if __name__ == "__main__":
    main()
