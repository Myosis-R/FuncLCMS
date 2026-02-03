import numpy as np

from . import hot, sot, fisher_rao, ref_utils, align_tmz, validation, plot, save
from .pipeline import Pipeline, PipelineStep
from .spectrum import List_of_Spectrum


def main():

    params_data = {
        "analyser": "tof",
        "folder": "Data/Eglantine",
        "format": "cache",  # TODO: both format at the same time
        "name_specification": "date-batch-type-order-drop",
        "name_tweak": True,
        "pattern": "[FGH]-QC",  # TODO: check if one
    }

    steps = [
        # PipelineStep(
        #     obj=List_of_Spectrum,
        #     attr="write",
        #     args={"with_grid": False},
        #     mode="per_list",
        #     name="save_spectrum",
        # ),
        PipelineStep(
            obj=List_of_Spectrum,
            attr="standardize_all",
            args={
                "standardize_rt": True,
                "standardize_tmz": True,
                "rt_axis": np.linspace(19, 1200, 2000),
                "tmz_axis": None,
            },
            mode="per_list",
            name="standardize_all",
        ),
        # PipelineStep(
        #     obj=save,
        #     attr="projections",
        #     args={},
        #     mode="per_list",
        #     name="save_projections",
        # ),
        PipelineStep(
            obj=ref_utils,
            attr="first_spec",
            args={},
            mode="per_list",
            name="use first spec as ref",
        ),
        # PipelineStep(
        #     obj=align_tmz,
        #     attr="translation_grad_tmz",
        #     args={},
        #     mode="per_list",
        #     name="align tmz by translation",
        # ),
        PipelineStep(
            obj=hot,
            attr="hierarchical_ot",
            args={
                "min_zero": 15,
                "min_points": 40000,
                "dust_cost": 500,
                "dust_cost_comp": 1000,
                "axis_weights": [1, 15],  # TODO: Tune it
                "cost": "sqeuclidean",
                "n_jobs": 4,
            },
            mode="per_list",
            name="hierarchical OT",
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
        #         "n_jobs": 4,
        #     },
        #     mode="per_list",
        #     name="strip_ot",
        # ),
        # PipelineStep(
        #     obj=fisher_rao,
        #     attr="alignment_FR",
        #     args={
        #         "binarize": False,
        #     },
        #     mode="per_list",
        #     name="Fisher_Rao",
        # ),
        # Per-list plotting
        PipelineStep(
            obj=validation,
            attr="validate_compounds",
            args={
                "csv_path": "Data/Eglantine/List-QC-compounds.csv",
                "delta": 5,
                "plot": True,
                "charge": "pos",
            },
            mode="per_list",
            name="compounds validation",
        ),
        PipelineStep(
            obj=plot,
            attr="TICs",
            args={"axis": 0, "boolean": False},
            mode="per_list",
            name="plot TICs",
        ),
    ]

    # TODO: check pipeline consistency and optim sparse

    los = List_of_Spectrum(params_data)
    los.sort()

    pipeline = Pipeline(steps)
    # id_utils.build_run_id(los, pipeline)  # TODO:
    pipeline.run(los)
    pipeline.print_timings()


if __name__ == "__main__":
    main()
