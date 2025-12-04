import numpy as np

import fisher_rao as fr
import optimal_transport as ot
import plot
import save
import tools
from pipeline import Pipeline, PipelineStep
from spectrum import List_of_Spectrum


def main():

    params_data = {
        "analyser": "tof",
        "folder": "Data/Eglantine",
        "format": "cache",  # TODO: both format at the same time
        "name_specification": "date-batch-type-order-drop",
        "name_tweak": True,
        "pattern": "[F]-QC-[^0]+",  # TODO: check if one
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
                "rt_axis": np.linspace(19, 1200, 2000),
                "tmz_axis": None,
            },
            mode="per_list",
            name="standardize_all",
        ),
        # Per-list plotting
        PipelineStep(
            obj=plot,
            attr="TICs",
            args={"axis": 1, "boolean": False},
            mode="per_list",
            name="TICs_false",
        ),
        PipelineStep(
            obj=plot,
            attr="TICs",
            args={"axis": 1, "boolean": True},
            mode="per_list",
            name="TICs_true",
        ),
    ]

    # TODO: check pipeline consistency and optim sparse

    los = List_of_Spectrum(params_data)
    los.sort()

    pipeline = Pipeline(steps)
    pipeline.run(los)
    pipeline.print_timings()


if __name__ == "__main__":
    main()
