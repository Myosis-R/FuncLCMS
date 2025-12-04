import time
from typing import Any, Dict, List


class PipelineStep:
    """
    A very simple pipeline step.

    - obj:  module / class / instance that holds the function
    - attr: name of the function/method to call (via getattr(obj, attr))
    - args: kwargs to pass to that function
    - mode:
        - "per_spectrum": call the function once for each spectrum in the list
        - "per_list":     call the function once on the entire list

    Timing:
    - last_duration: time (seconds) for the last call to this step
    """

    def __init__(self, obj: Any, attr: str,
                 args: Dict[str, Any] | None = None,
                 mode: str = "per_list",
                 name: str | None = None) -> None:
        self.obj = obj
        self.attr = attr
        self.args = args or {}
        self.mode = mode  # "per_spectrum" or "per_list"
        self.name = name or attr

        self.last_duration: float | None = None

    def run(self, data_list: Any) -> None:
        """
        Run this step on:
          - each element of data_list if mode == "per_spectrum"
          - the whole data_list if mode == "per_list"
        """
        func = getattr(self.obj, self.attr)

        t0 = time.perf_counter()

        if self.mode == "per_spectrum":
            for elem in data_list:
                func(elem, **self.args)
        elif self.mode == "per_list":
            func(data_list, **self.args)
        else:
            raise ValueError(f"Unknown mode: {self.mode!r}")

        t1 = time.perf_counter()
        self.last_duration = t1 - t0


class Pipeline:
    """
    Holds a list of PipelineStep and runs them in order.

    - steps: list of PipelineStep
    - total_duration: total time for the last run(data_list) call
    """

    def __init__(self, steps: List[PipelineStep]) -> None:
        self.steps = steps
        self.total_duration: float | None = None

    def run(self, data_list: Any) -> None:
        t0 = time.perf_counter()

        for step in self.steps:
            step.run(data_list)

        t1 = time.perf_counter()
        self.total_duration = t1 - t0

    def print_timings(self) -> None:
        print("Pipeline timings:")
        for step in self.steps:
            if step.last_duration is None:
                print(f"  - {step.name}: not run")
            else:
                print(f"  - {step.name}: {step.last_duration:.3f} s")
        if self.total_duration is not None:
            print(f"Total: {self.total_duration:.3f} s")
