import numpy as np
from PIL import Image
from pathlib import Path

def load_cases(path_folder):

    list_paths = list(
        Path()
        .absolute()
        .parent.joinpath(path_folder)
        .glob("*.{}".format("png"))
    )
    Image.open(path).convert("L")
