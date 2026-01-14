# DEV only
from pathlib import Path

import numpy as np
from PIL import Image
import pandas as pd


def png_to_toy(png_path, csv_path=None, threshold=0):
    """
    Convert a grayscale PNG to a sparse CSV with columns (rt, tmz, int).

    - rt  = row index [0 .. H-1]
    - tmz = column index [0 .. W-1]
    - int = pixel intensity (uint8 or float), only for pixels > threshold
    """
    png_path = Path(png_path)
    if csv_path is None:
        csv_path = png_path.with_suffix(".toy")
    else:
        csv_path = Path(csv_path)

    # Load image and ensure grayscale
    img = Image.open(png_path).convert("L")  # "L" = 8-bit grayscale
    arr = np.array(img, dtype=np.float64)    # shape (H, W)

    H, W = arr.shape

    # Find non-zero (or > threshold) pixels
    if threshold > 0:
        mask = arr > threshold
    else:
        mask = arr > 0

    rows, cols = np.nonzero(mask)
    intensities = arr[rows, cols]

    # Build DataFrame: rt=row, tmz=col, int=intensity
    df = pd.DataFrame(
        {
            "rt": rows.astype(int),
            "tmz": cols.astype(int),
            "int": intensities,
        }
    )

    # Sort by (rt, tmz) so Spectrum._build_local_grid grouping works as expected
    df = df.sort_values(["rt", "tmz"]).reset_index(drop=True)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} from {png_path}")


def convert_folder(input_folder, pattern="*.png", threshold=0):
    input_folder = Path(input_folder)
    for png_path in input_folder.glob(pattern):
        png_to_toy(png_path, threshold=threshold)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="PNG file or folder")
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    p = Path(args.input)
    if p.is_dir():
        convert_folder(p, threshold=args.threshold)
    else:
        png_to_toy(p, threshold=args.threshold)
