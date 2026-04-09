import os
import sys
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2hed


def path_from_s2(path):
    """
    Return the path substring starting at folder 'S2'.
    Example:
      /a/b/S2/x/y/img.tif -> x/y/img.tif
    If 'S2' is not present as a path component, return the full normalized path.
    """
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)

    if "Shikhar 2" in parts:
        i = parts.index("Shikhar 2")
        return os.sep.join(parts[i+1:])
    return norm

def trim_img(raw_image):
   # tolerance for channel equality
   tol = 2   # use 2 for uint8 images

   # Compute per-pixel color difference
   # If max difference between channels is small → grayscale pixel
   diff = np.max(raw_image, axis=2) - np.min(raw_image, axis=2)

   # Boolean mask: True where pixel is grayscale
   gray_pixel = diff <= tol

   # A column is removable if ALL pixels in that column are grayscale
   gray_column = np.all(gray_pixel, axis=0)

   # Keep columns that are NOT fully grayscale
   valid_cols = np.where(~gray_column)[0]

   if len(valid_cols) > 0:
      image_cropped = raw_image[:, :valid_cols[-1] + 1]
   else:
      image_cropped = raw_image
   return image_cropped

def infer_regions_path(image_path):
    return image_path + "_regions.npy"


def compute_stats_from_files(image_path):
    regions_path = infer_regions_path(image_path)

    if not os.path.exists(regions_path):
        raise FileNotFoundError(f"Regions file not found: {regions_path}")

    image = imread(image_path)
    image = image[:, :, :3]
    image = trim_img(image)

    max_channel_value = np.max(image)

    black_mask = (image == [0, 0, 0]).all(axis=2)

    # Replace black pixels with the maximum color
    image[black_mask] = max_channel_value

    hed = rgb2hed(image)
    gray = hed[:, :, 2]   # DAB channel

    # Average of RGB channels
    rgb_avg = image[..., :3].mean(axis=2)

    regions = np.load(regions_path)

    if gray.shape != regions.shape:
        raise ValueError(
            f"Shape mismatch: gray.shape={gray.shape}, regions.shape={regions.shape}"
        )

    if rgb_avg.shape != regions.shape:
        raise ValueError(
            f"Shape mismatch: rgb_avg.shape={rgb_avg.shape}, regions.shape={regions.shape}"
        )

    stats = {}
    labels = {
        1: "low_dab",   # bin 2
        2: "high_dab",  # bin 3
    }

    for i in [1, 2]:
        mask = (regions == i)
        label = labels[i]

        dab_vals = gray[mask]
        rgb_vals = rgb_avg[mask]

        if dab_vals.size > 0:
            stats[f"{label}_mean"] = float(dab_vals.mean())
            stats[f"{label}_std"] = float(dab_vals.std())
        else:
            stats[f"{label}_mean"] = np.nan
            stats[f"{label}_std"] = np.nan

        if rgb_vals.size > 0:
            stats[f"{label}_rgb_mean"] = float(rgb_vals.mean())
            stats[f"{label}_rgb_std"] = float(rgb_vals.std())
        else:
            stats[f"{label}_rgb_mean"] = np.nan
            stats[f"{label}_rgb_std"] = np.nan
    
    print(
        f"{image_path}\n"
        f"  low_dab : hed mean={stats['low_dab_mean']:.6f}, std={stats['low_dab_std']:.6f}\n"
        f"  high_dab: hed mean={stats['high_dab_mean']:.6f}, std={stats['high_dab_std']:.6f}\n"
    )


    return stats


def append_stats_to_csv(image_path, output_csv):
    row_name = path_from_s2(image_path)
    stats = compute_stats_from_files(image_path)

    row_df = pd.DataFrame([stats], index=[row_name])
    row_df.index.name = "file"

    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv, index_col=0)
        df.loc[row_name, row_df.columns] = row_df.iloc[0]
    else:
        df = row_df

    df.to_csv(output_csv)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <image_path> <output_csv>")
        sys.exit(1)

    image_path = sys.argv[1]
    output_csv = sys.argv[2]

    append_stats_to_csv(image_path, output_csv)