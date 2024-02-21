import pandas as pd
import os

Gans = ["ProGAN", "StyleGAN", "ProjectedGAN", "Diff-StyleGAN2", "Diff-ProjectedGAN"]
Diffs = ["DDPM", "IDDPM", "ADM", "PNDM", "LDM"]


def get_files_from_path(path: str, extension: str = None):
    """
    Returns a list of files from a given path
    """
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if extension is not None and extension in file:
                files.append(os.path.join(r, file))
            elif extension is None:
                files.append(os.path.join(r, file))
    return files


def combine_csv_files(directory_path: str):
    paths = get_files_from_path(directory_path)
    data_types = ["train", "val", "test"]

    for data_type in data_types:
        # Separate paths for GAN and Diff
        gan_paths = [
            path
            for path in paths
            if any(gan in path for gan in Gans) and data_type in path
        ]
        diff_paths = [
            path
            for path in paths
            if any(diff in path for diff in Diffs) and data_type in path
        ]

        # Combine GAN data
        if gan_paths:
            gan_data = pd.concat(
                [pd.read_csv(path, delimiter=" ", header=None) for path in gan_paths],
                ignore_index=True,
            )
            gan_data.to_csv(
                f"{directory_path}/{data_type}_data_gan.csv",
                index=False,
                sep=" ",
                header=False,
            )

        # Combine Diff data
        if diff_paths:
            diff_data = pd.concat(
                [pd.read_csv(path, delimiter=" ", header=None) for path in diff_paths],
                ignore_index=True,
            )
            diff_data.to_csv(
                f"{directory_path}/{data_type}_data_diff.csv",
                index=False,
                sep=" ",
                header=False,
            )


# Example usage
directory_path = "/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/data/csv"  # Replace with your actual directory path
combine_csv_files(directory_path)
