"""Analyze frequency characteristics of images."""
import argparse
import json
from functools import partial
from pathlib import Path
from typing import List, Tuple


import cytoolz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Union
import numpy as np
from scipy import ndimage
from scipy.fft import dctn
import os
import warnings
import cv2
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
from optimizer import with_caching, parallel_map
import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from typing import List, Optional
import re
import matplotlib as mpl
from torchvision.transforms.functional import resize
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable
from torchvision.datasets.folder import (
    ImageFolder,
    default_loader,
    has_file_allowed_extension,
)
from fft import butterworth_lowpass_filter_mask, filter_mask
from style import configure_matplotlib

configure_matplotlib()


def get_figsize(
    width: Union[int, float, str] = "iclr",
    ratio: float = (5**0.5 - 1) / 2,  # - 0.2, TODO attention!
    fraction: float = 1.0,
    nrows: int = 1,
    ncols: int = 1,
) -> Tuple[float, float]:
    """
    Return width and height of figure in inches.

    :param width: Width of figure in pt or name of conference template.
    :param ratio: Ratio between width and height (height = width * ratio). Defaults to golden ratio (~0.618).
    :param fraction: Scaling factor for both width and height.
    :param nrows, ncols: Number of rows/columns if subplots are used.
    :return: Tuple containing width and height in inches.
    """
    conference_lut = {"iclr": 397.48499}
    if isinstance(width, str):
        if width not in conference_lut:
            raise ValueError(
                f"Available width keys are: {list(conference_lut.keys())}, got:"
                f" {width}."
            )
        width = conference_lut[width]
    height = width * ratio * (nrows / ncols)
    factor = 1 / 72.72 * fraction
    return width * factor, height * factor


def _validate_dims(array: np.ndarray) -> None:
    if len(array.shape) != 3:
        raise ValueError(
            f"Image array should have 3 dimensions, got {len(array.shape)}."
        )


def dct(array: np.ndarray):
    """Compute two-dimensional DCT for an array of images."""
    _validate_dims(array)

    dct_coeffs = dctn(array, axes=(1, 2), norm="ortho")
    return dct_coeffs


def fft(
    array: np.ndarray, hp_filter: bool = False, hp_filter_size: int = 3
) -> np.ndarray:
    """
    Compute two-dimensional FFT for an array of images, with optional highpass-filtering.
    """
    _validate_dims(array)

    if hp_filter:
        array = array - ndimage.median_filter(
            array, size=(1, hp_filter_size, hp_filter_size)
        )
    fft_coeffs = np.fft.fft2(array, axes=(1, 2), norm="ortho")
    spectrum = np.fft.fftshift(fft_coeffs, axes=(1, 2))
    return spectrum


def center_crop(array: np.ndarray, size: int) -> np.ndarray:
    """Center crop an image or array of images to square of length size."""
    if size > min(array.shape[-2:]):
        raise ValueError(
            "Image cannot be smaller than crop size (256, 256), got"
            f" {array.shape[-2:]}."
        )

    top = array.shape[-2] // 2 - size // 2
    left = array.shape[-1] // 2 - size // 2
    return array[..., top : top + size, left : left + size]


def array_from_imgdir(
    directory: Path,
    crop_size: int = 256,
    grayscale: bool = False,
    lowpass_filter: bool = False,
    num_samples: int = None,
    num_workers: int = 1,
    lowpass_size: int = 5,
) -> np.ndarray:
    """
    Given a directory, load all (or num_samples if specified) images and return them as a numpy array.
    Optionally crop and/or apply grayscale conversion.
    """
    paths = []
    for path in directory.iterdir():
        if path.suffix.lower() == ".png":
            paths.append(path)
        if num_samples is not None and len(paths) == num_samples:
            break
    if num_samples and len(paths) < num_samples:
        warnings.warn(f"Found only {len(paths)} images instead of {num_samples}.")
    mask = butterworth_lowpass_filter_mask(crop_size, crop_size, lowpass_size, 1)
    if lowpass_filter:

        def loader(path):
            image = np.array(Image.open(path).convert("L"))
            image = torch.from_numpy(image)
            # image = cv2.resize(image, dsize=(224, 224))
            filter = filter_mask(image, mask)
            return filter.numpy()

    elif grayscale:

        def loader(path):
            return np.array(Image.open(path).convert("L"))

    else:

        def loader(path):
            return np.array(Image.open(path))

    if num_workers == 1:
        array = np.array(list(map(loader, paths)))
    else:
        array = np.array(
            Parallel(n_jobs=num_workers)(delayed(loader)(path) for path in paths)
        )
    if array.shape[1:] != (crop_size, crop_size):
        print(f"Cropping from {array.shape[1:]} to {crop_size, crop_size}.")
    array = center_crop(array, size=crop_size)

    array = array / 127.5 - 1  # scale to [-1, 1]
    return array


@with_caching(keys=["img_dir", "func", "crop_size", "grayscale", "lowpass_size"])
def apply_to_imgdir(
    img_dir: Path,
    func: Callable,
    crop_size: int = 256,
    grayscale: bool = False,
    lowpass_filter: bool = False,
    num_samples: int = None,
    num_workers: int = 1,
    lowpass_size: int = 5,
) -> np.ndarray:
    """Convenience function to load images from directory into numpy array and apply function to it."""
    return func(
        array_from_imgdir(
            directory=img_dir,
            grayscale=grayscale,
            crop_size=crop_size,
            lowpass_filter=lowpass_filter,
            lowpass_size=lowpass_size,
            num_samples=num_samples,
            num_workers=num_workers,
        )
    )


def get_mean_std(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.mean(array, axis=0), np.std(array, axis=0)


TRANSFORMS = {
    "dct": cytoolz.compose_left(dct, get_mean_std),
    "fft": cytoolz.compose_left(fft, get_mean_std),
    "fft_hp": cytoolz.compose_left(
        partial(fft, hp_filter=True, hp_filter_size=3), get_mean_std
    ),
    # "density": cytoolz.compose_left(spectral_density, get_mean_std),
}


@mpl.rc_context({"figure.dpi": 1000, "figure.constrained_layout.use": False})
def plot_spectra(
    data: np.ndarray,
    labels: List[str],
    width: float,
    log: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fixed_height: int = None,
) -> plt.Figure:
    """Plot 2D frequency spectra."""
    # get vim/vmax
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    # figure setup
    fig = plt.figure(
        figsize=(
            width,
            width / (len(labels) if fixed_height is None else fixed_height) + 0.1,
        )
    )
    grid = AxesGrid(
        fig, 111, nrows_ncols=(1, len(labels)), cbar_mode="edge", cbar_size="10%"
    )
    for ax in grid:
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    # plot
    if log:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        vmin, vmax = None, None
    else:
        norm = None

    for i, (array, label) in enumerate(zip(data, labels)):
        im = grid[i].imshow(
            array,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            cmap=sns.color_palette("mako", as_cmap=True),
        )
        if len(data) != 1:
            grid[i].set_title(label, pad=2)

        if i == len(labels) - 1:
            grid[-1].cax.colorbar(im)

    plt.tight_layout(pad=0.5)
    return fig


def sanitize(s: str) -> str:
    """Sanitize filename by removing illegal characters."""
    s = s.replace(" ", "_")
    return re.sub("[^A-Za-z0-9._-]", "", s)


def get_filename(
    file_format: str,
    kind: str,
    variant: Optional[str] = None,
    experiment: Optional[str] = None,
    data: Optional[str] = None,
    identifiers: Optional[Union[str, List[str]]] = None,
) -> str:
    """
    Return standardized filename with dashes to separate entities. Raises an error if dashes are used within an entity.

    :param file_format: File format.
    :param kind: Kind of output.
    :param variant: Variant for this kind of output.
    :param experiment: Name of the experiment.
    :param data: String identifying the used data.
    :param identifiers: One or multiple additional identifiers put at the end of the filename.
    :return: Composed filename.
    """
    if not isinstance(identifiers, list):
        identifiers = [identifiers]
    parts = []
    for part in [kind, variant, experiment, data] + identifiers:
        if part is not None:
            part.replace("-", "_")
            parts.append(sanitize(part))
    return "-".join(parts) + f".{file_format.lower().lstrip('.')}"


def main(
    image_root: Path,
    output_root: Path,
    transform: str,
    plot_subdir: Path,
    moment: str,
    log: bool,
    vmin: float,
    vmax: float,
    img_dirs: List[str],
    overwrite: bool,
    num_workers: int,
    experiment: str,
    fraction: float,
    zoom: bool,
    diff: bool,
    fixed_height: int,
    lowpass_size: int,
):
    output_dir = output_root / "frequency_analysis"
    plot_dir = output_dir / "plots"
    if plot_subdir is not None:
        plot_dir = plot_dir / plot_subdir
    plot_dir.mkdir(exist_ok=True, parents=True)

    # compute mean and std of transform
    mean_std = parallel_map(
        apply_to_imgdir,
        [image_root / dirname for dirname in img_dirs],
        num_workers=num_workers,
        mode="multiprocessing",
        func_kwargs=dict(
            func=TRANSFORMS[transform],
            grayscale=True,
            lowpass_filter=True,
            lowpass_size=lowpass_size,
            cache_dir=output_dir / "cache",
            overwrite=overwrite,
            crop_size=256,
        ),
    )
    means, stds = zip(*mean_std)
    df = pd.DataFrame({"mean": means, "std": stds}, index=img_dirs)

    # plot
    labels = img_dirs
    if (image_root / "labels.json").exists():
        with open(image_root / "labels.json") as f:
            label_updates = json.load(f)
        labels = [
            label_updates[label] if label in label_updates else label
            for label in labels
        ]

    data = np.stack(df[moment])
    """if transform == "density":
        plt.figure(figsize=get_figsize(fraction=fraction))
        if diff:
            data = data[1:] / data[0] - 1
            plot_power_spectrum(
                data=data, labels=labels[1:], log=log, zoom=zoom, first_black=False
            )
            plt.ylabel("Spectral Density Error")
            plt.ylim(-1, 1)
        else:
            plot_power_spectrum(data=data, labels=labels, log=log, zoom=zoom)
            plt.ylim(10**-5, 10**3)"""
    plot_spectra(
        data=np.abs(data),
        labels=labels,
        width=get_figsize(fraction=fraction)[0],
        log=log,
        vmin=vmin,
        vmax=vmax,
        fixed_height=fixed_height,
    )

    filename = get_filename(
        file_format="pdf",
        kind=transform,
        variant=moment,
        data="_".join(img_dirs),
        experiment=experiment,
        identifiers="diff" if diff else None,
    )
    plt.savefig(plot_dir / filename)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_root", type=Path, help="Root of image directory.")
    parser.add_argument("output_root", type=Path, help="Output directory.")
    parser.add_argument("transform", choices=TRANSFORMS, help="Transform to apply.")
    parser.add_argument("--plot-subdir", type=Path)
    parser.add_argument(
        "--moment",
        choices=["mean", "std"],
        default="mean",
        help="Whether to plot mean or std.",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Whether to plot difference between real and other dirs.",
    )
    parser.add_argument(
        "--img-dirs",
        nargs="+",
        required=True,
        help="Image directories to analyze, order is maintained.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute instead of using existing data.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers (default: 4)."
    )
    parser.add_argument(
        "--experiment",
        default="default",
        help="Custom experiment name to use for output files.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1,
        help="Fraction of paper width to use for plot.",
    )
    parser.add_argument(
        "--log", action="store_true", help="Whether to plot in log scale."
    )
    parser.add_argument("--vmin", type=float, help="Argument vmin for imshow.")
    parser.add_argument("--vmax", type=float, help="Argument vmax for imshow.")
    parser.add_argument(
        "--zoom",
        action="store_true",
        help="Whether to show zoomed in are at highest frequencies.",
    )
    parser.add_argument(
        "--fixed-height",
        type=int,
        help="Adjust height as if there were this many spectra.",
    )
    parser.add_argument(
        "--lowpass_size", type=int, default=5, help="Number of workers (default: 4)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
