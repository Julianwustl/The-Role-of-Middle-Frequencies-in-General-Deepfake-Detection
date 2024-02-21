import numpy as np
from scipy import ndimage
from scipy.fft import dctn
from PIL import Image
import torch
import scipy.misc
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_pil_image


def _validate_dims(array: np.ndarray) -> None:
    if len(array.shape) != 3:
        raise ValueError(
            f"Image array should have 3 dimensions, got {len(array.shape)}."
        )


def log_scale(array: np.ndarray) -> np.ndarray:
    """Take absolute of array and scale logarithmically. Avoid numerical error by adding small epsilon."""
    return np.log(np.abs(array) + 1e-12)


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


def save_torch_to_grey(mask, save_path):
    im = Image.fromarray(mask.numpy())
    im = im.point(lambda x: x * 255).convert("L")
    im.save(save_path)


def save_torch_to_rgb(image, save_path):
    if image.min() < 0:
        image = (image + 1) / 2  # Scale to [0, 1]

    # Scale to [0, 255] and convert to uint8
    # image = (image * 255).byte()

    # If the tensor is on the GPU or has gradient, move to CPU and remove gradient
    image = image.cpu().detach()

    # Convert the tensor to a PIL image
    image = ToPILImage()(image)

    # Save the image
    image.save(save_path)


def save_normalized_image(
    image, save_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    # The image should now be in the range [0, 1]
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).byte()
    # If the tensor is on the GPU or has gradient, move to CPU and remove gradient
    image = image.cpu().detach()
    # Convert the tensor to a PIL image
    pil_image = to_pil_image(image)
    # Save the image
    pil_image.save(save_path)


def save_np_to_rgb(image, save_path):
    im = Image.fromarray(image.transpose((1, 2, 0)))
    im.save(save_path)


def load_and_preprocess_image(image_path, hight: int = 224, width: int = 224):
    opened_image = Image.open(image_path).convert("RGB")
    resized_image = opened_image.resize((hight, width))
    image_tensor = torch.from_numpy(np.array(resized_image))
    return image_tensor.permute(2, 0, 1)


def read_and_resize_image(image_path, hight: int = 224, width: int = 224) -> np.array:
    opened_image = Image.open(image_path).convert("RGB")
    resized_image = opened_image.resize((hight, width))
    return np.array(resized_image)


def load_and_preprocess_grey_image(
    image_path,
    hight: int = 448,
    width: int = 448,
):
    opened_image = Image.open(image_path).convert("L")
    resized_image = opened_image.resize((hight, width))
    return torch.from_numpy(np.array(resized_image))


def dct(array: np.ndarray):
    """Compute two-dimensional DCT for an array of images."""
    _validate_dims(array)

    dct_coeffs = dctn(array, axes=(1, 2), norm="ortho")
    return dct_coeffs
