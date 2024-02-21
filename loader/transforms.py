from torch import nn
import torch
from core.fft import fft_mask_rgb_filter, create_2dft
from albumentations import fourier_domain_adaptation
from core.image import read_and_resize_image
import random
from PIL import Image
import io
import numpy as np
import cv2

transform_type_map = {
    "band_filter": fft_mask_rgb_filter,
}


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x)["image"], self.transform(x)["image"]]


class StandardTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x, **params):
        return self.transform(x, **params)


class BandFilter(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, fft_mask=None) -> None:
        super().__init__()
        self.fft_mask = fft_mask

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor, **params) -> torch.Tensor:
        return {
            "image": fft_mask_rgb_filter(x, self.fft_mask),
            "label": params["label"],
        }  # BxCxHxW


class JPEGCompression(nn.Module):
    """Module to perform JPEG compression as data augmentation on torch tensors."""

    def __init__(self, compression_rate: int) -> None:
        super().__init__()
        self.compression_rate = compression_rate

    @torch.no_grad()  # disable gradients for efficiency
    def forward(self, x: torch.Tensor, **params) -> torch.Tensor:
        # Convert the tensor to PIL Image for JPEG compression

        _, image = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.compression_rate]
        )
        return {
            "image": cv2.imdecode(image, 1),
            "label": params["label"],
        }
        pil_image = Image.fromarray(x.mul(255).permute(1, 2, 0).byte().numpy())
        buffer = io.BytesIO()
        # Apply JPEG compression
        pil_image.save(buffer, format="JPEG", quality=self.compression_rate)
        buffer.seek(0)

        # Load the compressed image back
        compressed_image = Image.open(buffer)
        compressed_tensor = (
            torch.from_numpy(np.array(compressed_image))
            .div(255)
            .permute(2, 0, 1)
            .float()
        )

        return {
            "image": compressed_tensor,
            "label": params["label"],
        }


class FDAFilter(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, reference_images, beta=0.01) -> None:
        super().__init__()
        self.beta = beta
        self.reference_images = reference_images

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor, **params) -> torch.Tensor:
        image_dict = random.choice(self.reference_images)

        target_img = read_and_resize_image(image_dict["image"])
        # target_img = torch.from_numpy(target_img).permute(2, 0, 1)
        # source_image = source_image.permute(2, 0, 1)
        target_label = image_dict["label"]
        # Randomly choose between butterworth_low_pass or butterworth_high_pass#

        fda_transform_image = fourier_domain_adaptation(
            x.permute(1, 2, 0).numpy(), target_img, self.beta
        )
        if target_label == 1.0:
            params["label"] = torch.tensor(1.0, dtype=torch.float32)

        return {
            "image": torch.from_numpy(fda_transform_image).permute(2, 0, 1),
            "label": params["label"],
        }


class ToFourier(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, fft_mask=None) -> None:
        super().__init__()
        self.fft_mask = fft_mask

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return create_2dft(x)  # BxCxHxW


class ButterworthBandFilter(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, fft_mask=None) -> None:
        super().__init__()
        self.fft_mask = fft_mask

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor, **param) -> torch.Tensor:
        return fft_mask_rgb_filter(x, self.fft_mask)  # BxCxHxW
