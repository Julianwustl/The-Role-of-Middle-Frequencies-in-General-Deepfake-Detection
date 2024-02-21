import os
import random
from typing import Any, Dict, List, Optional, Tuple
from pytorchvideo.data import Ucf101, make_clip_sampler, LabeledVideoDataset, Kinetics

# from torchvision.transforms import PILToTensor, Resize
from albumentations import (
    Compose,
    GaussNoise,
    GaussianBlur,
    HorizontalFlip,
    JpegCompression,
    OneOf,
    RandomBrightnessContrast,
    HueSaturationValue,
    ToGray,
    FDA,
    ShiftScaleRotate,
    ImageCompression,
    RandomCrop,
    Normalize,
    fourier_domain_adaptation,
    Resize,
    ImageOnlyTransform,
    BasicTransform,
)
import albumentations as A
from albumentations.pytorch import ToTensorV2


from torchvision.transforms import Compose, Lambda
from core.fft import (
    fft_mask_rgb_filter,
    get_mask_fn,
    # fourier_domain_adaptation,
    channel_fourier_domain_adaptation,
)
from core.image import read_and_resize_image
from torchvision.io import read_image
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torch import nn


class RandomButterworthFilter(BasicTransform):
    def __init__(self, always_apply=False, p=0.7):
        super(RandomButterworthFilter, self).__init__(always_apply, p)

    def apply_with_params(self, tensor_image, **params):
        # Convert numpy image to torch tensor
        tensor_image = torch.from_numpy(params["image"]).permute(2, 0, 1)
        # Randomly choose between butterworth_low_pass or butterworth_high_pass
        filter_type = "butterworth_lowpass"  # random.choice(["butterworth_lowpass"])  # , "butterworth_highpass"
        # if filter_type == "butterworth_lowpass":
        D0 = random.randint(30, 150)
        # else:
        #     D0 = random.randint(0, 20)
        # Randomly choose a value between 0 and 190 for D0
        # Generate filter mask
        mask_fn = get_mask_fn(filter_type)
        mask = mask_fn(
            width=tensor_image.shape[1], height=tensor_image.shape[2], D0=D0, n=1
        )
        # Apply mask filter
        filtered_tensor_image = fft_mask_rgb_filter(tensor_image, mask)
        # Convert torch tensor back to numpy image
        return {
            "image": filtered_tensor_image.permute(1, 2, 0).numpy(),
            "label": params["label"],
        }


class FDATransform(BasicTransform):
    def __init__(
        self,
        reference_images: List[Dict[str, Any]],
        beta_limit=0.1,
        read_fn=read_image,
        always_apply=False,
        p=0.5,
    ):
        super(FDATransform, self).__init__(always_apply, p)
        self.reference_images = reference_images
        self.beta = beta_limit
        # self.add_targets({"label": "label"})

    def apply_with_params(self, dummy, **params):
        # Convert numpy image to torch tensor
        image_dict = random.choice(self.reference_images)
        beta = random.uniform(0.01, 0.1)
        target_img = read_and_resize_image(image_dict["image"])
        # target_img = torch.from_numpy(target_img).permute(2, 0, 1)
        # source_image = source_image.permute(2, 0, 1)
        target_label = image_dict["label"]
        # Randomly choose between butterworth_low_pass or butterworth_high_pass#

        fda_transform_image = fourier_domain_adaptation(
            params["image"], target_img, beta
        )
        if target_label == 1.0:
            params["label"] = torch.tensor(1.0, dtype=torch.float32)

        return {
            "image": fda_transform_image,
            "label": params["label"],
        }


class TrainAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, image_size, image_mean, image_std, **params) -> None:
        super().__init__()

        self.transform = A.Compose(
            [
                Resize(
                    image_size, image_size
                ),  # Use interpolation=1 for bilinear (default in torchvision)
                # ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
                # GaussNoise(p=0.5),
                # GaussianBlur(blur_limit=3, sigma_limit=4, p=1),
                HorizontalFlip(),
                # IsotropicResize not in standard albumentations; you might need a custom function or library
                # PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                # OneOf([
                #     RandomBrightnessContrast(),
                #     # FancyPCA(),  # FancyPCA is not a standard albumentations transform. You might need a custom function or library.
                #     HueSaturationValue()
                # ], p=0.7),
                # ToGray(p=0.2),
                # JpegCompression(quality_lower=50, quality_upper=100, p=0.5),
                # RandomCrop(image_size, image_size, p=0.5),
                # ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.5),  # Use border_mode=0 for BORDER_CONSTANT
                # RandomButterworthFilter(p=1.0),
                # FDA(reference_images=params["reference_images"], p=1.0),
                # FDATransform(reference_images=params["reference_images"], p=0.5),
                Normalize(mean=image_mean, std=image_std),
                ToTensorV2(p=1.0),  # Use albumentations' ToTensorV2
            ],
            additional_targets={"label": "label"},
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor, **params) -> torch.Tensor:
        label = params["label"]
        return self.transform(image=x, label=label)  # BxCxHxW


class ValidationAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, image_size, image_mean, image_std, **params) -> None:
        super().__init__()
        self.image_mean = image_mean
        self.image_std = image_std
        self.transform = A.Compose(
            [
                Resize(image_size, image_size),
                Normalize(mean=image_mean, std=image_std),
                ToTensorV2(p=1.0),  # Use albumentations' ToTensorV2
            ],
            additional_targets={"label": "label"},
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor, **params) -> torch.Tensor:
        label = params["label"]
        return self.transform(image=x, label=label)  # BxCxHxW


class GreyAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, image_size) -> None:
        super().__init__()

        self.transform = A.Compose(
            [
                Resize(image_size, image_size),
                ToTensorV2(p=1.0),  # Use albumentations' ToTensorV2
            ]
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(image=x)  # BxCxHxW
