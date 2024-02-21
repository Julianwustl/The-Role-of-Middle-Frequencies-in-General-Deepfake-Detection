import numpy as np
from PIL import Image
import torch
from torch import nn
from PIL import Image as im
from image import (
    load_and_preprocess_image,
    save_torch_to_grey,
    save_torch_to_rgb,
    load_and_preprocess_grey_image,
    save_np_to_rgb,
)
import glob
from torchvision import transforms
from storage import ensure_dir, create_path
import argparse
import os
from functools import partial
import albumentations as A


def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)


def create_2dft(tensor: torch.Tensor):
    # Compute 2D Fourier Transform
    f_transform = torch.fft.fftn(tensor)
    # Compute the magnitude squared (power spectrum)
    # Shift the zero frequency component to the center
    power_spec = torch.fft.fftshift(power_spec)
    # return torch.log(power_spec + 1)
    power_spec = torch.abs(f_transform) ** 2
    return power_spec


def channel_fourier_domain_adaptation(
    img: torch.Tensor, target_img: torch.Tensor, beta: float
):
    img[..., 0, :, :] = filter_mask(img[..., 0, :, :], target_img, beta)
    img[..., 1, :, :] = filter_mask(img[..., 1, :, :], target_img, beta)
    img[..., 2, :, :] = filter_mask(img[..., 2, :, :], target_img, beta)
    return img


def fourier_domain_adaptation(
    img: torch.Tensor, target_img: torch.Tensor, beta: float
) -> torch.Tensor:
    """
    Fourier Domain Adaptation from https://github.com/YanchaoYang/FDA adapted for PyTorch Tensors

    Args:
        img:  source image tensor
        target_img:  target image tensor for domain adaptation
        beta: coefficient from source paper

    Returns:
        transformed image tensor

    """
    # get fft of both source and target
    fft_src = torch.fft.fft2(img.to(torch.complex64))
    fft_trg = torch.fft.fft2(target_img.to(torch.complex64))

    # extract amplitude and phase of both fft-s
    amplitude_src, phase_src = torch.abs(fft_src), torch.angle(fft_src)
    amplitude_trg = torch.abs(fft_trg)

    # mutate the amplitude part of source with target
    amplitude_src = torch.fft.fftshift(amplitude_src)
    amplitude_trg = torch.fft.fftshift(amplitude_trg)
    height, width = amplitude_src.shape[-2:]
    border = int(min(height, width) * beta)  # fix applied here
    center_y, center_x = height // 2, width // 2

    y1, y2 = center_y - border, center_y + border + 1
    x1, x2 = center_x - border, center_x + border + 1

    amplitude_src[..., y1:y2, x1:x2] = amplitude_trg[..., y1:y2, x1:x2]
    amplitude_src = torch.fft.ifftshift(amplitude_src)

    # get mutated image
    src_image_transformed = torch.fft.ifft2(amplitude_src * torch.exp(1j * phase_src))
    src_image_transformed = torch.abs(src_image_transformed).to(torch.uint8)

    return src_image_transformed


def hard_bandpass_filter_mask(width, height, D0, n=1, band_size=5):
    u = torch.arange(width) - width // 2
    v = torch.arange(height) - height // 2

    U, V = torch.meshgrid(u, v)

    # Calculate the distance from the origin
    D = torch.sqrt(U**2 + V**2)

    # Creating a hard bandpass filter
    # Values within the band [D0 - band_size/2, D0 + band_size/2] are set to 1
    # Others are set to 0
    lower_bound = D0 - band_size / 2
    upper_bound = D0 + band_size / 2

    mask = ((D >= lower_bound) & (D <= upper_bound)).float()

    return mask


def fourier_band_injection(
    source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
):
    """This function takes a source image, a target image and mask.

    We use the mask which should be a bandpass filter mask to extract the target domain (fake, real)
    from the target image. Then we replace the same band from the source image with the target domain.

    """
    # Apply Fourier Transform
    t_transform = torch.fft.fft2(target)
    s_transform = torch.fft.fft2(source)
    t_transform_shifted = torch.fft.fftshift(t_transform)
    s_transform_shifted = torch.fft.fftshift(s_transform)
    # Apply the mask and blend with source
    combined_transform_shifted = (
        s_transform_shifted * (1 - mask) + t_transform_shifted * mask
    )  #
    # Inverse Fourier Transform
    combined_transform = torch.fft.ifftshift(combined_transform_shifted)
    combined_image = torch.fft.ifft2(combined_transform)
    # Convert to absolute values and adjust data type
    combined_image_abs = torch.abs(combined_image)
    # Convert to an appropriate format, e.g., uint8 for 8-bit images
    combined_image_uint8 = combined_image_abs.to(torch.uint8)
    return combined_image_uint8


# def fourier_band_injection(
#     target: torch.Tensor, source: torch.Tensor, mask: torch.Tensor
# ):
#     """This function takes a source image, a target image and mask.

#     We use the mask which should be a bandpass filter mask to extract the target domain (fake, real)
#     from the target image. Then we replace the same band from the source image with the target domain.

#     """
#     # Apply Fourier Transform
#     t_transform = torch.fft.fft2(target)
#     s_transform = torch.fft.fft2(source)
#     t_transform_shifted = torch.fft.fftshift(t_transform)
#     s_transform_shifted = torch.fft.fftshift(s_transform)
#     # Apply the mask and blend with source
#     s_transform_shifted[..., y1:y2, x1:x2] = t_transform_shifted[..., y1:y2, x1:x2]

#     # Inverse Fourier Transform
#     combined_transform = torch.fft.ifftshift(s_transform_shifted)
#     combined_image = torch.fft.ifft2(combined_transform)
#     # Convert to absolute values and adjust data type
#     combined_image_abs = torch.abs(combined_image)
#     # Convert to an appropriate format, e.g., uint8 for 8-bit images
#     combined_image_uint8 = combined_image_abs.to(torch.uint8)
#     return combined_image_uint8


def hardpass_bandpass_filter_mask(rows, cols, low_radius, high_radius):
    center_x, center_y = rows // 2, cols // 2
    y = torch.arange(0, rows).view(-1, 1).repeat(1, cols)
    x = torch.arange(0, cols).view(1, -1).repeat(rows, 1)
    distance_from_center = torch.sqrt(
        (x - center_x) ** 2 + (y - center_y) ** 2
    ) / np.sqrt(center_x**2 + center_y**2)
    mask = torch.zeros((rows, cols), dtype=torch.uint8)
    mask[
        (low_radius <= distance_from_center) & (distance_from_center < high_radius)
    ] = 1
    return (low_radius, mask)


def butterworth_highpass_filter_mask(width, height, D0, n):
    # Generate frequency coordinates in the 2D plane
    u = torch.arange(width) - width // 2
    v = torch.arange(height) - height // 2
    U, V = torch.meshgrid(u, v, indexing="xy")  # add the indexing argument
    # Calculate the distance from the origin
    D = torch.sqrt(U**2 + V**2)
    # Calculate high-pass and low-pass components
    high_pass = 1 / (1 + (D0 / D) ** (2 * n))
    return high_pass


def butterworth_lowpass_filter_mask(width, height, D0, n):
    """
    Generate a 2D Butterworth low-pass filter mask in PyTorch.
    """
    # Generate frequency coordinates in the 2D plane
    u = torch.arange(width) - width // 2
    v = torch.arange(height) - height // 2
    U, V = torch.meshgrid(u, v, indexing="xy")  # add the indexing argument

    # Calculate the distance from the origin
    D = torch.sqrt(U**2 + V**2)

    return 1 / (1 + (D / D0) ** (2 * n))


def butterworth_bandpass_filter_mask(width, height, D0, n, band_size=20):
    u = torch.arange(width) - width // 2
    v = torch.arange(height) - height // 2

    U, V = torch.meshgrid(u, v)

    # Calculate the distance from the origin
    D = torch.sqrt(U**2 + V**2)
    return 1 - 1 / (1 + (D * band_size / ((D**2) - D0**2)) ** (2 * n))


def fft_mask_rgb_filter(tensor: torch.Tensor, mask: torch.Tensor):
    tensor[..., 0, :, :] = filter_mask(tensor[..., 0, :, :], mask)
    tensor[..., 1, :, :] = filter_mask(tensor[..., 1, :, :], mask)
    tensor[..., 2, :, :] = filter_mask(tensor[..., 2, :, :], mask)
    return tensor


def fft_transform(tensor: torch.Tensor):
    tensor[..., 0, :, :] = create_2dft(tensor[..., 0, :, :])
    tensor[..., 1, :, :] = create_2dft(tensor[..., 1, :, :])
    tensor[..., 2, :, :] = create_2dft(tensor[..., 2, :, :])
    return tensor


def fft_mask_hsv_filter(tensor: torch.Tensor, mask: torch.Tensor):
    tensor[..., -1, :, :] = filter_mask(tensor[..., -1, :, :], mask)
    return tensor


def filter_mask(tensor: torch.Tensor, mask: torch.Tensor):
    # Step 1: Convert the tensor to NumPy array
    # image_np = tensor.cpu().numpy()
    image = tensor
    # Step 2: Apply Fourier Transform
    f_transform = torch.fft.fft2(image)
    f_transform_shifted = torch.fft.fftshift(f_transform)
    # Step 4: Apply the mask through convultion
    f_transform_shifted_masked = f_transform_shifted * mask
    # Step 5: Inverse Fourier Transform
    f_transform_inv_shifted = torch.fft.ifftshift(f_transform_shifted_masked)
    f_transform_inv = torch.fft.ifft2(f_transform_inv_shifted)
    f_transform_inv = torch.abs(f_transform_inv).to(torch.uint8)

    # print(torch.max(f_transform_inv))
    # print(torch.min(f_transform_inv))
    return f_transform_inv


def get_mask_fn(filter_type):
    if filter_type == "butterworth_lowpass":
        return butterworth_lowpass_filter_mask
    elif filter_type == "butterworth_highpass":
        return butterworth_highpass_filter_mask
    elif filter_type == "butterworth_bandpass":
        return butterworth_bandpass_filter_mask

    else:
        raise ValueError(f"Invalid filter type: {filter_type}")


def main(args):
    path = create_path(args.save_path, args.experiment_name)
    ensure_dir(path)
    mask_fn = None
    if args.filter_type == "butterworth_lowpass":
        mask_fn = butterworth_lowpass_filter_mask
    elif args.filter_type == "butterworth_highpass":
        mask_fn = butterworth_highpass_filter_mask
    elif args.filter_type == "butterworth_bandpass":
        mask_fn = butterworth_bandpass_filter_mask
    elif args.filter_type == "FDA":
        image_source = load_and_preprocess_image(args.image_path)
        image_target = load_and_preprocess_image(args.target_image_path)
        fda_image = fourier_domain_adaptation(image_source, image_target, 0.01)
        save_torch_to_rgb(
            fda_image, os.path.join(path, f"{args.filter_type}_image.png")
        )
        return
    else:
        raise ValueError(f"Invalid filter type: {args.filter_type}")
    image_target = load_and_preprocess_image(args.target_image_path)
    for i in range(0, 190, args.filter_radius):
        if i == 0:
            i = 1
        image = load_and_preprocess_image(args.image_path)
        mask = mask_fn(
            width=image.shape[1],
            height=image.shape[2],
            D0=i,
            n=1,  # , band_size=20, D0=i, n=4
        )
        # filtered_image = fourier_domain_adaptation_2(
        #     image,
        #     image_target,
        #     D0=i,
        #     band_size=20,
        # )

        filtered_image = fourier_band_injection(image, image_target, mask)

        save_torch_to_rgb(
            filtered_image, os.path.join(path, f"{args.filter_type}_{i}_image.png")
        )
        save_torch_to_grey(mask, os.path.join(path, f"{args.filter_type}_{i}mask.png"))


def create_lowpass(image_path:str,):
    image_target = load_and_preprocess_image(image_path)
    for i in range(0, 190, 25):
        if i == 0:
            i = 1
        image = load_and_preprocess_image(image_path)
        mask = butterworth_lowpass_filter_mask(
            width=image.shape[1],
            height=image.shape[2],
            D0=i,
            n=1,  # , band_size=20, D0=i, n=4
        )
        # filtered_image = fourier_domain_adaptation_2(
        #     image,
        #     image_target,
        #     D0=i,
        #     band_size=20,
        # )

        return fft_mask_rgb_filter(image, image_target, mask)

        

def main_2():
    """This Function runs all models against all datasets."""
    rows = 5
    path = "/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/results/data_paper/filtered"
    data_types = [
        "ADM",
        "DDPM",
        "Diff-ProjectedGAN",
        "Diff-StyleGAN2",
        "IDDPM",
        "LDM",
        "PNDM",
        "ProGAN",
        "ProjectedGAN",
        "StyleGAN",
        "Real"
    ]
    for data_type in data_types:
        image_directory = f"/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/data/lsun_bedroom/test/{data_type}/"
        image_paths = []
        for filename in os.listdir(image_directory)[:1]:
            image_path  = image_directory+filename
        
            image_target = load_and_preprocess_image(image_path)
            for i in range(0, 190, 25):
                if i == 0:
                    i = 5
                image = load_and_preprocess_image(image_path)
                mask = butterworth_lowpass_filter_mask(
                    width=image.shape[1],
                    height=image.shape[2],
                    D0=i,
                    n=1,  # , band_size=20, D0=i, n=4
                )
                # filtered_image = fourier_domain_adaptation_2(
                #     image,
                #     image_target,
                #     D0=i,
                #     band_size=20,
                # )
                if i not  in [5,25,100,175]:
                    continue
                filtered_image = fft_mask_rgb_filter(image, mask)
                # load the data. 
                save_torch_to_rgb(
                    filtered_image, os.path.join(path, f"{data_type}_{i}_{filename}")
                )
                #save_torch_to_grey(mask, os.path.join(path, f"{args.filter_type}_{i}mask.png"))

if __name__ == "__main__":
    """parser = argparse.ArgumentParser(description="FFT Image Processor")
    parser.add_argument(
        "--image_path", "-p", type=str, required=True, help="Path to the image"
    )
    parser.add_argument(
        "--target_image_path",
        "-tip",
        type=str,
        required=False,
        help="Target image Path",
    )
    parser.add_argument(
        "--experiment_name", "-e", type=str, required=True, help="Experiment name"
    )
    parser.add_argument(
        "--save_path", "-sp", type=str, required=True, help="Path to save results"
    )
    parser.add_argument(
        "--filter_type",
        "-t",
        type=str,
        required=True,
        choices=[
            "butterworth_lowpass",
            "butterworth_highpass",
            "butterworth_bandpass",
            "FDA",
        ],
        help="Type of filter to use",
    )
    parser.add_argument(
        "--filter_radius",
        "-r",
        type=int,
        required=True,
        help="Radius of the filter",
        default=5,
    )
    args = parser.parse_args()"""
    main_2()
    #main(args)
