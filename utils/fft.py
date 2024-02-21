
import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image


def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft =np.fft.fft2(ft)
    return np.fft.fftshift(ft)
    

def create_2dft(tensor):
    f_transform = np.fft.fft2(tensor)
    f_transform_shifted_masked = np.fft.fftshift(f_transform)
    return 
    f_transform_inv_shifted = np.fft.ifftshift(f_transform_shifted_masked)
    f_transform_inv = np.fft.ifft2(f_transform_inv_shifted)
    

def filter_frequency(tensor, low_radius, high_radius):
    """
    Filters the frequencies of an image represented as a PyTorch tensor.
    Parameters:
        tensor (torch.Tensor): The image tensor (2D).
        low_radius (float): The lower bound of the frequency range to remove.
        high_radius (float): The upper bound of the frequency range to remove.
    Returns:
        torch.Tensor: The filtered image tensor (2D).
    """

    # Step 1: Convert the tensor to NumPy array
    #image_np = tensor.cpu().numpy()
    image_np = tensor
    # Step 2: Apply Fourier Transform
    f_transform = np.fft.fft2(image_np)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Step 3: Design the mask
    rows, cols = image_np.shape
    center_x, center_y = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
 
    # The mask can be calculated before hand, so we do not need this, and can just pass the band filter mask directly. 
    for x in range(rows):
        for y in range(cols):
            # Get the relative distance from the center based on the l1 norm
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2) / np.sqrt(center_x**2 + center_y**2)
            if low_radius <= distance < high_radius:
                mask[x, y] = 0
            
            
    #mask = 1-mask            
    # Step 4: Apply the mask
    f_transform_shifted_masked = f_transform_shifted * mask
    # Step 5: Inverse Fourier Transform
    f_transform_inv_shifted = np.fft.ifftshift(f_transform_shifted_masked)
    f_transform_inv = np.fft.ifft2(f_transform_inv_shifted)
    f_transform_inv = np.abs(f_transform_inv).astype(np.uint8)
    # Step 6: Convert back to PyTorch tensor
    #filtered_tensor = torch.from_numpy(f_transform_inv).float()
    return f_transform_inv

# def filter_frequency(tensor, low_radius, high_radius):
#     """
#     Filters the frequencies of an image represented as a PyTorch tensor.
#     Parameters:
#         tensor (torch.Tensor): The image tensor (2D).
#         low_radius (float): The lower bound of the frequency range to remove.
#         high_radius (float): The upper bound of the frequency range to remove.
#     Returns:
#         np.ndarray: The filtered image array (2D).
#     """
#     # Step 1: Convert the tensor to NumPy array (if needed)
#     image_np = tensor if isinstance(tensor, np.ndarray) else tensor.cpu().numpy()

#     # Step 2: Apply Fourier Transform
#     f_transform = np.fft.fft2(image_np)
#     f_transform_shifted = np.fft.fftshift(f_transform)

#     # Step 3: Design the mask
#     rows, cols = image_np.shape
#     center_x, center_y = rows // 2, cols // 2
    
#     y, x = np.ogrid[:rows, :cols]
#     distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2) / np.sqrt(center_x**2 + center_y**2)
    
#     mask = np.ones((rows, cols), dtype=np.uint8)
#     mask[(low_radius <= distance_from_center) & (distance_from_center < high_radius)] = 0

#     # Step 4: Apply the mask
#     f_transform_shifted_masked = f_transform_shifted * mask

#     # Step 5: Inverse Fourier Transform
#     f_transform_inv_shifted = np.fft.ifftshift(f_transform_shifted_masked)
#     f_transform_inv = np.fft.ifft2(f_transform_inv_shifted)
    
#     f_transform_inv = np.abs(f_transform_inv).astype(np.uint8)
    
#     return f_transform_inv

def fft_and_back(tensor, low_radius=0.9, high_radius=1.0):
    """
    Filters the frequencies of an image represented as a PyTorch tensor.
    Parameters:
        tensor (torch.Tensor): The image tensor (2D).
        low_radius (float): The lower bound of the frequency range to remove.
        high_radius (float): The upper bound of the frequency range to remove.
    Returns:
        torch.Tensor: The filtered image tensor (2D).
    """

    # Step 1: Convert the tensor to NumPy array
    #image_np = tensor.cpu().numpy()
    image_np = tensor
    # Step 2: Apply Fourier Transform
    f_transform = np.fft.fft2(image_np)
    f_transform_shifted = np.fft.fftshift(f_transform)
    f_transform_shifted = f_transform_shifted
    rows, cols = image_np.shape
    center_x, center_y = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
 
    for x in range(rows):
        for y in range(cols):
            # Get the relative distance from the center based on the l1 norm
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2) / np.sqrt(center_x**2 + center_y**2)
            if low_radius <= distance < high_radius:
                mask[x, y] = 0
            
    # Step 4: Apply the mask
    f_transform_shifted_masked = f_transform_shifted * mask
    # Step 5: Inverse Fourier Transform
    f_transform_inv_shifted = np.fft.ifftshift(f_transform_shifted_masked)
    f_transform_inv = np.fft.ifft2(f_transform_inv_shifted)
    f_transform_inv = np.real(f_transform_inv)
    # Step 6: Convert back to PyTorch tensor
    #filtered_tensor = torch.from_numpy(f_transform_inv).float()
    return f_transform_inv, np.abs(f_transform_shifted_masked)



if __name__ == "__main__":
    image = np.array(Image.open("/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/000_frame0.png").convert("RGB"))
    # for v in range(0,1):
    #     image[:,:,v] , mask = fft_and_back(image[:,:,v])
    low = 0.8
    high = 1
    image[:,:,0] = calculate_2dft(image[:,:,0])#,low, high)
    #image[:,:,1] = create_2dft(image[:,:,1])#,low, high)
    #image[:,:,2] = create_2dft(image[:,:,2])#,low, high)
    im = Image.fromarray(image)
    #im = im.convert("RGB")
    im.save("removed_fourir.png")
    #save the image
    