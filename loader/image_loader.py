from typing import Any, Dict, Tuple
from pytorchvideo.data import LabeledVideoDataset
from torchvision.transforms.functional import resize
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os
from core.image import save_torch_to_rgb, save_normalized_image


class CustomImageDataLoader(Dataset):
    def __init__(self, path: str, transform=None, filter=None, nrows=-1):
        self.data = pd.read_csv(path, delimiter=" ", header=None).sample(frac=1)[:nrows]
        self.transform = transform
        self.filter = filter
        self.file_paths = self.data[0].tolist()
        self.newlabels = self.data[1].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        file_path = self.file_paths[index]
        label = self.newlabels[index]
        image = np.array(
            Image.open(file_path).convert("RGB")
        )  # assuming the image path is in the first column

        label = torch.tensor(
            label, dtype=torch.float
        )  # assuming the label is in the second column

        if self.filter:
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)
            image = resize(image, (224, 224), antialias=True)
            image_dict = self.filter(image, label=label)
            image = image_dict["image"]
            # if label != image_dict["label"]:
            #     print(f"Label changed from {label} to {image_dict['label']}")
            label = image_dict["label"]
            image = image.permute(1, 2, 0)
            image = image.numpy()
        transformed_image = self.transform(image, label=label)
        if transformed_image.get("label") is not None:
            label = transformed_image["label"]

        # save_normalized_image(
        #     transformed_image["image"],
        #     f"/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/results/example_data/test/test_{index}.png",
        # )
        return transformed_image["image"], label

    @classmethod
    def from_csv(
        cls, path: str, transform=None, filter=None, nrows=None
    ) -> LabeledVideoDataset:
        return cls(path, transform, filter, nrows)

    @property
    def labels(self) -> torch.Tensor:
        return torch.tensor(self.data[1].tolist())

    @property
    def ratio(self) -> float:
        sample_tensor = torch.tensor(self.data[1].tolist())
        num_zeros = torch.sum(sample_tensor == 0).item()
        num_ones = torch.sum(sample_tensor == 1).item()

        return num_zeros / num_ones


class CustomImageDataLoaderWithTransform(Dataset):
    def __init__(self, path: str, transform=None, filter=None, nrows=None):
        self.data = pd.read_csv(path, delimiter=" ", header=None, nrows=nrows)
        self.transform = transform
        self.filter = filter
        self.file_paths = self.data[0].tolist()
        self.newlabels = self.data[1].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        file_path = self.file_paths[index]
        label = self.newlabels[index]
        image = np.array(
            Image.open(file_path).convert("RGB")
        )  # assuming the image path is in the first column

        label = torch.tensor(
            label, dtype=torch.float
        )  # assuming the label is in the second column

        if self.filter:
            imageF = torch.from_numpy(image)
            imageF = imageF.permute(2, 0, 1)
            imageF = resize(imageF, (224, 224), antialias=True)
            image_dict = self.filter(imageF, label=label)
            imageF = image_dict["image"]
            # if label != image_dict["label"]:
            #     print(f"Label changed from {label} to {image_dict['label']}")
            label = image_dict["label"]
            imageF = imageF.permute(1, 2, 0)
            imageF = imageF.numpy()
        transformed_image = self.transform(image, label=label)
        transformed_imageF = self.transform(imageF, label=label)
        if transformed_image.get("label") is not None:
            label = transformed_image["label"]

        # save_normalized_image(
        #     transformed_image["image"],
        #     f"/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/results/example_data/test/test_{index}.png",
        # )

        return (
            torch.cat(
                [
                    transformed_image["image"].unsqueeze(dim=0),
                    transformed_imageF["image"].unsqueeze(dim=0),
                ],
                dim=0,
            ),
            label,
        )

    @classmethod
    def from_csv(
        cls, path: str, transform=None, filter=None, nrows=None
    ) -> LabeledVideoDataset:
        return cls(path, transform, filter, nrows)

    @property
    def labels(self) -> torch.Tensor:
        return torch.tensor(self.data[1].tolist())

    @property
    def ratio(self) -> float:
        sample_tensor = torch.tensor(self.data[1].tolist())
        num_zeros = torch.sum(sample_tensor == 0).item()
        num_ones = torch.sum(sample_tensor == 1).item()

        return num_zeros / num_ones


class CustomTensorDataLoader(Dataset):
    def __init__(self, root_dir: str, filtersize: str, transform=None, filter=None):
        self.tensor_paths = []
        self.labels = []

        for label in ["0", "1"]:
            dir_path = os.path.join(root_dir, label)

            if os.path.exists(dir_path):
                for filename in os.listdir(dir_path):
                    # Check if the filename contains the desired filtersize
                    if filename.endswith(".pt") and f"_{filtersize}_" in filename:
                        self.tensor_paths.append(os.path.join(dir_path, filename))
                        self.labels.append(int(label))

        self.transform = transform
        self.filter = filter

    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        tensor_path = self.tensor_paths[index]
        tensor = torch.load(tensor_path)

        label = torch.tensor(self.labels[index], dtype=torch.float)

        if self.transform:
            tensor = self.transform(tensor)
        if self.filter:
            tensor = self.filter(tensor)
        return tensor, label
