from typing import Any, Dict

from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from loader.image_loader import (
    CustomImageDataLoader,
    CustomTensorDataLoader,
    CustomImageDataLoaderWithTransform,
)
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from loader.sampler import StratifiedSampler
import torch.utils.data as data
from loader.augmentation import (
    TrainAugmentation,
    ValidationAugmentation,
)
from loader.transforms import TwoCropTransform, StandardTransform
import os
import pandas as pd


def my_collate(batch):
    batch = filter(lambda img: img[0] is not None, batch)
    return data.dataloader.default_collate(list(batch))


class ImageLoadingModule(LightningDataModule):
    def __init__(
        self,
        type: str,
        set: str,
        batch_size: int,
        num_workers: int,
        image_size: int,
        image_mean: any,
        image_std: any,
        train_data_path: str,
        val_data_path: str,
        test_data_path: str,
        filter=None,
        nrows=None,
        extractiontype="train",
    ):
        super().__init__()

        self.filter = filter
        self.nrows = nrows
        train_data = train_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        test_data = val_data_path
        val_data = test_data_path
        self.extractiontype = extractiontype
        self.train_data_path = f"{train_data}/{set}/train_data_{type}.csv"
        self.val_data_path = f"{val_data}/{set}/val_data_ADM.csv"
        self.test_data_path = f"{test_data}/{set}/test_data_{type}.csv"

    def setup(self, stage=None):
        train_transform = TrainAugmentation(
            self.image_size,
            self.image_mean,
            self.image_std,
            reference_images=self.train_data_to_dict(),
        )
        val_transform = ValidationAugmentation(
            self.image_size,
            self.image_mean,
            self.image_std,
            reference_images=self.train_data_to_dict(),
        )
        # if self.filter is not None:
        #     train_transform = TwoCropTransform(train_transform)
        #     val_transform = TwoCropTransform(val_transform)
        # else:
        train_transform = StandardTransform(train_transform)
        val_transform = StandardTransform(val_transform)
        test_transform = StandardTransform(
            ValidationAugmentation(
                self.image_size,
                self.image_mean,
                self.image_std,
                reference_images=self.train_data_to_dict(),
            )
        )
        self.train_data = CustomImageDataLoader.from_csv(
            self.train_data_path,
            transform=train_transform,
            nrows=self.nrows,
            filter=self.filter,
        )
        self.val_data = CustomImageDataLoader.from_csv(
            self.val_data_path,
            transform=val_transform,
            filter=self.filter,
        )
        self.test_data = CustomImageDataLoader.from_csv(
            self.test_data_path,
            transform=test_transform,
            filter=self.filter,
            nrows=self.nrows,
        )

    def train_data_to_dict(self):
        return pd.read_csv(
            self.train_data_path, delimiter=" ", names=["image", "label"]
        ).to_dict(orient="records")

        return pd.read_csv(self.train_data_path, delimiter=" ", header=None)[0].tolist()

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        if self.extractiontype == "train":
            return DataLoader(
                self.train_data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
            )
        elif self.extractiontype == "val":
            return DataLoader(
                self.val_data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
        elif self.extractiontype == "test":
            return DataLoader(
                self.test_data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )


class TensorLoadingModule(LightningDataModule):
    def __init__(
        self,
        filtersize: str,
        type: str,
        set: str,
        batch_size: int,
        num_workers: int,
        image_size: int,
        image_mean: any,
        image_std: any,
        train_data_path: str,
        val_data_path: str,
        test_data_path: str,
        filter=None,
        nrows=None,
        extractiontype="train",
    ):
        super().__init__()

        self.filtersize = filtersize
        self.filter = filter
        self.nrows = nrows
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.extractiontype = extractiontype
        self.train_data_path = os.path.join(train_data_path, set, type)
        self.val_data_path = os.path.join(val_data_path, set, type)
        self.test_data_path = os.path.join(test_data_path, set, type)

    def setup(self, stage=None):
        self.train_data = CustomTensorDataLoader(
            root_dir=self.train_data_path,
            filtersize=self.filtersize,
        )
        self.val_data = CustomTensorDataLoader(
            root_dir=self.val_data_path,
            filtersize=self.filtersize,
        )
        self.test_data = CustomTensorDataLoader(
            root_dir=self.test_data_path,
            filtersize=self.filtersize,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
