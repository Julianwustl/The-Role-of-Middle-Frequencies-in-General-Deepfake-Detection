from functools import partial
from typing import Any, Dict, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy, recall, precision, confusion_matrix
from torchmetrics import F1Score

import torch
from torchmetrics.classification.stat_scores import StatScores
from torchmetrics.functional.classification.roc import binary_roc
from torchmetrics.functional.classification.auroc import binary_auroc


def pd_at(y_true: torch.Tensor, y_score: torch.Tensor, x: float) -> float:
    """
    Compute probability of detection (true positive rate) given a certain false positive/alarm rate.
    Corresponds to the y-coordinate on a ROC curve given a certain x-coordinate.
    """
    fpr, tpr, _ = binary_roc(y_true, y_score)
    idx = torch.argmin(torch.abs(fpr - x))  # get index where fpr is closest to x

    return tpr[idx].item()


class ImageBatchClassifier(LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        loss: nn.Module,
        transform: Optional[nn.Module] = None,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.encoder = encoder
        self.head = head
        self.loss_fn = loss
        self.f1 = F1Score(task="binary", num_classes=1)
        self.pd_10 = partial(pd_at, x=0.1)
        self.pd_05 = partial(pd_at, x=0.05)
        self.pd_01 = partial(pd_at, x=0.01)
        self.transform = transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder:
            x = self.encoder(x)
        return self.head(x)

    def add_transform(self, transform: nn.Module) -> None:
        self.transform = transform

    def is_bce(self) -> bool:
        return isinstance(self.loss_fn, nn.BCEWithLogitsLoss)

    def compute_features(self, x):
        bsz = x[0].shape[0]
        images = torch.cat([x[0], x[1]], dim=0)
        features = self(images)
        features = torch.functional.F.normalize(features, dim=1)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        return torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

    def shared_step(self, batch):
        x, y = batch
        if not self.is_bce():
            y_hat = self.compute_features(x)
            loss = self.loss_fn(y_hat, y)
        else:
            y_hat = self.forward(x)
            loss = self.loss_fn(y_hat.view(-1), y)

        if torch.isnan(loss):
            raise ValueError("Loss is NaN")

        return y_hat, y, loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat.view(-1), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.view(-1)
        loss = self.loss_fn(y_hat, y)

        acc = accuracy(y_hat, y, task="binary")
        f1_acc = self.f1(y_hat, y)
        pd_01 = self.pd_01(y_hat, y.int())
        conf = confusion_matrix(y_hat, y.int(), task="binary")
        self.log("test_acc", acc)
        self.log("test_f1", f1_acc)
        self.log("test_pd_01", pd_01)
        self.log("val_loss", loss)
        self.log("test_fn", conf[0][1])
        self.log("test_fp", conf[1][0])
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        encoder = self.encoder(x)
        y_hat = self.head(encoder)
        if self.is_bce():
            loss = self.loss_fn(y_hat.view(-1), y)
            acc = accuracy(y_hat.view(-1), y, task="binary")
            rec = recall(y_hat.view(-1), y.int(), task="binary")
            prec = precision(y_hat.view(-1), y.int(), task="binary")
            roc_auc = binary_auroc(y_hat.view(-1), y.int())
            pd_10 = self.pd_10(y_hat.view(-1), y.int())
            pd_05 = self.pd_05(y_hat.view(-1), y.int())
            pd_01 = self.pd_01(y_hat.view(-1), y.int())
            f1_acc = self.f1(y_hat.view(-1), y)

            self.log("test_acc", acc)
            self.log("test_recall", rec)
            self.log("test_precision", prec)
            self.log("test_roc_auc", roc_auc)
            self.log("test_pd_10", pd_10)
            self.log("test_pd_05", pd_05)
            self.log("test_pd_01", pd_01)
            self.log("test_f1", f1_acc)

            return {"loss": loss, "preds": y_hat, "label": y}
        return {"encoder_embedding": encoder, "embedding": y_hat, "label": y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class FeatureExtractor(LightningModule):
    def __init__(self, encoder: nn.Module, learning_rate: float = 1e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.encoder = encoder

    def forward(self, batch) -> torch.Tensor:
        x, y = batch
        return {"embedding": x, "label": y}

    def training_step(self, batch, batch_idx):
        return self(batch)

    def validation_step(self, batch, batch_idx):
        return self(batch)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
