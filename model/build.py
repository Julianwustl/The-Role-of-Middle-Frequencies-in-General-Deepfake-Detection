from typing import Any, Dict

from torch import nn
import torch
from model.loss import FocalLoss, NegativeCosineSimilarityLoss, SupConLoss
from model.image import ImageBatchClassifier, FeatureExtractor
from model.head import (
    SingleLayerHead,
    NonLinearHead,
    MHA_Classifier,
    Focushead,
    LinareProjectionLayer,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from model.encoder import ClipModels, DinoV2, OpenClip, DinoClipTwoWayTransformer
from pytorch_lightning import LightningModule
from model.loss import FocusLoss, TribletMarginLoss


image_model_encoders_map = {
    "Clip": ClipModels,
    "DinoV2": DinoV2,
    "OpenClip": OpenClip,
    "DinoClip": DinoClipTwoWayTransformer,
}

heads_map: Dict[str, nn.Module] = {
    "linear": SingleLayerHead,
    "projection": LinareProjectionLayer,
    "MLP": NonLinearHead,
    "MHA": MHA_Classifier,
    "Focushead": Focushead,
}

loss_map = {
    "BCE": BCEWithLogitsLoss(),
    "CrossEntropy": CrossEntropyLoss(),
    "Contrastive": NegativeCosineSimilarityLoss(),
    "Triblet": TribletMarginLoss(),
    "FocusLoss": FocusLoss(pos_weight=torch.tensor([4.5])),
    "FocalLoss": FocalLoss(),
    "SubConLoss": SupConLoss(),
}


def build_encoder(encoder_type: str, checkpoint: str) -> nn.Module:
    if model := image_model_encoders_map.get(encoder_type):
        return model(checkpoint)
    raise ValueError(f"Unknown image encoder type: {encoder_type}")


def build_model(
    encoder_type: str,
    checkpoint: str,
    hidden_size: str,
    head_type: str,
    loss_type: str,
    num_labels: str,
    learning_rate: float,
    transform=None,
    extract=False,
) -> LightningModule:
    heads = heads_map.get(head_type)
    loss = loss_map.get(loss_type)
    num_labels = 1 if head_type == "BCE" else num_labels

    encoder = build_encoder(encoder_type, checkpoint)
    head = heads(hidden_size, num_labels)
    if extract:
        return FeatureExtractor(encoder, learning_rate)
    return ImageBatchClassifier(encoder, head, loss, transform, learning_rate)
