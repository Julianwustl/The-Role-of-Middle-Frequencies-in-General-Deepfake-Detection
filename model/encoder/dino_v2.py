from typing import Any, Dict
from torch import nn
from transformers import AutoImageProcessor, AutoModel
import torch
import open_clip
from transformers import Dinov2Model, Dinov2PreTrainedModel


class DinoV2(nn.Module):
    def __init__(
        self,
        checkpoint: str,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.dino = self._load_model()
        for param in self.dino.parameters():
            param.requires_grad = False

    def _load_model(self):
        return Dinov2Model.from_pretrained(self.checkpoint)

    def forward(self, batch):
        # fil, nofil = batch[:, 0, ...], batch[:, 1, ...]
        # batch_size = fil.shape[0]
        # batch = torch.cat([fil, nofil], dim=0)
        # print(fil.shape), print(nofil.shape)
        outputs = self.dino(batch)[0]
        cls_token = outputs[:, 0]
        patch_tokens = outputs[:, 1:].mean(dim=1)

        return cls_token
        # splitt based on batch size
        # outputs1, outputs2 = torch.split(outputs, batch_size, dim=0)
        # cls_token = torch.cat([outputs1[:, 0], outputs2[:, 0]], dim=1)
        # print(cls_token.shape)
        # patch_tokens = outputs[:, 1:]
        # return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
