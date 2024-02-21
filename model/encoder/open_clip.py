from typing import Any, Dict
from torch import nn


import open_clip


class OpenClip(nn.Module):
    def __init__(
        self,
        checkpoint: str,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.clip = self._load_clip()
        for param in  self.clip.parameters():
            param.requires_grad = False
        # Here we freeze all the weights of the encoder laion2b_s34b_b88k
    def _load_clip(self):
        return open_clip.create_model("ViT-g-14", pretrained=self.checkpoint)
    def forward(self, batch):
        # training_step defines the train loop.
        # Use Grounding to get the Bounding boxes
        # Pass the Bounding boxes to SAM to predict
        return  self.clip.visual(batch)
