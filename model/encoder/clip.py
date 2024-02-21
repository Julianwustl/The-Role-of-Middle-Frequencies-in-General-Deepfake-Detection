from typing import Any, Dict
from torch import nn


from transformers import CLIPModel, CLIPVisionModel


class ClipModels(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any],
    ):
        super().__init__()
        self.config = config
        self.clip = self._load_clip() 
        for param in  self.clip.parameters():
            param.requires_grad = False
        
        # Here we freeze all the weights of the encoder
    def _load_clip(self):
        #"openai/clip-vit-large-patch14"
        # 
        return CLIPModel.from_pretrained(self.config["image"]["checkpoint"])
    def forward(self, batch):
        # training_step defines the train loop.
        # Use Grounding to get the Bounding boxes
        # Pass the Bounding boxes to SAM to predict
        image_embeddings = self.clip.get_image_features(batch)
        



        return image_embeddings