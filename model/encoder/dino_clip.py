from typing import Any, Dict
from torch import nn
import torch
import open_clip
from typing import Type
from transformers import Dinov2Model


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LinearClassifier(nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = nn.Conv2d(in_channels, num_labels, (1, 1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        return self.classifier(embeddings)


class DinoClipTwoWayTransformer(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any],
    ):
        super().__init__()
        self.config = config
        self.dino = self._load_model()
        self.clip = self._load_clip()
        for param in self.dino.parameters():
            param.requires_grad = False
        for param in self.clip.parameters():
            param.requires_grad = False

    def _load_model(self):
        return Dinov2Model.from_pretrained("facebook/dinov2-giant")

        # return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

        # return CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

    def _load_clip(self):
        return open_clip.create_model("ViT-g-14", pretrained="laion2b_s34b_b88k")

    def forward(self, batch):
        # training_step defines the train loop.
        # Use Grounding to get the Bounding boxes
        # Pass the Bounding boxes to SAM to predict
        fil, nofil = batch[:, 0, ...], batch[:, 1, ...]
        # batch_size = fil.shape[0]
        # batch = torch.cat([fil, nofil], dim=0)
        # print(fil.shape), print(nofil.shape)
        outputs = self.dino(fil)[0][:, 0]
        outputsClip = self.clip.visual(nofil)
        # splitt based on batch size
        # outputs1, outputs2 = torch.split(outputs, batch_size, dim=0)
        # cls_token = torch.cat([outputs1[:, 0], outputs2[:, 0]], dim=1)

        # outputsClip1, outputsClip2 = torch.split(outputsClip, batch_size, dim=0)
        # cls_token_clip = torch.cat([outputsClip1, outputsClip2], dim=1)

        # cls_token_clip = self.clip_Projections(cls_token_clip)
        # cls_token_dino = self.dino_Projections(cls_token)

        # patch_tokens = dino_embeddings[:, 1:]
        # x = patch_tokens
        # Linare Project The CLS Token

        return torch.cat([outputs, outputsClip], dim=1)
