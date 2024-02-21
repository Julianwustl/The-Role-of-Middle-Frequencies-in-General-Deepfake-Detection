from torch import nn
import torch


class SingleLayerHead(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.act = nn.Sigmoid()

    def forward(self, features, **kwargs):
        return self.act(self.fc(features))


class LinareProjectionLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.projection = nn.Linear(input_size, input_size)
        self.classifcation = nn.Linear(input_size, output_size)

    def forward(self, features, **kwargs):
        features = self.projection(features)
        return self.classifcation(features)


# TODO: Rename
class NonLinearHead(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.dense = nn.Sequential(
            nn.BatchNorm1d(2 * input_size),
            nn.Linear(2 * input_size, 1 * input_size),
            nn.ReLU(inplace=True),
            nn.Linear(1 * input_size, output_size),
        )

    def forward(self, features, **kwargs):
        return self.dense(features)


# TODO: Rename
class Focushead(nn.Module):
    def __init__(self, hidden_dim: int, num_labels: int, *args, **kwargs):
        super().__init__()
        self.dense = nn.Sequential(
            nn.BatchNorm1d(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, 1 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(1 * hidden_dim, num_labels),
        )

    def forward(self, features, **kwargs):
        return self.dense(features)


class MHA_Classifier(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, num_heads=8):
        super().__init__()
        self.num_features = hidden_dim  # Number of channels in latent space
        self.cls_token = nn.Embedding(1, self.num_features)
        self.mha = nn.MultiheadAttention(
            embed_dim=self.num_features, num_heads=num_heads, batch_first=True
        )
        self.classifier = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        b = x.shape[0]

        # x shape: (b, 256, 64, 64)

        # Flatten spatial dimensions
        x = x.view(b, self.num_features, -1)  # shape: (b, 256, 4096)

        # Prepend CLS token
        cls_token = self.cls_token.weight.unsqueeze(0).expand(x.size(0), -1, -1)
        # cls_token = torch.repeat_interleave(self.cls_token.weight, b, dim=0)
        # cls_tokens = nn.Rep(1, b, 1)  # shape: (1, b, 256)
        x = x.permute(0, 2, 1)  # shape: (4096, b, 256)
        x = torch.cat((cls_token, x), dim=1)  #

        # Self-attention
        attn_output, _ = self.mha(x, x, x)  # shape: (4097, b, 256)

        # Use only the CLS token for classification
        iou_token_out = attn_output[:, 0, :]
        # x = attn_output[0]  # shape: (b, 256)

        # Classification
        x = self.classifier(iou_token_out)  # shape: (b, num_classes)

        return x
