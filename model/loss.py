from typing import Any, Dict, Optional, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
import torch
from torch.nn.functional import one_hot
from torch.functional import F
from pytorch_metric_learning import miners, losses, distances
from torchvision.ops import sigmoid_focal_loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1, *args, **kwargs):
        super().__init__()
        self.temperature = temperature

    def forward(self, feats, label, mode="train"):
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> based on the label
        pos_mask = label == 1
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        return nll.mean()


class NegativeCosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(NegativeCosineSimilarityLoss, self).__init__()

    def forward(self, x1, x2):
        # Compute cosine similarity
        cosine_similarity = F.cosine_similarity(x1, x2, dim=-1)

        # Return negative cosine similarity
        return 1 - cosine_similarity.mean()


class FocusLoss(nn.Module):
    def __init__(self, weight=0.8, pos_weight=torch.tensor([2.0])):
        super(FocusLoss, self).__init__()
        self.weight = weight
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, vector1, vector2):
        return 1 - self.cosine_similarity(vector1, vector2).mean()


class TribletMarginLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.miner = miners.MultiSimilarityMiner(distance=distances.CosineSimilarity())
        self.loss_func = losses.TripletMarginLoss(
            margin=0.2, distance=distances.CosineSimilarity()
        )

    def forward(
        self, values: torch.Tensor, labels: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        indices_tuple = self.miner(values, labels)
        loss = self.loss_func(values, labels, indices_tuple)
        return loss


class FocalLoss(nn.Module):
    def __init__(
        self,
    ):
        super(FocalLoss, self).__init__()

    def forward(self, feats, label, mode="train"):
        return sigmoid_focal_loss(feats, label, reduction="mean")


# class SupConLoss(nn.Module):
#     def __init__(self, temperature=0.1):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature


#     def forward(self, feature_vectors, labels):
#         # Normalize feature vectors
#         feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
#         # Compute logits
#         logits = torch.div(
#             torch.matmul(
#                 feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
#             ),
#             self.temperature,
#         )
#         return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels))
def device_as(t1, t2):
    """
    Moves t1 to the device of t2
    """
    return t1.to(t2.device)


# class SupConLoss(nn.Module):
#    """
#    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
#    """
#    def __init__(self, batch_size=128, temperature=0.5):
#        super().__init__()
#        self.batch_size = batch_size
#        self.temperature = temperature
#        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

#    def calc_similarity_batch(self, a, b):
#        representations = torch.cat([a, b], dim=0)
#        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

#    def forward(self, proj_1, proj_2):
#        """
#        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
#        where corresponding indices are pairs
#        z_i, z_j in the SimCLR paper
#        """
#        batch_size = proj_1.shape[0]
#        z_i = F.normalize(proj_1, p=2, dim=1)
#        z_j = F.normalize(proj_2, p=2, dim=1)

#        similarity_matrix = self.calc_similarity_batch(z_i, z_j)
#        sim_ij = torch.diag(similarity_matrix, batch_size)
#        sim_ji = torch.diag(similarity_matrix, -batch_size)

#        positives = torch.cat([sim_ij, sim_ji], dim=0)

#        nominator = torch.exp(positives / self.temperature)

#        denominator =device_as(self.mask,similarity_matrix)  * torch.exp(similarity_matrix / self.temperature)

#        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
#        loss = torch.sum(all_losses) / (2 * self.batch_size)
#        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)

        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask

        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
