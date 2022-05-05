import torch.nn as nn


def triplet_loss(anchor_emb, positive_emb, negative_emb, margin: float = 0.5, reduction: str = "mean"):
    triplet_loss_f = nn.TripletMarginLoss(margin=margin, p=2, reduction=reduction)
    return triplet_loss_f(anchor_emb, positive_emb, negative_emb)