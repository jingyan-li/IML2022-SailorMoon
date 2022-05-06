"""
Siamese Neural Network
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class EmbeddingNet(nn.Module):
    def __init__(self, img_size: tuple, out_channels: int):
        super(EmbeddingNet, self).__init__()
        # Input is [BatchSize, 3, 348, 512]
        # Input is [BatchSize, 3, 240, 240] after image resize
        # TODO: Replace CNN by pre-trained network & fine tuning
        # TODO: CNN structure
        self.convnet = nn.Sequential(nn.Conv2d(3, 8, 5, padding=2),
                                     nn.PReLU(), nn.BatchNorm2d(8),
                                     nn.MaxPool2d(2, stride=2),     # [B, 8, 174, 256] # 120
                                     # ==========================================================
                                     nn.Conv2d(8, 16, 5, padding=2),
                                     nn.PReLU(), nn.BatchNorm2d(16),
                                     nn.MaxPool2d(2, stride=2))     # [B, 16, 87, 128] # 60

        self.fc = nn.Sequential(nn.Linear(16 * (img_size[0]//4) * (img_size[1]//4), 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, out_channels)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletModel(nn.Module):
    def __init__(self, embedding_model: nn.Module, margin: float):
        super(TripletModel, self).__init__()
        self.embedding_model = embedding_model
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)
        self.triplet_loss_batch = nn.TripletMarginLoss(margin=self.margin, p=2, reduction='none')

    def forward(self, anchor, positive, negative):
        anchor_emb = self.get_embedding(anchor)
        positive_emb = self.get_embedding(positive)
        negative_emb = self.get_embedding(negative)
        return anchor_emb, positive_emb, negative_emb

    def get_embedding(self, x):
        # TODO: Embedding by pre-trained network
        emb_weight = self.embedding_model.weight.clone()
        embedding = nn.Embedding.from_pretrained(emb_weight)
        return embedding(x)

    def loss(self, anchor, positive, negative):
        """
        Calculate loss with batch-wise mean reduction
        :param anchor:
        :param positive:
        :param negative:
        :return:
        """
        loss = self.triplet_loss(anchor, positive, negative)
        return loss

    def predict(self, anchor, positive, negative):
        """
        Make prediction on minibatch
        :param anchor: anchor img of size [B, C, B, W]
        :param positive: positive img
        :param negative: negative img
        :return: pred_y of size [B,] and loss of size [B,]
        """
        # print(f"Triplet model predict: anchor shape {anchor.shape}")
        anchor = self.get_embedding(anchor)
        positive = self.get_embedding(anchor, positive)
        negative = self.get_embedding(anchor, negative)
        # print(f"Triplet model predict: anchor shape {anchor.shape}")
        loss = self.triplet_loss_batch(anchor, positive, negative)
        # print(f"Triplet model predict: loss shape {loss.shape}")
        pred_y = torch.where(loss - self.margin < 0, 1, 0)
        # print(f"Triplet model predict: pred_y shape {pred_y.shape}")
        return pred_y, loss