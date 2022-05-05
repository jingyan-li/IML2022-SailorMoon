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
        # 240
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

    def forward(self, anchor, positive, negative):
        anchor_emb = self.embedding_model(anchor)
        positive_emb = self.embedding_model(positive)
        negative_emb = self.embedding_model(negative)
        return anchor_emb, positive_emb, negative_emb
        # loss = self.loss(anchor_emb, positive_emb, negative_emb)
        #
        # return loss

    def get_embedding(self, x):
        return self.embedding_model(x)

    def loss(self, anchor, positive, negative, size_average=True):
        loss = self.triplet_loss(anchor, positive, negative)
        return loss

    def predict(self, anchor, positive, negative):
        loss = self.triplet_loss(anchor, positive, negative)
        pred_y = 1 if loss - self.margin < 0 else 0
        # pred_y = 1 if np.random.rand() < 0.5 else 0
        return pred_y, loss