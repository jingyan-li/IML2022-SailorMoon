"""
Siamese Neural Network
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torchvision import models


class EmbeddingNet(nn.Module):
    def __init__(self, img_size: tuple, out_channels: int):
        super(EmbeddingNet, self).__init__()
        # Input is [BatchSize, 3, 348, 512]
        # Input is [BatchSize, 3, 240, 240] after image resize
        # TODO: CNN structure
        self.convnet = nn.Sequential(nn.Conv2d(3, 8, 5, padding=2),
                                     nn.PReLU(), nn.BatchNorm2d(8),
                                     nn.MaxPool2d(2, stride=2),     # [B, 8, 174, 256] # 120
                                     # ==========================================================
                                     nn.Conv2d(8, 16, 5, padding=2),
                                     nn.PReLU(), nn.BatchNorm2d(16),
                                     nn.MaxPool2d(2, stride=2))     # [B, 16, 87, 128] # 60

        self.fc = nn.Sequential(nn.Linear(16 * (img_size[0]//4) * (img_size[1]//4), 128),
                                nn.PReLU(),
                                nn.Linear(128, 128),
                                nn.PReLU(),
                                nn.Linear(128, out_channels)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetPretrain(nn.Module):
    def __init__(self, out_channels: int, pretrain_model: str = "resnet", feature_extract: bool = True):
        """
        Use pretrained vision model as backbone
        :param out_channels: Dimension of embedding to be extracted
        :param pretrain_model: name of pretrain model
        :param feature_extract: True if only train the last layer of pretrain model, False to do fine-tuning for all layers
        """
        super(EmbeddingNetPretrain, self).__init__()
        # Input is [BatchSize, 3, 240, 240] after image resize
        # TODO: Pretrained network
        self.pretrained_net, self.img_size = initialize_model(model_name=pretrain_model,
                                                              num_classes=out_channels,
                                                              feature_extract=feature_extract)

    def forward(self, x):
        output = self.pretrained_net(x)
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
        return self.embedding_model(x)

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
        positive = self.get_embedding(positive)
        negative = self.get_embedding(negative)
        # print(f"Triplet model predict: anchor shape {anchor.shape}")
        loss = self.triplet_loss_batch(anchor, positive, negative)
        # print(f"Triplet model predict: loss shape {loss.shape}")
        pred_y = torch.where(loss - self.margin < 0, 1, 0)
        # print(f"Triplet model predict: pred_y shape {pred_y.shape}")
        return pred_y, loss


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False