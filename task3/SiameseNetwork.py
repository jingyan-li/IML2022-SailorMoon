"""
Siamese Neural Network
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tr
import torch
from torchvision import models


def getEmbeddingModel(run: dict, device):
    img_size = run['data_cfg.image_resize']
    USE_PRETRAIN = False if run['model_cfg.embedding_model.pretrain_model'] == "" else True
    FEATURE_EXTRACT = run['model_cfg.embedding_model.feature_extract']
    net = EmbeddingNet(img_size, run['model_cfg.embedding_model.out_channels'])
    if USE_PRETRAIN:
        if isinstance(run['model_cfg.embedding_model.pretrain_model'], str):
            net = EmbeddingNetPretrain(**run['model_cfg.embedding_model'])
        else:
            net = EmbeddingNetPretrainMix(**run['model_cfg.embedding_model'], device=device)
    return net, USE_PRETRAIN, FEATURE_EXTRACT


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


class EmbeddingNetPretrainMix(nn.Module):
    def __init__(self, out_channels: int, device, pretrain_model: list = ["resnet"], feature_extract: bool = True):
        """
        Use pretrained vision model as backbone
        :param out_channels: Dimension of embedding to be extracted
        :param device: Whether to put pretrained model to cuda
        :param pretrain_model: name of pretrain model
        :param feature_extract: True if only train the last layer of pretrain model, False to do fine-tuning for all layers
        """
        super(EmbeddingNetPretrainMix, self).__init__()
        # Input is [BatchSize, 3, 240, 240] after image resize
        # TODO: Pretrained networks
        self.device = device
        self.pretrained_nets, embed_dim = self._mix_pretrain_embedding(pretrain_model)

        self.embedding = Embedding1D_v2(in_features=embed_dim, out_features=out_channels)

    def forward(self, x):
        """
        Go through pretrained embedding and concat them as output
        :param x:
        :return:
        """
        mix_embed = []
        for k, m in self.pretrained_nets.items():
            # print(k)
            # print(x.shape)
            if k != "inception":
                x_ = tr.Resize([224, 224])(x)
                mix_embed.append(m(x_))
                del x_
            else:
                incep_out, incep_aux = m(x)
                incep_all = incep_out + 0.4*incep_aux
                mix_embed.append(incep_all)
        embed = torch.cat(mix_embed, 1)
        del mix_embed
        output = self.embedding(embed)
        return output

    def get_embedding(self, x):
        return self.forward(x)

    def _mix_pretrain_embedding(self, pretrain_models):
        mix_models = {}
        model_dim = 0
        for k in pretrain_models:
            if k == "resnet":
                mix_models[k] = models.resnet152(pretrained=True, progress=True).to(self.device)
                model_dim += mix_models[k].fc.out_features
            elif k == "inception":
                mix_models[k] = models.inception_v3(pretrained=True, progress=True).to(self.device)
                model_dim += mix_models[k].fc.out_features
            elif k == "vgg":
                mix_models[k] = models.vgg19_bn(pretrained=True, progress=True).to(self.device)
                model_dim += mix_models[k].classifier[6].out_features
            for param in mix_models[k].parameters():
                param.requires_grad = False
        return mix_models, model_dim


class Embedding1D_v2(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(Embedding1D, self).__init__()

        self.layer1 = nn.Sequential(
            # nn.BatchNorm2d(in_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout2d(0.5),
            nn.Linear(in_features, out_features*2),
            nn.BatchNorm1d(out_features*2),
            nn.PReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Linear(out_features*2, out_features),
            nn.BatchNorm1d(out_features),
            # nn.PReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = x.view(x.shape[0], -1)
        # x = self.layer3(x)
        return x

    def get_embedding(self, x):
        return self.forward(x)


class Embedding1D(nn.Module):
    def __init__(self, in_features: int):
        super(Embedding1D, self).__init__()

        self.layer1 = nn.Sequential(
            # nn.BatchNorm2d(in_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout2d(0.5),
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.PReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Dropout2d(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            # nn.PReLU()
        )


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = x.view(x.shape[0], -1)
        # x = self.layer3(x)
        return x

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


class TripletModel_v2(nn.Module):
    def __init__(self, embedding_model: nn.Module, margin: float, device):
        super(TripletModel_v2, self).__init__()
        self.device = device
        self.embedding_model = embedding_model
        self.margin = margin
        # self.cos_similarity = nn.CosineSimilarity(dim=0, eps=1e-8)
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin,
                                                 p=2).to(
            device)

    def forward(self, anchor, positive, negative):
        anchor_emb = self.embedding_model(anchor)
        positive_emb = self.embedding_model(positive)
        negative_emb = self.embedding_model(negative)

        loss = self.loss(anchor_emb, positive_emb, negative_emb)
        return loss

    def get_embedding(self, x):
        return self.embedding_model(x)

    def loss(self, anchor_emb, positive_emb, negative_emb):
        loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
        return loss

    def _predict(self, anchor_emb, positive_emb, negative_emb):
        distance_positive = torch.norm(anchor_emb - positive_emb, p=2, dim=1)
        distance_negative = torch.norm(anchor_emb - negative_emb, p=2, dim=1)
        dist = distance_positive - distance_negative
        dist = dist.double()
        pred_y = torch.where(dist >= 0, torch.zeros(dist.size()).to(self.device),
                             torch.ones(dist.size()).to(self.device))
        return pred_y

    def predict_with_loss(self, anchor, positive, negative):
        anchor_emb = self.embedding_model(anchor)
        positive_emb = self.embedding_model(positive)
        negative_emb = self.embedding_model(negative)
        loss = self.loss(anchor_emb, positive_emb, negative_emb)
        pred_y = self._predict(anchor_emb, positive_emb, negative_emb)
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

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
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