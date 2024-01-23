import numpy as np
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from torchsummary import summary

from Attention.ANN import APNB, AFNB
from Attention.CBAM import CBAM
from Attention.GCNet import GCNet
from Attention.RANet import RANet
from Attention.SE import SE
from Attention.Triplet import Triplet
from Attention.scSE import scSE
from nets.inception_resnetv1 import InceptionResnetV1
from nets.mobilenet import MobileNetV1


class mobilenet(nn.Module):
    def __init__(self, pretrained):
        super(mobilenet, self).__init__()
        self.model = MobileNetV1()
        if pretrained:
            state_dict = load_state_dict_from_url(
                "https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_mobilenetv1.pth",
                model_dir="model_data",
                progress=True)
            self.model.load_state_dict(state_dict)

        del self.model.fc
        del self.model.avg

    def forward(self, x):
        x = self.model.stage1(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        return x


class inception_resnet(nn.Module):
    def __init__(self, pretrained):
        super(inception_resnet, self).__init__()
        self.model = InceptionResnetV1()
        if pretrained:
            state_dict = load_state_dict_from_url(
                "https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_inception_resnetv1.pth",
                model_dir="model_data",
                progress=True)
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        x = self.model.conv2d_1a(x)
        x = self.model.conv2d_2a(x)
        x = self.model.conv2d_2b(x)
        x = self.model.maxpool_3a(x)
        x = self.model.conv2d_3b(x)
        x = self.model.conv2d_4a(x)
        x = self.model.conv2d_4b(x)
        x = self.model.repeat_1(x)
        x = self.model.mixed_6a(x)
        x = self.model.repeat_2(x)
        x = self.model.mixed_7a(x)
        x = self.model.repeat_3(x)
        x = self.model.block8(x)
        return x


class Facenet(nn.Module):
    def __init__(self, backbone="mobilenet", attention='CBAM', dropout_keep_prob=0.5, embedding_size=128,
                 num_classes=None, mode="train",
                 pretrained=False):
        super(Facenet, self).__init__()
        if backbone == "mobilenet":
            self.backbone = mobilenet(pretrained)
            flat_shape = 1024
        elif backbone == "inception_resnetv1":
            self.backbone = inception_resnet(pretrained)
            flat_shape = 1792
        else:
            raise ValueError('Unsupported backbone - `{}`.'.format(backbone))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.Dropout = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size, bias=False)
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        if attention == 'CBAM':
            self.attention = CBAM(planes=flat_shape)
        elif attention == 'APNB':
            self.attention = APNB(channel=flat_shape)
        elif attention == 'AFNB':
            self.attention = AFNB(channel=flat_shape)
        elif attention == 'GCNet':
            self.attention = GCNet(inplanes=flat_shape, ratio=0.25)
        elif attention == 'SE':
            self.attention = SE(in_chnls=flat_shape, ratio=16)
        elif attention == 'scSE':
            self.attention = scSE(channel=flat_shape, ratio=16)
        elif attention == 'Triplet':
            self.attention = Triplet()
        else:
            self.attention = None
        if mode == "train":
            self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x, mode="predict"):
        x = self.backbone(x)
        # attention
        if self.attention is not None:
            x = self.attention(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        x = F.normalize(before_normalize, p=2, dim=1)

        if mode == 'predict':
            return x
        cls = self.classifier(before_normalize)
        return x, cls


if __name__ == '__main__':
    a = Facenet(mode='predict', attention="Triplet")
    for name, value in a.named_parameters():
        print(name)
    # device = torch.device('cuda:0')
    # a = a.to(device)
    # a.cuda()
    # summary(a, (3, 224, 224))
