# This file is licensed under AGPL-3.0
# Copyright (c) NXAI GmbH and its affiliates 2024
# Benedikt Alkin, Maximilian Beck, Korbinian Pöppel
import os

import torch
from torch import nn
from torch.nn import BatchNorm2d
from torchvision import models
from torchvision.models import ResNet18_Weights

from Model_Code.Detect_head import DetectHead
from Model_Code.RIFM import RIFM
from Model_Code.AICXLSTM import AICXLSTM1, AICXLSTM2, \
    AICXLSTM3, AICXLSTM4

resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
class Layer1(nn.Module):
    def __init__(self):
        super(Layer1, self).__init__()
                # 定义与 ResNet-18 的 layer1 相同的层结构
        self.layer1 = resnet18.layer1

    def forward(self, x):
        return self.layer1(x)
class Layer2(nn.Module):
    def __init__(self):
        super(Layer2, self).__init__()
                # 定义与 ResNet-18 的 layer1 相同的层结构
        self.layer2 = resnet18.layer2

    def forward(self, x):
        return self.layer2(x)
class Layer3(nn.Module):
    def __init__(self):
        super(Layer3, self).__init__()
                # 定义与 ResNet-18 的 layer1 相同的层结构
        self.layer3 = resnet18.layer3

    def forward(self, x):
        return self.layer3(x)

class Conv1(nn.Module):
    def __init__(self):
        super(Conv1, self).__init__()
        # 定义与 ResNet-18 的 layer1 相同的层结构
        self.conv1 = resnet18.conv1

    def forward(self, x):
        return self.conv1(x)
class MMDNet(nn.Module):
    def __init__(self,):
        super(MMDNet, self).__init__()
        self.SICP = nn.ModuleList()
        in_channels = [3, 32,64, 128]
        out_channels = [32,64, 128, 256]
        input_shape = [(3,1024,1024),(32,512,512),(64,256,256),(128,128,128)]
        patch_size = [16,8,4,2]
        resnet_1 = Conv1()
        data_dar = os.path.abspath(__file__)
        data_dar = os.path.dirname(data_dar)
        resnet_1.load_state_dict(
            torch.load(os.path.join(data_dar, 'conv1_resnet18_weights.pth'),weights_only=True))
        self.SICP.append(resnet_1)
        resnet_2 = Layer1()
        resnet_2.load_state_dict(torch.load(os.path.join(data_dar, 'layer1_resnet18_weights.pth'), weights_only=True))
        self.SICP.append(resnet_2)
        resnet_3 = Layer2()
        resnet_3.load_state_dict(torch.load(os.path.join(data_dar, 'layer2_resnet18_weights.pth'),weights_only=True))
        self.SICP.append(resnet_3)
        resnet_4 = Layer3()
        resnet_4.load_state_dict(torch.load(os.path.join(data_dar, 'layer3_resnet18_weights.pth'),weights_only=True))
        self.SICP.append(resnet_4)
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.xlstm = vil_small(pretrained=True)
        self.RIFM1 = RIFM(64,64)
        self.RIFM2 = RIFM(128, 128)
        self.Detect_head1 = DetectHead()
        self.Detect_head2 = DetectHead()
        self.AIC_XLSTM = nn.ModuleList()
        layer1 = AICXLSTM1(in_channels[0],out_channels[0],input_shape=input_shape[0],patch_size=patch_size[0])
        self.AIC_XLSTM.append(layer1)
        layer2 = AICXLSTM2(in_channels[1],out_channels[1],input_shape=input_shape[1],patch_size=patch_size[1])
        self.AIC_XLSTM.append(layer2)
        layer3 = AICXLSTM3(in_channels[2],out_channels[2],input_shape=input_shape[2],patch_size=patch_size[2])
        self.AIC_XLSTM.append(layer3)
        layer4 = AICXLSTM4(in_channels[3],out_channels[3],input_shape=input_shape[3],patch_size=patch_size[3])
        self.AIC_XLSTM.append(layer4)
    def forward(self, x):
        x1 = self.SICP[0](x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = self.AIC_XLSTM[0](x)
        x2 = self.AIC_XLSTM[1](x2)
        tmp_64_1 = x2
        x2 = self.AIC_XLSTM[2](x2)
        tmp_128_1 = x2
        x2 = self.AIC_XLSTM[3](x2)
        x1 = self.maxpool(x1)
        x1 = self.SICP[1](x1)
        x_conv_tmp = self.RIFM1(tmp_64_1,x1)
        x1 = x1+x_conv_tmp
        tmp_64 = x1
        x1 = self.SICP[2](x1)
        x_conv_tmp = self.RIFM2(tmp_128_1,x1)
        x1 = x1+x_conv_tmp
        tmp_128 = x1
        x1 = self.SICP[3](x1)
        x1 = self.Detect_head1(x1,tmp_128,tmp_64)
        x2 = self.Detect_head2(x2,tmp_128_1,tmp_64_1)
        x = x1+x2

        return x
