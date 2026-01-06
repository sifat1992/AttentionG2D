# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 08:49:30 2025

@author: szk9
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import numpy as np
from torch.nn import init
import gaussiand2D_layer_pytorch as gauss
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
from ptflops import get_model_complexity_info


## ======Check for GPU==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
#==============================================================

##==================The hyperparameters for the model==================
BATCH = 32
EPOCH = 100
classes = 10
kernel_size = 3
#=========================================

#============reprodtivility===================
seed = 4051
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#==============================================

#==============SE Attention block==================
class SEAttention(nn.Module):
    def __init__(self, channel,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
#================================================

#===================ECA Attention===============================
class ECAAttention(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
#=========================================================

#============== Defining the VGG-like network architecture====================================
class CustomVGG(nn.Module):
    def __init__(self, num_classes=classes):
        super(CustomVGG, self).__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

       
       
        self.fc1 = nn.Linear(128 * 10 * 10, 512)
        
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
#========================================================================================

##=====================Resnet50==================================================
class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

## Assembling the model
class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(
            image_channels, 16, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=16, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=32, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=64, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=128, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        self.in_channels = intermediate_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

## Utilizing the resnet50 
def ResNet50(img_channel=1, num_classes=classes):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)
#============================================================================

##===========AlexNet===========================================
class Alex(nn.Module):
    def __init__(self, num_classes):
        super(Alex, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=4, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(96)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(96)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.output = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        return x
##=====================================================================

## ====================DenseNet121=====================================
#The bottlneck layers
class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BottleneckLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], 1)  ## Concatenate the input and the output
    
## Transition layers 
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out

## The denseblocks
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(n_layers):
            layers.append(BottleneckLayer(in_channels + i * growth_rate, growth_rate))
        self.denseblock = nn.Sequential(*layers)

    def forward(self, x):
        return self.denseblock(x)

## Assembling the densenet model
class DenseNet(nn.Module):
    def __init__(self, block_config=(6, 12, 24, 16), growth_rate=32, num_classes=classes):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        self.conv1 = nn.Conv2d(1, 2 * growth_rate, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * growth_rate)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        num_channels = 2 * growth_rate
        self.dense_layers = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_channels, growth_rate, num_layers)
            self.dense_layers.append(block)
            num_channels += num_layers * growth_rate

            if i != len(block_config) - 1:
                transition = TransitionLayer(num_channels, num_channels // 2)
                self.dense_layers.append(transition)
                num_channels = num_channels // 2

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        for layer in self.dense_layers:
            x = layer(x)

        x = F.relu(self.bn2(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

## Creating a DenseNet model (among DenseNet-121, DenseNet-169, DenseNet-201, and DenseNet-264, we will use densenet121)
def densenet121(num_classes=classes):
    return DenseNet(block_config=(6, 12, 24, 16), growth_rate=32, num_classes=classes)
#=====================================================================

#=========================MobileNet===========================
class MobileNetV1(nn.Module):
    def __init__(self, ch_in=1, n_classes=classes):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            #conv_dw(256, 512, 2),
            #conv_dw(512, 512, 1),
            #conv_dw(512, 512, 1),
            #conv_dw(512, 512, 1),
            #conv_dw(512, 512, 1),
            #conv_dw(512, 512, 1),
            #conv_dw(512, 1024, 2),
            #conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x
#===============================================================

#===============CustomCNN======================================
class CustomCNN(nn.Module):
    def __init__(self, num_classes=classes):
        super(CustomCNN, self).__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 1 * 1, 256)  
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
#===================================================================

##======================Gauss2d===========================================
class Gauss(nn.Module):
    def __init__(self, num_classes=classes):
        super(Gauss, self).__init__()

        self.block1 = nn.Sequential(gauss.GaussNetLayer2D(32, (kernel_size, kernel_size)),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 1 * 1, 256)  
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        #x = self.gauss_layer(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
#==========================================================

##==================================Gauss+SE================================================
class GaussSE(nn.Module):
    def __init__(self, num_classes=classes):
        super(GaussSE, self).__init__()
        
        #SEblock==================
        self.se1 = SEAttention(64)
        self.se2 = SEAttention(64)
        self.se3 = SEAttention(128)
        self.se4 = SEAttention(128)

        
        #Block 1
        self.block1 = nn.Sequential( gauss.GaussNetLayer2D(64, (kernel_size, kernel_size)),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 1 * 1, 256)  
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        #x = self.gauss_layer(x)
        x = self.block1(x)
        x = self.se1(x)
        x = self.block2(x)
        x = self.se2(x)
        x = self.block3(x)
        x = self.se3(x)
        x = self.block4(x)
        x = self.se4(x)
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
#===============================================================

#===============ECA64===========================================
class ECA(nn.Module):
    def __init__(self, num_classes=classes):
        super(ECA, self).__init__()
        
        #SEblock=======================================================================================================
        self.ec1 = ECAAttention(64)
        self.ec2 =ECAAttention(64)
        self.ec3 = ECAAttention(128)
        self.ec4 = ECAAttention(128)

       
        #Block 1
        self.block1 = nn.Sequential( gauss.GaussNetLayer2D(64, (kernel_size, kernel_size)),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 1 * 1, 256)  
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
      
        x = self.block1(x)
        x = self.ec1(x)
        x = self.block2(x)
        x = self.ec2(x)
        x = self.block3(x)
        x = self.ec3(x)
        x = self.block4(x)
        x = self.ec4(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#==========Model Registry================================
MODEL_REGISTRY = {
    "gauss": Gauss,
    "gauss_se": GaussSE,
    "custom": CustomCNN,
    "mobilenet": MobileNetV1,
    "resnet50": ResNet50,
    "dense121": DenseNet,
    "vgg16": CustomVGG,
    "alex" : Alex,
    "eca": ECA
}

# ============================================
# # Function to fetch model by name
# # ============================================
# def get_model(model_name, num_classes=classes):
#     if model_name not in MODEL_REGISTRY:
#         raise ValueError(f"Unknown model: {model_name}")
#     return MODEL_REGISTRY[model_name](num_classes)  
# #====================================================   

def get_model(model_name, num_classes=classes):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    ModelClass = MODEL_REGISTRY[model_name]

    # Check what arguments the model wants
    import inspect
    sig = inspect.signature(ModelClass.__init__)
    params = sig.parameters

    # Build kwargs dynamically
    kwargs = {}

    if "ch_in" in params:
        kwargs["ch_in"] = 1
    if "num_classes" in params:
        kwargs["num_classes"] = num_classes
    if "n_classes" in params:
        kwargs["n_classes"] = num_classes

    return ModelClass(**kwargs)

