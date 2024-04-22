
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import *
from utils import *
from MLP import MLP
class ConvNet(nn.Module):
    def __init__(self, in_chans: int=3, base_dims: int=32, num_classes:int=10, act_layer=nn.ReLU)->None:
        super().__init__()
        # TODO: 2 points -- write code for two Conv+ReLU+MaxPool blocks
        # raise NotImplementedError
        
        self.conv1 = nn.Conv2d(in_chans, base_dims, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(base_dims, base_dims * 2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act2 = act_layer()
        
    

        self.mlp = MLP(input_dim=base_dims*2, num_classes=num_classes, hidden_dims=[base_dims*4, base_dims*2], act_layer=act_layer)

    def forward(self, x):
        x = self.act1(self.pool1(self.conv1(x)))
        x = self.act2(self.pool2(self.conv2(x)))
        x = x.mean(dim=(2, 3))
        x = self.mlp(x)
        return x