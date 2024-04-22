import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import *
from utils import *
from MLP import MLP

class ConvNet_modified(nn.Module):
    def __init__(self, in_chans: int=3, num_classes:int=10, act_layer=nn.ReLU)->None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_chans, 256, kernel_size=3, padding=1)  #Increased the width of the network by using 256,128,64 units in ConvNet Layers
        self.bn1 = nn.BatchNorm2d(256) # Used Batch Normalization to improve the performance and avoid overfitting 
        self.dropout1 = nn.Dropout2d(0.3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1) # 
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout2d(0.2) # Tuned values for various dropout values, best results were achieved with [0.3,0.2,0.2,0.3] in the corresponding layers
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act2 = act_layer()

        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout2d(0.2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.act3 = act_layer()
        
    
        # self.num_flatten_features = None
        self.fc1 = nn.Linear(1024, 128) # Increased the model depth by using 3 ConvNet layers and 2 Fully Connected layers
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.3)
        self.act4= act_layer()

        self.fc2 = nn.Linear(128,num_classes)
        
        

    def forward(self, x):
        
        x = self.act1(self.dropout1(self.bn1(self.pool1(self.conv1(x)))))
        x = self.act2(self.dropout2(self.bn2(self.pool2(self.conv2(x)))))
        x = self.act3(self.dropout3(self.bn3(self.pool3(self.conv3(x)))))

        x = x.view(x.size(0), -1)
        x = self.act4(self.dropout4(self.bn4(self.fc1(x))))
        x = self.fc2(x)
        return x