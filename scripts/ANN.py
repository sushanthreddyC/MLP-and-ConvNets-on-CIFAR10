import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import *
from utils import *


class ANN(nn.Module):
    def __init__(self, input_dim: int = 3072, l1=512, l2=512, l3=512,l4=64, p1=0.3, p2=0.3, p3=0.3, p4=0.3, num_classes: int = 10) -> None:
        super(ANN, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, l1) # Increased the width and depth of model to capture more complex relationships.
        self.bn1 = nn.BatchNorm1d(l1) #Implemented Batch Normalization which helps in regularizing the model.
        self.dp1 = nn.Dropout(p1) # Since the model is wider and deeper it tends to overfit the training data easily, to counter it have used Dropout for each layer.
        self.relu1 = nn.ReLU()  

        self.fc2 = nn.Linear(l1, l2) # Experimented with 512,256,128,64 and 32 units for each of the layers. 
        self.bn2 = nn.BatchNorm1d(l2)
        self.dp2 = nn.Dropout(p2)  # Experimented dropout values for 0.1,0.2,0.3. Best results were achieved for Dropout = 0.3 for each layer
        self.relu2 = nn.ReLU()  

        self.fc3 = nn.Linear(l2,l3)
        self.bn3 = nn.BatchNorm1d(l3)
        self.dp3 = nn.Dropout(p3) # Increasing Dropout over 0.3 didn't help model performance. 
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(l3,l4)
        self.bn4 = nn.BatchNorm1d(l4)
        self.dp4 = nn.Dropout(p4)
        self.relu4 = nn.ReLU()

        self.fc5 = nn.Linear(l4, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1) 

        
        x = self.dp1(self.bn1(self.relu1(self.fc1(x))))
        x = self.dp2(self.bn2(self.relu2(self.fc2(x))))
        x = self.dp3(self.bn3(self.relu3(self.fc3(x))))
        x = self.dp4(self.bn4(self.relu4(self.fc4(x))))
        x = self.fc5(x)  # Output layer

        return x
