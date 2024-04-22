import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import *
from utils import *
class MLP(nn.Module):

    def __init__(self, input_dim: int=3072, num_classes: int=10, hidden_dims: List[int]=[256, 128], act_layer: nn.Module=nn.ReLU)->None:
        """
        Inputs:
            input_dim - Dimension of the input images in pixels
            num_classes - Number of classes we want to predict. The output size of the MLP
                          should be num_classes.
            hidden_dims - A list of integers specifying the hidden layer dimensions in the MLP. 
                           The MLP should have len(hidden_sizes)+1 linear layers.
            act_layer - Activation function.
        """
        super(MLP, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(len(dims)-1):
            layer = nn.Linear(dims[i], dims[i+1])
            layers.append(layer)
            if i < len(dims)-2 :
                layers.append(act_layer())

        self.layers = nn.Sequential(*layers)
        
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x= x.view(x.size(0),-1)
        for layer in self.layers :
            x = layer(x)
        return x

def sanity_check():
    # Let's test the MLP implementation
    input_dim = np.random.randint(low=64, high=3072)
    num_classes = np.random.randint(low=5, high=20)
    hidden_dims = [np.random.randint(low=32, high=256) for _ in range(np.random.randint(low=1, high=3))]
    my_mlp = MLP(input_dim=input_dim, num_classes=num_classes, hidden_dims=hidden_dims)
    my_mlp.to(device)
    random_input = torch.randn(32, input_dim, device=device)
    random_output = my_mlp(random_input)
    assert random_output.shape[0] == random_input.shape[0]
    assert random_output.shape[1] == num_classes


def main():
    sanity_check()