## Standard libraries
import os
import json
import math
import random
import numpy as np 
import copy
import time

## Imports for plotting
# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
sns.set_theme()

## Progress bar
from tqdm.notebook import tqdm

## typing
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Subset
import torch.optim as optim

## PyTorch Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms

from data import *
from utils import *
from ANN import ANN
from sgd import SGDMomentum
from vanila_convnet import ConvNet
from convnet_improved import ConvNet_modified

def train_ANN(train_loader, val_loader):
    # Create model, optimizer, and start training
    model_ann = ANN().to(device)
    optimizer = SGDMomentum(model_ann.parameters(), lr=0.1) # Tuned the model with various learning rates [0.1,0.01,0.001]. Best results were achieved with 0.1
    loss_module = nn.CrossEntropyLoss().to(device)
    print("=" * 40, f"Training started for seed value={seed}", "=" * 40)
    model_ann=train_model(model_ann, optimizer, loss_module, train_loader, val_loader, num_epochs=30, model_name="ann_1") # Trained the model for 30 epochs

    # Test best model on test set

    print("=" * 40, f"Test best model on test set for seed value={seed}", "=" * 40)
    vanilla_Ann_test_acc = test_model(model_ann, test_loader)
    print(f'Test accuracy: {vanilla_Ann_test_acc*100.0:05.2f}%')


def train_vanila_convnet(train_loader, val_loader):
    model_convnet = ConvNet(act_layer=nn.ReLU).to(device)
    optimizer = SGDMomentum(model_convnet.parameters(), lr=0.1)
    loss_module = nn.CrossEntropyLoss().to(device)

    print(f"model convnet created: {count_parameters(model_convnet):05.3f}M")
    model_convnet = train_model(
        model_convnet,
        optimizer,
        loss_module,
        train_loader,
        val_loader,
        num_epochs=5,
        model_name="myConvNet_ReLU",
    )
    # Test best model on test set
    vanilla_convnet_test_acc = test_model(model_convnet, test_loader)
    print(f"Test accuracy: {vanilla_convnet_test_acc*100.0:05.2f}%")


def improved_convnet(train_loader, val_loader):
    model_convnet_modified = ConvNet_modified(act_layer=nn.ReLU).to(device)
    optimizer = SGDMomentum(model_convnet_modified.parameters(), lr=0.1)
    loss_module = nn.CrossEntropyLoss().to(device)

    print(f"model convnet created: {count_parameters(model_convnet_modified):05.3f}M")
    print("=" * 40, f"Training started for seed value={seed}", "=" * 40)
    model_convnet_modified = train_model(
        model_convnet_modified,
        optimizer,
        loss_module,
        train_loader,
        val_loader,
        num_epochs=30,
        model_name="ConvNet_modified_ReLU",
    )
    print("=" * 40, f"Test best model on test set for seed value={seed}", "=" * 40)
    # Test best model on test set
    convnet_test_acc = test_model(model_convnet_modified, test_loader)
    print(f"Test accuracy: {convnet_test_acc*100.0:05.2f}%")



def main():

    random_seed(seed=seed, deterministic=True)

    # Create data loaders for later
    train_loader, val_loader= prepare_dataloaders(main_dataset)

    





