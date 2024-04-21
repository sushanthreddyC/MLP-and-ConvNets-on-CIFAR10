## Standard libraries
import os
import json
import math
import random
import numpy as np 
import copy
import time

## Imports for plotting
%matplotlib inline
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

INV_DATA_MEANS = torch.tensor([-m for m in DATA_MEANS]).view(-1, 1, 1)
INV_DATA_STD = torch.tensor([1.0 / s for s in DATA_STD]).view(-1, 1, 1)

def loading_batch_time(train_loader):
    start_time = time.time()
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    end_time = time.time()
    print(f"Time for loading a batch: {(end_time - start_time):6.5f}s")

def imshow(img):
    img = img.div_(INV_DATA_STD).sub_(INV_DATA_MEANS) # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()
    plt.close()


def main():
    seed = 42
    random_seed(seed=seed, deterministic=True)

    # Fetching the device that will be used throughout this notebook
    device = (
        torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    )
    print("Using device", device)


    # Loading the training dataset. We need to split it into a training and validation part
    main_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=data_transforms, download=True)
    train_set, val_set = torch.utils.data.random_split(main_dataset, [45000, 5000], generator=torch.Generator().manual_seed(seed))

    # Loading the test set
    # test_set = CIFAR10(root=DATASET_PATH, train=False, transform=data_transforms, download=True)

    # Create data loaders for later
    train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=os.cpu_count())
    # val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers= os.cpu_count())
    # test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers= os.cpu_count())

    loading_batch_time(train_loader)
    imshow(torchvision.utils.make_grid(images))
    print("GroundTruth (1st row): ", " ".join(f"{classes[labels[j]]:5s}" for j in range(8)))