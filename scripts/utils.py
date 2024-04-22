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





# Function for setting the seed
def random_seed(seed: int = 42, rank: int = 0, deterministic: bool = False) -> None:
    # TODO: 2 points  - write your code below
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    # GPU operations have a separate seed we also want to set
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)

        # Additionally, some operations on a GPU are implemented stochastic for efficiency
        # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = True


def prepare_dataloaders(main_dataset=main_dataset, seed: int = 42):
    # Loading the training dataset. We need to split it into a training and validation part    
    train_set, val_set = torch.utils.data.random_split(main_dataset, [45000, 5000], generator=torch.Generator().manual_seed(seed))

    # Create data loaders for later
    train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=os.cpu_count())
    val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers= os.cpu_count())
    # test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers= os.cpu_count())
    return train_loader,val_loader

class OptimizerTemplate:
    
    def __init__(self, params: nn.ParameterList, lr: float)->None:
        self.params = list(params)
        self.lr = lr
        
    def zero_grad(self)->None:
        ## Set gradients of all parameters to zero
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_() # For second-order optimizers important
                p.grad.zero_()
    
    @torch.no_grad()
    def step(self)->None:
        ## Apply update step to all parameters
        for p in self.params:
            if p.grad is None: # We skip parameters without any gradients
                continue
            self.update_param(p)
            
    def update_param(self, p: nn.Parameter)->None:
        # To be implemented in optimizer-specific classes
        raise NotImplementedError
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def train_one_epoch(model: nn.Module, optimizer: OptimizerTemplate, loss_module, data_loader) -> Tuple[float, int]:
    true_preds, count = 0.0, 0
    model.train()  
    for imgs, labels in data_loader:
        # TODO: 10 points -- Implement training loop with training on classification
        # raise NotImplementedError
        optimizer.zero_grad()  
        img = imgs.to(device)
        label=labels.to(device)
        outputs = model(img) 
        loss = loss_module(outputs, label)  
        loss.backward()  
        optimizer.step() 
        _, predicted = torch.max(outputs, 1)
        
        true_preds += (predicted == label).sum().item()
        count += labels.size(0)
        
    train_acc = true_preds / count 
    return train_acc

@torch.no_grad()
def test_model(model, data_loader):
    # TODO: 10 points - Test model and return accuracy
    model.to(device)
    model.eval()
    total_count, correct_preds = 0,0.0
    for imgs,labels in data_loader:
        img = imgs.to(device)
        label=labels.to(device)
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        correct_preds += (label==predicted).sum().item()
        total_count += labels.size(0)
    test_accuracy = (correct_preds/total_count)
    return test_accuracy
    # raise NotImplementedError
    
def save_model(model, model_name, root_dir=CHECKPOINT_PATH):
    # TODO: 2 points -- Save the parameters of the model
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)  # Create the directory if it does not exist
    save_path = os.path.join(root_dir, model_name + '.pth')  # Path to save the model
    torch.save(model.state_dict(), save_path)
    # raise NotImplementedError
    

def load_model(model, model_name, root_dir=CHECKPOINT_PATH):
    # TODO: 2 points -- Load the parameters of the model
    load_path = os.path.join(root_dir, model_name + '.pth')  # Path to load the model from
    model.load_state_dict(torch.load(load_path))
    return model
    # raise NotImplementedError

def train_model(model, optimizer, loss_module, train_data_loader, val_data_loader, num_epochs=25, model_name="MyModel"):
    # Set model to train mode
    model.to(device)
    best_val_acc = -1.0

    # Training loop
    for epoch in range(1, num_epochs+1):
        train_acc = train_one_epoch(model, optimizer, loss_module, train_data_loader)

        if (epoch+1) % 2 == 0 or epoch == num_epochs:
            # Evaluate the model and save if best
            acc = test_model(model, val_data_loader)
            if acc > best_val_acc:
                best_val_acc = acc 
                save_model(model, model_name, CHECKPOINT_PATH)

            print(
                f"[Epoch {epoch+1:2d}] Training accuracy: {train_acc*100.0:05.2f}%, Validation accuracy: {acc*100.0:05.2f}%, Best validation accuracy: {best_val_acc*100.0:05.2f}%"
            )

    # Load best model after training
    model = load_model(model, model_name, CHECKPOINT_PATH)
    return model 

def shuffle_pixels(imgs: torch.Tensor, shuffle_idx_shared: Optional[torch.Tensor] = None) -> torch.Tensor:
    B, C, H, W = imgs.shape
    imgs_flat = imgs.view(B, C, H*W)
    if shuffle_idx_shared is not None:
        # shuffle the pixels using the provided shuffle idx
        imgs_shuffled = torch.gather(imgs_flat, 2, shuffle_idx_shared[None, None, :].expand(B, C, H*W))
        
    else:
        # Sample a shuffle idx and then shuffle the pixels
        shuffle_idx_shared = torch.randperm(H*W, device=imgs.device)
        imgs_shuffled = torch.gather(imgs_flat, 2, shuffle_idx_shared[None, None, :].expand(B, C, H*W))
        
    imgs_shuffled = imgs_shuffled.view(B, C, H, W)
    return imgs_shuffled