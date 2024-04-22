import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import *
from utils import *

class SGDMomentum(OptimizerTemplate):
    
    def __init__(self, params: nn.ParameterList, lr: float, momentum: float=0.9)->None:
        super().__init__(params, lr)
        self.momentum = momentum # Corresponds to beta_1 in the equation above
        self.param_momentum = {p: torch.zeros_like(p.data) for p in self.params} # Dict to store m_t
        
    def update_param(self, p:nn.Parameter)->None:
        self.param_momentum[p]= self.momentum*self.param_momentum[p] + (1-self.momentum)*p.grad
        p.data -= self.lr*self.param_momentum[p]
        