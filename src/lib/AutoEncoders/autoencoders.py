import torch
import torch.nn as nn
from torch.optim import adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



class FFAutoEncoder(nn.Module):
    
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, latent_dim), # bottle neck
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 50), # bottle neck
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x) # encoding
        x = self.decoder(x) # reconstruction
        return x