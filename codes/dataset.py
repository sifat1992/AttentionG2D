# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 12:46:14 2025

@author: szk9
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class H5AugmentedDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx].squeeze()  # [H, W]
        image = (image * 255).astype("uint8")
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = self.y[idx].item()  
        return image, label
    
    
    
    

