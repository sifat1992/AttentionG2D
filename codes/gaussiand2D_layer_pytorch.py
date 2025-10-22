# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:40:04 2024

@author: szk9
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussNetLayer2D(nn.Module):
    def __init__(self, filter_num, filter_size):
        super(GaussNetLayer2D, self).__init__()
        
        if isinstance(filter_size, int):
            raise TypeError("filter_size must be a tuple (height, width), not an integer")
        
        self.fnum = filter_num
        self.fsize = filter_size
        
        ## Learnable parameters for the means (mu_x, mu_y) and standard deviations (sigma_x, sigma_y)
        self.mean_x = nn.Parameter(torch.full((self.fnum,), 0.01))
        self.mean_y = nn.Parameter(torch.full((self.fnum,), 0.01))
        self.sigma_x = nn.Parameter(torch.full((self.fnum,), 1.0))
        self.sigma_y = nn.Parameter(torch.full((self.fnum,), 1.0))

    def forward(self, input_tensor):
       
        
        ## Creating a mesh grid for X and Y coordinates
        x = torch.linspace(-1.0, 1.0, self.fsize[0], device=input_tensor.device)
        y = torch.linspace(-1.0, 1.0, self.fsize[1], device=input_tensor.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')  # Shape: (height, width)
        
        ## Expand dimensions to match required shape for broadcasting
        X = X.unsqueeze(-1)  # Shape: (height, width, 1)
        Y = Y.unsqueeze(-1)  # Shape: (height, width, 1)

        ## Generating Gaussian filters based on the parameters
        output_list = []
        for i in range(self.fnum):
            gaussian_2d = self.compute_gaussian_2d(X, Y, self.mean_x[i], self.mean_y[i], self.sigma_x[i], self.sigma_y[i])
            output_list.append(gaussian_2d)

        ## Stacking the filters along a new dimension for conv2d compatibility
        filters = torch.stack(output_list, dim=0)  # Shape: (fnum, height, width)
        filters = filters.unsqueeze(1)  # Shape: (fnum, 1, height, width) for conv2d
        filters = filters.squeeze(-1)  # Remove last dimension, resulting in shape [fnum, 1, height, width]

       
        ## Ensuring input_tensor is also in 4D shape
        assert input_tensor.dim() == 4, "input_tensor should be a 4D tensor (batch_size, channels, height, width)"

        ## Calculating padding size to keep output size the same as input size
        padding = (self.fsize[0] // 2, self.fsize[1] // 2)  # Tuple of (height_padding, width_padding)
        

        ## Applying the 2D convolution
        out = F.conv2d(input_tensor, filters, stride=1, padding=padding)
        
        return out
## Utilizing the formula
    def compute_gaussian_2d(self, X, Y, mu_x, mu_y, sigma_x, sigma_y):
        ## Clipping sigma values to avoid very small values
        epsilon = 1e-3  ## To prevent division by zero
        sigma_x = torch.clamp(sigma_x, min=epsilon)
        sigma_y = torch.clamp(sigma_y, min=epsilon)
        
        # 2D Gaussian formula
        gaussian = torch.exp(-(((X - mu_x) ** 2) / (2 * sigma_x ** 2) + ((Y - mu_y) ** 2) / (2 * sigma_y ** 2)))
        gaussian = gaussian / (torch.max(gaussian) + epsilon)  # Normalize to avoid very large values
        return gaussian




