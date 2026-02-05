# -*- coding: utf-8 -*-
"""
Unified PyTorch Kernels Library
Implementing standard kernels as differentiable modules.
"""

import torch
import torch.nn as nn

class RBFKernel(nn.Module):
    """
    Radial Basis Function (Gaussian) Kernel
    K(x, y) = alpha * exp(-0.5 * gamma * ||x - y||^2)
    """
    def __init__(self, input_dim, ARD=False):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(1))
        # ARD: Automatic Relevance Determination (individual weight for each dimension)
        self.log_gamma = nn.Parameter(torch.zeros(input_dim if ARD else 1))
        self.ARD = ARD

    def forward(self, x1, x2):
        alpha = torch.exp(self.log_alpha)
        gamma = torch.exp(self.log_gamma)
        
        # Scale inputs
        x1_scaled = x1 * torch.sqrt(gamma)
        x2_scaled = x2 * torch.sqrt(gamma)
        
        # Compute squared distance
        # cdist is highly optimized in PyTorch
        dist_sq = torch.cdist(x1_scaled, x2_scaled, p=2)**2
        return alpha * torch.exp(-0.5 * dist_sq)

class LinearKernel(nn.Module):
    """K(x, y) = alpha * x^T y + bias"""
    def __init__(self, input_dim):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.log_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        alpha = torch.exp(self.log_alpha)
        bias = torch.exp(self.log_bias)
        return alpha * (x1 @ x2.T) + bias

class PolynomialKernel(nn.Module):
    """K(x, y) = (alpha * x^T y + bias)^order"""
    def __init__(self, input_dim, order=2):
        super().__init__()
        self.order = order
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.log_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        alpha = torch.exp(self.log_alpha)
        bias = torch.exp(self.log_bias)
        return (alpha * (x1 @ x2.T) + bias) ** self.order

class CombinedKernel(nn.Module):
    """Sum or Product of multiple kernels"""
    def __init__(self, kernels, mode='add'):
        super().__init__()
        self.kernels = nn.ModuleList(kernels)
        self.mode = mode

    def forward(self, x1, x2):
        res = self.kernels[0](x1, x2)
        for i in range(1, len(self.kernels)):
            if self.mode == 'add':
                res = res + self.kernels[i](x1, x2)
            else:
                res = res * self.kernels[i](x1, x2)
        return res

		