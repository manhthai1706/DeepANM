"""
Unified PyTorch Kernels Library
Implementing standard kernels as differentiable modules.
"""

import torch
import torch.nn as nn
import numpy as np

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

class MaternKernel(nn.Module):
    """
    Matern Kernel (nu=1.5 or 2.5)
    nu=1.5: (1 + sqrt(3)*d) * exp(-sqrt(3)*d)
    nu=2.5: (1 + sqrt(5)*d + 5/3*d^2) * exp(-sqrt(5)*d)
    """
    def __init__(self, input_dim, nu=1.5):
        super().__init__()
        self.nu = nu
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.log_gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        alpha = torch.exp(self.log_alpha)
        gamma = torch.exp(self.log_gamma)
        
        dist = torch.cdist(x1 * torch.sqrt(gamma), x2 * torch.sqrt(gamma), p=2)
        
        if self.nu == 1.5:
            sqrt3 = np.sqrt(3.0)
            res = (1.0 + sqrt3 * dist) * torch.exp(-sqrt3 * dist)
        elif self.nu == 2.5:
            sqrt5 = np.sqrt(5.0)
            res = (1.0 + sqrt5 * dist + (5.0/3.0) * dist**2) * torch.exp(-sqrt5 * dist)
        else:
            res = torch.exp(-dist) # nu = 0.5 (Laplacian/Exponential)
            
        return alpha * res

class RationalQuadraticKernel(nn.Module):
    """K(x, y) = alpha * (1 + ||x-y||^2 / (2 * alpha_rk * l^2))^-alpha_rk"""
    def __init__(self, input_dim):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.log_l = nn.Parameter(torch.zeros(1))
        self.log_alpha_rk = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        alpha = torch.exp(self.log_alpha)
        l = torch.exp(self.log_l)
        alpha_rk = torch.exp(self.log_alpha_rk)
        
        dist_sq = torch.cdist(x1, x2, p=2)**2
        res = (1.0 + dist_sq / (2 * alpha_rk * l**2))**(-alpha_rk)
        return alpha * res
