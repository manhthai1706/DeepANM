"""
Unified PyTorch Kernels Library / Thư viện Kernel thống nhất
"""

import torch
import torch.nn as nn
import numpy as np

class RBFKernel(nn.Module):
    """Gaussian (RBF) Kernel / Hàm nhân Gaussian"""
    def __init__(self, input_dim, ARD=False):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(1)) # Variance parameter / Tham số phương sai
        # ARD: Individual weights for each dimension / Trọng số riêng cho từng chiều
        self.log_gamma = nn.Parameter(torch.zeros(input_dim if ARD else 1)) 
        self.ARD = ARD # ARD flag / Cờ hiệu ARD

    def forward(self, x1, x2):
        alpha = torch.exp(self.log_alpha) # Scale factor / Hệ số tỉ lệ
        gamma = torch.exp(self.log_gamma) # Bandwidth factor / Hệ số độ rộng
        
        # Scale inputs by bandwidth / Thu phóng đầu vào theo độ rộng
        x1_scaled = x1 * torch.sqrt(gamma)
        x2_scaled = x2 * torch.sqrt(gamma)
        
        # Optimized squared distance / Tính bình phương khoảng cách tối ưu
        dist_sq = torch.cdist(x1_scaled, x2_scaled, p=2)**2
        return alpha * torch.exp(-0.5 * dist_sq) # Gaussian formula / Công thức Gaussian

class LinearKernel(nn.Module):
    """Linear Kernel: K(x, y) = alpha * x^T y + bias / Hàm nhân tuyến tính"""
    def __init__(self, input_dim):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(1)) # Scale parameter / Tham số tỉ lệ
        self.log_bias = nn.Parameter(torch.zeros(1)) # Bias parameter / Tham số chệch

    def forward(self, x1, x2):
        alpha = torch.exp(self.log_alpha) # Get alpha / Lấy giá trị alpha
        bias = torch.exp(self.log_bias) # Get bias / Lấy giá trị bias
        return alpha * (x1 @ x2.T) + bias # Dot product plus bias / Tích vô hướng cộng sai số

class PolynomialKernel(nn.Module):
    """Polynomial Kernel / Hàm nhân đa thức"""
    def __init__(self, input_dim, order=2):
        super().__init__()
        self.order = order # Polynomial degree / Bậc đa thức
        self.log_alpha = nn.Parameter(torch.zeros(1)) # Scale parameter / Tham số tỉ lệ
        self.log_bias = nn.Parameter(torch.zeros(1)) # Bias parameter / Tham số chệch

    def forward(self, x1, x2):
        alpha = torch.exp(self.log_alpha) # Get alpha / Lấy giá trị alpha
        bias = torch.exp(self.log_bias) # Get bias / Lấy giá trị bias
        return (alpha * (x1 @ x2.T) + bias) ** self.order # Polynomial formula / Công thức đa thức

class CombinedKernel(nn.Module):
    """Combined Kernels (Add/Mul) / Kết hợp nhiều hàm nhân"""
    def __init__(self, kernels, mode='add'):
        super().__init__()
        self.kernels = nn.ModuleList(kernels) # List of kernels / Danh sách hàm nhân
        self.mode = mode # Combination mode / Chế độ kết hợp

    def forward(self, x1, x2):
        res = self.kernels[0](x1, x2) # Initial kernel result / Kết quả hàm nhân đầu tiên
        for i in range(1, len(self.kernels)): # Loop through others / Lặp qua các hàm còn lại
            if self.mode == 'add': # Summode / Chế độ cộng
                res = res + self.kernels[i](x1, x2)
            else: # Product mode / Chế độ nhân
                res = res * self.kernels[i](x1, x2)
        return res # Combined result / Kết quả tổng hợp

class MaternKernel(nn.Module):
    """Matern Kernel (nu=1.5 or 2.5) / Hàm nhân Matern"""
    def __init__(self, input_dim, nu=1.5):
        super().__init__()
        self.nu = nu # nu parameter / Tham số nu
        self.log_alpha = nn.Parameter(torch.zeros(1)) # Scale / Tham số tỉ lệ
        self.log_gamma = nn.Parameter(torch.zeros(1)) # Bandwidth / Tham số độ rộng

    def forward(self, x1, x2):
        alpha = torch.exp(self.log_alpha) # Get alpha / Lấy alpha
        gamma = torch.exp(self.log_gamma) # Get gamma / Lấy gamma
        
        # Calculate scaled distance / Tính khoảng cách đã thu phóng
        dist = torch.cdist(x1 * torch.sqrt(gamma), x2 * torch.sqrt(gamma), p=2)
        
        if self.nu == 1.5: # nu=1.5 formula / Công thức nu=1.5
            sqrt3 = np.sqrt(3.0)
            res = (1.0 + sqrt3 * dist) * torch.exp(-sqrt3 * dist)
        elif self.nu == 2.5: # nu=2.5 formula / Công thức nu=2.5
            sqrt5 = np.sqrt(5.0)
            res = (1.0 + sqrt5 * dist + (5.0/3.0) * dist**2) * torch.exp(-sqrt5 * dist)
        else: # nu=0.5 fallback / Trường hợp nu=0.5
            res = torch.exp(-dist) # Laplacian kernel / Hàm nhân Laplacian
            
        return alpha * res # Final result / Kết quả cuối cùng

class RationalQuadraticKernel(nn.Module):
    """Rational Quadratic Kernel / Hàm nhân Rational Quadratic"""
    def __init__(self, input_dim):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(1)) # Output scale / Tỉ lệ đầu ra
        self.log_l = nn.Parameter(torch.zeros(1)) # Lengthscale / Độ dài tỉ lệ
        self.log_alpha_rk = nn.Parameter(torch.zeros(1)) # RK alpha / Tham số alpha RK

    def forward(self, x1, x2):
        alpha = torch.exp(self.log_alpha) # Get alpha / Lấy alpha
        l = torch.exp(self.log_l) # Get l / Lấy l
        alpha_rk = torch.exp(self.log_alpha_rk) # Get RK_alpha / Lấy alpha_rk
        
        dist_sq = torch.cdist(x1, x2, p=2)**2 # Squared distance / Bình phương khoảng cách
        res = (1.0 + dist_sq / (2 * alpha_rk * l**2))**(-alpha_rk) # Formula / Công thức
        return alpha * res # Final result / Kết quả cuối cùng
