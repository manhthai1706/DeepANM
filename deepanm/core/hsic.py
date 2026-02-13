"""
Hilbert Schmidt Independence Criterion (HSIC)
Kiểm định tính độc lập thống kê HSIC
"""

import torch
import numpy as np
from scipy.stats import gamma

def center_kernel(K):
    """Memory-efficient kernel centering / Chuẩn hóa tâm ma trận Kernel"""
    n = K.shape[0] # Number of samples / Số lượng mẫu
    row_mean = torch.mean(K, dim=0, keepdim=True) # Column mean / Trung bình cột
    col_mean = torch.mean(K, dim=1, keepdim=True) # Row mean / Trung bình hàng
    total_mean = torch.mean(K) # Grand mean / Trung bình tổng
    return K - row_mean - col_mean + total_mean # Centered matrix / Ma trận đã chuẩn hóa tâm

def rbf_kernel_fixed(x, y, sigma):
    """Stable RBF Kernel / Hàm nhân RBF ổn định"""
    dist_sq = torch.cdist(x, y, p=2)**2 # Squared Euclidean distance / Bình phương khoảng cách Euclid
    return torch.exp(-dist_sq / (2 * sigma**2)) # Gaussian RBF formula / Công thức Gaussian RBF

def hsic_gam(X, Y, alpha=0.05):
    """Gamma-approximated HSIC Test / Kiểm định HSIC bằng xấp xỉ Gamma"""
    if isinstance(X, np.ndarray): X = torch.from_numpy(X).float() # Convert numpy to torch / Chuyển từ numpy sang torch
    if isinstance(Y, np.ndarray): Y = torch.from_numpy(Y).float() # Convert numpy to torch / Chuyển từ numpy sang torch
    
    n = X.shape[0] # Number of samples / Số lượng mẫu
    X, Y = X.double(), Y.double() # Use double precision / Sử dụng độ chính xác kép

    def get_sigma(data):
        """Estimate kernel bandwidth / Ước lượng độ rộng kernel"""
        dists = torch.cdist(data, data, p=2)**2 # Target distances / Khoảng cách đích
        median_dist = torch.median(dists[dists > 0]) # Median trick / Kỹ thuật lấy trung vị
        return torch.sqrt(0.5 * median_dist + 1e-8) # Resulting sigma / Giá trị sigma kết quả

    # Center the kernels / Chuẩn hóa tâm các kernel
    K = center_kernel(rbf_kernel_fixed(X, X, get_sigma(X)))
    L = center_kernel(rbf_kernel_fixed(Y, Y, get_sigma(Y)))

    test_stat = torch.sum(K * L) / n # HSIC statistic / Chỉ số thống kê HSIC

    # Expected value under H0 (Independence) / Giá trị kỳ vọng khi X, Y độc lập
    mu_x = (torch.sum(K) - torch.trace(K)) / (n * (n - 1))
    mu_y = (torch.sum(L) - torch.trace(L)) / (n * (n - 1))
    m_hsic = (1.0 + mu_x * mu_y - mu_x - mu_y) / n

    # Variance under H0 / Phương sai khi X, Y độc lập
    var_hsic = (K * L / 6.0)**2
    var_hsic = (torch.sum(var_hsic) - torch.trace(var_hsic)) / (n * (n - 1))
    var_hsic = var_hsic * 72 * (n - 4) * (n - 5) / (n * (n - 1) * (n - 2) * (n - 3))

    # Calculate Gamma parameters / Tính toán tham số phân phối Gamma
    al = (m_hsic**2) / (var_hsic + 1e-9) # Shape parameter / Tham số hình dạng
    bet = (var_hsic * n) / (m_hsic + 1e-9) # Scale parameter / Tham số tỉ lệ

    # Compute p-value and threshold / Tính giá trị p và ngưỡng
    thresh = gamma.ppf(1 - alpha, al.cpu().item(), scale=bet.cpu().item())
    p_value = 1 - gamma.cdf(test_stat.cpu().item(), al.cpu().item(), scale=bet.cpu().item())

    return test_stat.item(), thresh, p_value # Return results / Trả về kết quả

def hsic_perm(X, Y, n_perms=500):
    """Permutation HSIC Test / Kiểm định HSIC bằng hoán vị"""
    if isinstance(X, np.ndarray): X = torch.from_numpy(X).float()
    if isinstance(Y, np.ndarray): Y = torch.from_numpy(Y).float()
    n = X.shape[0] # Number of samples / Số lượng mẫu
    
    stat, _, _ = hsic_gam(X, Y) # Base statistic / Thống kê gốc
    
    count = 0 # Counter for permutations / Biến đếm hoán vị
    for _ in range(n_perms):
        idx = torch.randperm(n) # Shuffle indices / Xáo trộn chỉ số
        Y_perm = Y[idx] # Permute Y / Hoán vị biến Y
        p_stat, _, _ = hsic_gam(X, Y_perm) # New statistic / Thống kê mới
        if p_stat >= stat: # Compare / So sánh
            count += 1 # Increment / Tăng biến đếm
            
    return count / n_perms # Estimated p-value / Giá trị p ước lượng
