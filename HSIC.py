"""
Hilbert Schmidt Independence Criterion (HSIC) - Professional Statistical Suite
Includes Gamma Approximation and Permutation Testing.
"""

import torch
import numpy as np
from scipy.stats import gamma

def center_kernel(K):
    """Memory-efficient kernel centering: O(N^2) time, O(1) extra space"""
    n = K.shape[0]
    row_mean = torch.mean(K, dim=0, keepdim=True)
    col_mean = torch.mean(K, dim=1, keepdim=True)
    total_mean = torch.mean(K)
    return K - row_mean - col_mean + total_mean

def rbf_kernel_fixed(x, y, sigma):
    """Stable RBF Kernel"""
    dist_sq = torch.cdist(x, y, p=2)**2
    return torch.exp(-dist_sq / (2 * sigma**2))

def hsic_gam(X, Y, alpha=0.05):
    """
    Gamma-approximated HSIC Test.
    Returns: test_stat, threshold, p_value
    """
    if isinstance(X, np.ndarray): X = torch.from_numpy(X).float()
    if isinstance(Y, np.ndarray): Y = torch.from_numpy(Y).float()
    
    n = X.shape[0]
    # Use double precision for statistical significance
    X, Y = X.double(), Y.double()

    def get_sigma(data):
        dists = torch.cdist(data, data, p=2)**2
        median_dist = torch.median(dists[dists > 0])
        return torch.sqrt(0.5 * median_dist + 1e-8)

    K = center_kernel(rbf_kernel_fixed(X, X, get_sigma(X)))
    L = center_kernel(rbf_kernel_fixed(Y, Y, get_sigma(Y)))

    test_stat = torch.sum(K * L) / n

    # Expected value under H0
    # Reference: Gretton et al. (2007)
    mu_x = (torch.sum(K) - torch.trace(K)) / (n * (n - 1))
    mu_y = (torch.sum(L) - torch.trace(L)) / (n * (n - 1))
    m_hsic = (1.0 + mu_x * mu_y - mu_x - mu_y) / n

    # Variance under H0
    var_hsic = (K * L / 6.0)**2
    var_hsic = (torch.sum(var_hsic) - torch.trace(var_hsic)) / (n * (n - 1))
    var_hsic = var_hsic * 72 * (n - 4) * (n - 5) / (n * (n - 1) * (n - 2) * (n - 3))

    # Fit Gamma distribution
    al = (m_hsic**2) / (var_hsic + 1e-9)
    bet = (var_hsic * n) / (m_hsic + 1e-9)

    # Scipy for threshold and p-value
    thresh = gamma.ppf(1 - alpha, al.cpu().item(), scale=bet.cpu().item())
    p_value = 1 - gamma.cdf(test_stat.cpu().item(), al.cpu().item(), scale=bet.cpu().item())

    return test_stat.item(), thresh, p_value

def hsic_perm(X, Y, n_perms=500):
    """
    Permutation-based HSIC Test (Robust but slower).
    Returns: p_value
    """
    if isinstance(X, np.ndarray): X = torch.from_numpy(X).float()
    if isinstance(Y, np.ndarray): Y = torch.from_numpy(Y).float()
    n = X.shape[0]
    
    # Calculate base HSIC
    stat, _, _ = hsic_gam(X, Y)
    
    count = 0
    for _ in range(n_perms):
        # Permute Y only
        idx = torch.randperm(n)
        Y_perm = Y[idx]
        p_stat, _, _ = hsic_gam(X, Y_perm)
        if p_stat >= stat:
            count += 1
            
    return count / n_perms
