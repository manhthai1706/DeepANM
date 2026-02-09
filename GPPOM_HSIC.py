"""
Full-Spectrum Deep Learning GPPOM - PyTorch Implementation
Integrated with Unified Kernel Library.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kernels import RBFKernel
import MLP

class RFFGPLayer(nn.Module):
    """Random Fourier Features for Sparse GP Approximation (O(N) complexity)"""
    def __init__(self, input_dim, n_features=256):
        super().__init__()
        self.n_features = n_features
        self.register_buffer("W", torch.randn(input_dim, n_features))
        self.register_buffer("b", torch.rand(n_features) * 2 * np.pi)
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.log_gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        gamma = torch.exp(self.log_gamma)
        alpha = torch.exp(self.log_alpha)
        projection = (x * gamma) @ self.W + self.b
        phi = torch.sqrt(torch.tensor(2.0 / self.n_features)) * torch.cos(projection)
        return phi * torch.sqrt(alpha)

class FastHSIC(nn.Module):
    """
    Adaptive Fast HSIC using RFF (O(N) complexity)
    Automatically optimizes kernel bandwidth (gamma)
    """
    def __init__(self, x_dim, z_dim, n_features=128):
        super().__init__()
        self.phi_x = RFFGPLayer(x_dim, n_features)
        self.phi_z = RFFGPLayer(z_dim, n_features)

    def forward(self, X, Z):
        n = X.shape[0]
        if n < 2: return torch.tensor(0.0, device=X.device)
        
        # Adaptive Bandwidth: we could implement Median Trick or just let SGD optimize log_gamma
        feat_x = self.phi_x(X)
        feat_z = self.phi_z(Z)
        
        feat_x = feat_x - feat_x.mean(dim=0, keepdim=True)
        feat_z = feat_z - feat_z.mean(dim=0, keepdim=True)
        
        covariance = (feat_x.T @ feat_z) / (n - 1)
        return torch.sum(covariance**2)

class GPPOMC_lnhsic_Core(nn.Module):
    """
    Core Architecture Upgraded for:
    - NOTEARS DAG Discovery
    - Multivariate causal relationships
    - Adaptive kernels
    """
    def __init__(self, x_dim, y_dim, n_clusters, hidden_dim, lda, device):
        super().__init__()
        self.lda = lda
        self.device = device
        self.d = x_dim + y_dim # Total variables
        
        # Structural Matrix for DAG Learning (NOTEARS)
        self.W_dag = nn.Parameter(torch.zeros(self.d, self.d))
        
        self.MLP = MLP.MLP(input_dim=self.d, hidden_dim=hidden_dim, 
                          output_dim=self.d, n_clusters=n_clusters, device=device)
        
        self.gp_phi_z = RFFGPLayer(n_clusters, n_features=128)
        self.gp_phi_x = RFFGPLayer(self.d, n_features=128)
        self.linear_head = nn.Linear(128, self.d, bias=False)
        
        # Adaptive HSICs
        self.fast_hsic = FastHSIC(self.d, n_clusters, n_features=128)
        self.pnl_hsic = FastHSIC(self.d, self.d, n_features=128)

    def get_dag_penalty(self):
        """NOTEARS acyclicity constraint h(W)"""
        W_sq = self.W_dag * self.W_dag
        E = torch.matrix_exp(W_sq)
        h = torch.trace(E) - self.d
        return h

    def forward(self, batch_data, temperature=1.0):
        """
        Multivariate forward pass
        batch_data: [batch, d]
        """
        # 1. Structural Masking: Applied via the W_dag matrix
        # In multi-variable learning, we want to predict each node as a function of its parents
        # For simplicity in this framework, we use W_dag to guide the latent mechanism discovery
        
        out = self.MLP(batch_data, temperature=temperature)
        z_soft, kl_loss = out['z_soft'], out['kl_loss']
        mu_mlp, log_var_mlp = out['mu'], out['log_var']
        
        # 2. GP Prediction with Structural Bias
        # Mask the input variables based on learned DAG structure
        masked_input = batch_data @ torch.abs(self.W_dag)
        
        phi = self.gp_phi_z(z_soft) * self.gp_phi_x(masked_input)
        y_pred_gp = self.linear_head(phi)
        
        # 3. NOTEARS & Residual Independence
        loss_dag = self.get_dag_penalty()
        loss_reg = F.mse_loss(y_pred_gp, batch_data)
        
        # PNL Independence
        h_y = self.MLP.pnl_transform(batch_data)
        res_pnl = h_y - y_pred_gp
        loss_hsic_pnl = self.pnl_hsic(batch_data, res_pnl)
        
        loss_hsic_clu = self.fast_hsic(batch_data, z_soft)
        
        # Total Loss with NOTEARS, PNL, and VAE components
        total_loss = (loss_reg + 
                      2.0 * loss_dag + # DAG Constraint
                      self.lda * torch.log(loss_hsic_clu + 1e-8) + 
                      3.0 * torch.log(loss_hsic_pnl + 1e-8) + # Adaptive Kernel PNL
                      0.2 * kl_loss) # VAE Latent discovery
        
        return total_loss, loss_reg, loss_hsic_clu
