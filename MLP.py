# -*- coding: utf-8 -*-
"""
Ultimate Deep Learning MLP - PyTorch Implementation
Integrated Features:
1. Self-Attention (Feature Importance)
2. Gumbel-Softmax (End-to-End Differentiable Clustering)
3. Normalizing Flows (RealNVP Coupling Layers for Complex Noise)
4. Residual Connections & LayerNorm
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class AttentionLayer(nn.Module):
    """Self-Attention for weighting input features"""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        # x: (Batch, Dim) -> (Batch, 1, Dim) for attention
        x_in = x.unsqueeze(1)
        q = self.query(x_in)
        k = self.key(x_in)
        v = self.value(x_in)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).squeeze(1)
        return out + x # Residual

class RealNVPLayer(nn.Module):
    """Coupling Layer for Normalizing Flows"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.mask = nn.Parameter(torch.arange(dim) % 2, requires_grad=False)
        self.s_net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim), nn.Tanh())
        self.t_net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x):
        x1 = x * self.mask
        s = self.s_net(x1) * (1 - self.mask)
        t = self.t_net(x1) * (1 - self.mask)
        y = x1 + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=-1)
        return y, log_det

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
    def forward(self, x):
        return F.gelu(x + self.block(x))

class MLP(nn.Module):
    """
    Advanced Multi-Head Architecture for Causal Inference.
    - Latent Head: Gumbel-Softmax for clustering (Mechanism identification)
    - Regressor Head: Attention + Residuals for mapping X to Y
    - Noise Head: RealNVP for modeling complex non-Gaussian noise
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_clusters=2, device='cpu'):
        super().__init__()
        self.device = device
        self.n_clusters = n_clusters
        
        # 1. Attention & Shared Backbone
        self.attention = AttentionLayer(input_dim)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim)
        )
        
        # 2. Clustering Head (Gumbel-Softmax)
        self.cluster_head = nn.Linear(hidden_dim, n_clusters)
        
        # 3. Probabilistic Regressor Head
        self.regressor = nn.Linear(hidden_dim, output_dim * 2) # Mean and LogVar
        
        # 4. Normalizing Flow for Noise
        self.flow = RealNVPLayer(output_dim, hidden_dim // 2)
        
        self.to(device)
        
    def forward(self, x, temperature=1.0):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
            
        # Attention
        x_weighted = self.attention(x)
        feat = self.backbone(x_weighted)
        
        # Latent Clustering (Categorical Latent / VAE Head)
        logits = self.cluster_head(feat)
        probs = F.softmax(logits, dim=-1)
        
        # KL Divergence against Uniform Prior (Variational Regularization)
        # KL = sum(q * log(q/p))
        log_probs = F.log_softmax(logits, dim=-1)
        kl_div = (probs * (log_probs - torch.log(torch.tensor(1.0 / self.n_clusters, device=self.device)))).sum(dim=-1)
        
        # Gumbel-Softmax for differentiable sampling (reparameterization trick)
        z_soft = F.gumbel_softmax(logits, tau=temperature, hard=False)
        z_hard = F.gumbel_softmax(logits, tau=temperature, hard=True)
        
        # Regression
        reg_out = self.regressor(feat)
        mu, log_var = torch.chunk(reg_out, 2, dim=-1)
        
        # Flow transformation (Base noise -> Complex noise)
        noise_base = torch.randn_like(mu)
        noise_complex, log_det = self.flow(noise_base)
        
        return {
            "mu": mu,
            "log_var": log_var,
            "z_soft": z_soft,
            "z_hard": z_hard,
            "logits": logits,
            "probs": probs,
            "kl_loss": kl_div.mean(),
            "noise_complex": noise_complex,
            "log_det": log_det
        }

    def train_model(self, x, y, epochs=200, lr=1e-3, lda_clu=1.0):
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-2)
        self.train()
        
        if isinstance(x, np.ndarray): x = torch.from_numpy(x).float().to(self.device)
        if isinstance(y, np.ndarray): y = torch.from_numpy(y).float().to(self.device)
        
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.forward(bx)
                
                # 1. Regression Loss (NLL)
                loss_reg = F.gaussian_nll_loss(out['mu'], by, torch.exp(out['log_var']))
                
                # 2. Diversity Loss for Clustering (avoid collapsing to one cluster)
                avg_prob = out['z_soft'].mean(dim=0)
                loss_div = -torch.sum(avg_prob * torch.log(avg_prob + 1e-8))
                
                # 3. Flow Regularization (Complexity of noise)
                loss_flow = -out['log_det'].mean()
                
                loss = loss_reg - 0.1 * loss_div + 0.01 * loss_flow
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch+1) % 50 == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
            return out['mu'].cpu().numpy(), torch.exp(0.5*out['log_var']).cpu().numpy(), out['z_hard'].cpu().numpy()

if __name__ == '__main__':
    # Test high-end MLP
    X = np.random.randn(100, 2).astype(np.float32)
    Y = (np.sin(X[:, 0:1]) + 0.1 * np.random.randn(100, 1)).astype(np.float32)
    
    model = MLP(input_dim=2, hidden_dim=64, output_dim=1, n_clusters=3)
    model.train_model(X, Y, epochs=100)
    mu, std, clusters = model.predict(X)
    print("Clusters found:", clusters.argmax(axis=1))

