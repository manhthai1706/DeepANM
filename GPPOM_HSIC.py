# -*- coding: utf-8 -*-
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
    Fast HSIC using RFF approximation (O(N) complexity)
    HSIC(X, Z) = ||Cov(Phi(X), Psi(Z))||^2
    """
    def __init__(self, x_dim, z_dim, n_features=128):
        super().__init__()
        self.phi_x = RFFGPLayer(x_dim, n_features)
        self.phi_z = RFFGPLayer(z_dim, n_features)

    def forward(self, X, Z):
        n = X.shape[0]
        if n < 2: return torch.tensor(0.0, device=X.device)
        
        # Project to RFF space
        feat_x = self.phi_x(X)
        feat_z = self.phi_z(Z)
        
        # Center features
        feat_x = feat_x - feat_x.mean(dim=0, keepdim=True)
        feat_z = feat_z - feat_z.mean(dim=0, keepdim=True)
        
        # HSIC is the Frobenius norm of cross-covariance
        covariance = (feat_x.T @ feat_z) / (n - 1)
        return torch.sum(covariance**2)

class CausalFlow(nn.Module):
    """
    Unified CausalFlow Architecture
    Provides a clean ML-style interface: fit(), predict(), save()
    """
    def __init__(self, x_dim, y_dim, n_clusters=2, hidden_dim=64, lda=1.0, device=None):
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.lda = lda
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_clusters = n_clusters
        
        # Internal Model Components
        self.model = GPPOMC_lnhsic_Core(x_dim, y_dim, n_clusters, hidden_dim, lda, self.device)
        self.history = {"loss": [], "hsic": [], "gp": []}
        
    def fit(self, X, Y, epochs=200, batch_size=64, lr=2e-3):
        """Standard ML fitting method"""
        print(f"Training CausalFlow on {self.device}...")
        self.model.train()
        
        # Convert to tensors if numpy
        if isinstance(X, np.ndarray): X = torch.from_numpy(X).float()
        if isinstance(Y, np.ndarray): Y = torch.from_numpy(Y).float()
        
        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        
        best_loss = float('inf')
        for epoch in range(epochs):
            epoch_loss, epoch_hsic, epoch_gp = 0, 0, 0
            temp = max(0.5, 1.0 - epoch / epochs)
            
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                loss, l_gp, l_hsic = self.model(bx, by, temperature=temp)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_hsic += l_hsic.item()
                epoch_gp += l_gp.item()
            
            avg_loss = epoch_loss / len(loader)
            self.history["loss"].append(avg_loss)
            self.history["hsic"].append(epoch_hsic / len(loader))
            self.history["gp"].append(epoch_gp / len(loader))
            
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | HSIC: {self.history['hsic'][-1]:.6f}")

    def predict_clusters(self, X, Y):
        """Predict mechanism labels for given data"""
        self.model.eval()
        if isinstance(X, np.ndarray): X = torch.from_numpy(X).float()
        if isinstance(Y, np.ndarray): Y = torch.from_numpy(Y).float()
        
        with torch.no_grad():
            bx, by = X.to(self.device), Y.to(self.device)
            xy = torch.cat([bx, by], dim=1)
            out = self.model.MLP(xy)
            return out['z_hard'].argmax(axis=1).cpu().numpy()

    def save(self, path="causalflow_model.safetensors"):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path="causalflow_model.safetensors"):
        self.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")

class GPPOMC_lnhsic_Core(nn.Module):
    """Internal Core Architecture (Renamed from GPPOMC_lnhsic)"""
    def __init__(self, x_dim, y_dim, n_clusters, hidden_dim, lda, device):
        super().__init__()
        self.lda = lda
        self.device = device
        
        self.MLP = MLP.MLP(input_dim=x_dim+y_dim, hidden_dim=hidden_dim, 
                          output_dim=y_dim, n_clusters=n_clusters, device=device)
        
        self.gp_phi_z = RFFGPLayer(n_clusters, n_features=128)
        self.gp_phi_x = RFFGPLayer(x_dim, n_features=128)
        self.linear_head = nn.Linear(128, y_dim, bias=False)
        self.log_beta = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.fast_hsic = FastHSIC(x_dim, n_clusters, n_features=128)

    def forward(self, batch_x, batch_y, temperature=1.0):
        xy = torch.cat([batch_x, batch_y], dim=1)
        out = self.MLP(xy, temperature=temperature)
        z_soft, kl_loss = out['z_soft'], out['kl_loss']
        
        phi = self.gp_phi_z(z_soft) * self.gp_phi_x(batch_x)
        y_pred = self.linear_head(phi)
        
        beta = torch.exp(self.log_beta)
        loss_likelihood = 0.5 * beta * torch.sum((batch_y - y_pred)**2)
        loss_ridge = 0.5 * torch.sum(self.linear_head.weight**2)
        loss_hsic = self.fast_hsic(batch_x, z_soft)
        
        total_loss = loss_likelihood + loss_ridge + self.lda * torch.log(loss_hsic + 1e-8) + 0.1 * kl_loss
        return total_loss, loss_likelihood, loss_hsic
