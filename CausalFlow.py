"""
CausalFlow - Model Architecture
Standard PyTorch model class following best practices
"""

import numpy as np
import torch
import torch.nn as nn


class CausalFlow(nn.Module):
    """
    CausalFlow Model Architecture - SOTA Upgrade
    """
    def __init__(self, x_dim, y_dim=0, n_clusters=2, hidden_dim=64, lda=1.0, device=None):
        """
        Initialize CausalFlow model
        If y_dim=0, it treats x_dim as the total number of variables in multivariate mode.
        """
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.lda = lda
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_clusters = n_clusters
        self.hidden_dim = hidden_dim
        self.d = x_dim + y_dim
        
        # Import here to avoid circular dependency
        from GPPOM_HSIC import GPPOMC_lnhsic_Core
        
        # Core model architecture (Now multivariate aware)
        self.core = GPPOMC_lnhsic_Core(x_dim, y_dim, n_clusters, hidden_dim, lda, self.device)
        
        self.history = None
        self.trainer = None
        self.to(self.device)
        
    def fit(self, X, Y=None, epochs=200, batch_size=64, lr=2e-3, verbose=True):
        """
        Train the model. Supports both bivariate (X, Y) and multivariate (X) data.
        """
        from train import CausalFlowTrainer
        
        # Combine if bivariate
        if Y is not None:
            if isinstance(X, np.ndarray):
                X_all = np.hstack([X, Y])
            else:
                X_all = torch.cat([X, Y], dim=1)
        else:
            X_all = X
            
        self.trainer = CausalFlowTrainer(self, lr=lr)
        self.history = self.trainer.train(X_all, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return self.history
        
    def get_dag_matrix(self, threshold=0.1):
        """Returns the learned adjacency matrix W"""
        with torch.no_grad():
            W = self.core.W_dag.detach().cpu().numpy()
            W_bin = (np.abs(W) > threshold).astype(float)
            return W, W_bin

    def forward(self, x, temperature=1.0):
        return self.core(x, temperature=temperature)
    
    def predict_clusters(self, X):
        self.eval()
        if isinstance(X, np.ndarray): X = torch.from_numpy(X).float()
        with torch.no_grad():
            X = X.to(self.device)
            out = self.core.MLP(X)
            return out['z_soft'].argmax(dim=1).cpu().numpy()

    def get_residuals(self, X, use_pnl=True):
        self.eval()
        if isinstance(X, np.ndarray): X = torch.from_numpy(X).float()
        with torch.no_grad():
            X = X.to(self.device)
            out = self.core.MLP(X)
            z = out['z_soft']
            masked_input = X @ torch.abs(self.core.W_dag)
            phi = self.core.gp_phi_z(z) * self.core.gp_phi_x(masked_input)
            y_pred = self.core.linear_head(phi)
            if use_pnl:
                residuals = self.core.MLP.pnl_transform(X) - y_pred
            else:
                residuals = X - y_pred
            return residuals.cpu().numpy()

