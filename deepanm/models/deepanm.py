"""
DeepANM - Model Architecture
Deep Additive Noise Model for Causal Discovery
"""

import numpy as np
import torch
import torch.nn as nn


class DeepANM(nn.Module):
    """
    DeepANM (Deep Additive Noise Model)
    """
    def __init__(self, x_dim=None, y_dim=0, n_clusters=2, hidden_dim=64, lda=1.0, device=None, data=None, **kwargs):
        """
        Initialize DeepANM model.
        If data is provided, it automatically infers dimensions and trains.
        """
        super().__init__()
        
        # If data provided, infer x_dim
        if data is not None and x_dim is None:
            x_dim = data.shape[1]
            
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.lda = lda
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_clusters = n_clusters
        self.hidden_dim = hidden_dim
        self.d = x_dim + y_dim if x_dim is not None else 0
        
        # Import here to avoid circular dependency
        from deepanm.core.gppom_hsic import GPPOMC_lnhsic_Core
        
        # Core model architecture
        if x_dim is not None:
            self.core = GPPOMC_lnhsic_Core(x_dim, y_dim, n_clusters, hidden_dim, lda, self.device)
        else:
            self.core = None # Will be initialized during first fit
            
        self.history = None
        self.trainer = None
        self.to(self.device)
        
        # Auto-train if data provided
        if data is not None:
            self.fit(data, **kwargs)

    def __call__(self, X=None, Y=None, **kwargs):
        """
        Enables model(data, train=True) pattern.
        """
        if kwargs.get('train', False) or (X is not None and not torch.is_tensor(X) and not isinstance(X, torch.nn.Parameter)):
            # If input is not a tensor and we're not in eval mode, assume training intent
            kwargs.pop('train', None)
            return self.fit(X, Y, **kwargs)
        
        # Standard PyTorch forward behavior
        return super().__call__(X, **kwargs)
        
    def fit(self, X, Y=None, epochs=200, batch_size=64, lr=2e-3, verbose=True):
        """
        Train the model. Supports both bivariate (X, Y) and multivariate (X) data.
        """
        from deepanm.models.trainer import DeepANMTrainer
        
        # Combine if bivariate
        if Y is not None:
            if isinstance(X, np.ndarray):
                X_all = np.hstack([X, Y])
            else:
                X_all = torch.cat([X, Y], dim=1)
        else:
            X_all = X
            
        # Dynamic re-init if self.core is None
        if self.core is None:
            from deepanm.core.gppom_hsic import GPPOMC_lnhsic_Core
            self.x_dim = X_all.shape[1]
            self.d = self.x_dim
            self.core = GPPOMC_lnhsic_Core(self.x_dim, self.y_dim, self.n_clusters, 
                                          self.hidden_dim, self.lda, self.device)
            self.to(self.device)

        self.trainer = DeepANMTrainer(self, lr=lr)
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
        """Extracts residuals from the trained model"""
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

    def check_stability(self, X, n_splits=3):
        """Checks if the learned mechanism remains stable across different data segments"""
        if isinstance(X, np.ndarray): X = torch.from_numpy(X).float()
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        splits = np.array_split(indices, n_splits)
        
        losses = []
        self.eval()
        with torch.no_grad():
            for split in splits:
                batch_x = X[split].to(self.device)
                total_loss, _, _ = self.core(batch_x)
                losses.append(total_loss.item())
        
        stability = np.std(losses) / (np.abs(np.mean(losses)) + 1e-8)
        return stability, losses

    def predict_counterfactual(self, x_orig, y_orig, x_new):
        """
        Computes counterfactual for bivariate case Y|X using the GP-head discovery.
        Y_cf = Y_obs - Prediction(X_orig) + Prediction(X_new)
        """
        self.eval()
        with torch.no_grad():
            def get_y_pred(x_val, y_val_for_z):
                xt = torch.tensor([[x_val]]).float().to(self.device)
                yt = torch.tensor([[y_val_for_z]]).float().to(self.device)
                xy = torch.cat([xt, yt], dim=1)
                
                # Get mechanism (z) from MLP
                out = self.core.MLP(xy)
                z = out['z_soft']
                
                # Get prediction from GP head (assuming bivariate X->Y)
                phi = self.core.gp_phi_z(z) * self.core.gp_phi_x(xt)
                return self.core.linear_head(phi).item()

            y_pred_orig = get_y_pred(x_orig, y_orig)
            y_pred_new = get_y_pred(x_new, y_orig)
            
            y_cf = y_orig - y_pred_orig + y_pred_new
            return y_cf

    def predict_direction(self, data=None, lda=None):
        """
        Predicts causal direction for bivariate data.
        Returns 1 (X->Y) or -1 (Y->X).
        """
        if data is None:
            # If no data provided, try to infer from learned DAG
            W, _ = self.get_dag_matrix()
            return 1 if W[0, 1] > W[1, 0] else -1
            
        # If data provided, use the high-accuracy hypothesis testing logic
        if lda is None: lda = self.lda
        from deepanm.models.analysis import ANMMM_cd
        direction, _ = ANMMM_cd(data, lda=lda)
        return direction

    @staticmethod
    def infer_bivariate_direction(data, lda=12.0, epochs=200):
        """
        Static helper for Causal Direction Inference.
        """
        from deepanm.models.analysis import ANMMM_cd
        return ANMMM_cd(data, lda=lda)


# Backward compatibility alias
CausalFlow = DeepANM
