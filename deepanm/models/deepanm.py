"""
DeepANM - Deep Additive Noise Model for Causal Discovery
"""

import numpy as np
import torch
import torch.nn as nn


class DeepANM(nn.Module):
    """
    Deep Additive Noise Model for multivariate causal graph discovery.
    
    Core workflow:
        1. fit() or fit_bootstrap() to train on observational data
        2. get_dag_matrix() to extract the learned adjacency matrix
    """

    def __init__(self, x_dim=None, n_clusters=2, hidden_dim=64, lda=1.0, device=None):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.lda = lda
        self.x_dim = x_dim
        self.n_clusters = n_clusters
        self.hidden_dim = hidden_dim
        self.core = None   # built lazily on first fit()
        self.history = None

        if x_dim is not None:
            self._build_core(x_dim)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_core(self, x_dim):
        from deepanm.core.gppom_hsic import GPPOMC_lnhsic_Core
        self.x_dim = x_dim
        self.core = GPPOMC_lnhsic_Core(
            x_dim, 0, self.n_clusters, self.hidden_dim, self.lda, self.device
        )
        self.to(self.device)

    def _preprocess(self, X, apply_isolation=False, apply_quantile=False):
        """Optional sklearn preprocessing before fitting."""
        if apply_isolation:
            from sklearn.ensemble import IsolationForest
            mask = IsolationForest(contamination=0.05, random_state=42).fit_predict(X) == 1
            X = X[mask]
        if apply_quantile:
            from sklearn.preprocessing import QuantileTransformer
            X = QuantileTransformer(output_distribution='normal').fit_transform(X)
        return X

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, epochs=200, batch_size=64, lr=2e-3, verbose=True,
            apply_quantile=False, apply_isolation=False):
        """Train on raw data X (numpy array, shape [n_samples, n_vars])."""
        from deepanm.models.trainer import DeepANMTrainer

        X = self._preprocess(X, apply_isolation=apply_isolation, apply_quantile=apply_quantile)

        if self.core is None:
            self._build_core(X.shape[1])
        else:
            # Reset core so each fit() starts fresh
            self._build_core(X.shape[1])

        trainer = DeepANMTrainer(self, lr=lr)
        self.history = trainer.train(X, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return self.history

    def fit_bootstrap(self, X, n_bootstraps=5, threshold=0.015,
                      epochs=200, batch_size=64, lr=5e-3, verbose=True,
                      apply_quantile=False, apply_isolation=False):
        """
        Stability Selection via Bootstrap resampling.
        Returns (prob_matrix, avg_ATE_matrix) where prob_matrix[i,j]
        is the fraction of bootstrap runs that detected edge i->j.
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        X = self._preprocess(X, apply_isolation=apply_isolation, apply_quantile=apply_quantile)

        n_samples, n_vars = X.shape
        agg_W   = np.zeros((n_vars, n_vars))
        agg_bin = np.zeros((n_vars, n_vars))

        for b in range(n_bootstraps):
            if verbose:
                print(f"[Bootstrap] Round {b+1}/{n_bootstraps}...")

            # Resample with replacement
            boot_data = X[np.random.choice(n_samples, n_samples, replace=True)]

            # Fresh model each round to avoid prior bias
            self.core = None
            self.fit(boot_data, epochs=epochs, batch_size=batch_size, lr=lr, verbose=False)

            W_raw, W_bin = self.get_dag_matrix(threshold=threshold, X=boot_data)
            agg_W   += W_raw
            agg_bin += W_bin

        return agg_bin / n_bootstraps, agg_W / n_bootstraps

    def get_dag_matrix(self, threshold=0.1, X=None):
        """
        Extract the learned causal adjacency matrix.
        
        If X is provided, uses Neural ATE (Jacobian) to confirm edge strengths.
        Returns (W_raw, W_binary).
        """
        with torch.no_grad():
            W_dag_masked = self.core.W_dag * self.core.constraint_mask
            W = W_dag_masked.detach().cpu().numpy()

            if X is not None:
                ATE = self.core.MLP.get_global_ate_matrix(X, W_dag=W_dag_masked).cpu().numpy()
                W_bin = ((np.abs(W) > threshold) & (np.abs(ATE) > 1e-4)).astype(float)
                return ATE, W_bin
            else:
                return W, (np.abs(W) > threshold).astype(float)

    def set_exogenous(self, exog_indices):
        """
        Mark variables as exogenous (no incoming edges allowed).
        Must be called after fit() or after x_dim is known.
        """
        self.exog_indices = exog_indices
        if self.core is not None:
            for idx in exog_indices:
                self.core.constraint_mask[:, idx] = 0.0

    def predict_clusters(self, X):
        """Return the most likely mechanism cluster assignment for each sample."""
        self.eval()
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        with torch.no_grad():
            out = self.core.MLP(X.to(self.device))
            return out['z_soft'].argmax(dim=1).cpu().numpy()

    def forward(self, x, temperature=1.0):
        """Standard PyTorch forward — delegates to Core engine."""
        return self.core(x, temperature=temperature)
