"""
DeepANM - Deep Additive Noise Model for Causal Discovery
"""

import numpy as np
import torch
import torch.nn as nn


def _order_from_topo_mask(topo_mask: np.ndarray, n_vars: int) -> list:
    """
    Reconstruct the causal order (root-first list) from a stored triangular topo_mask.
    Uses topological sort (Kahn's algorithm) on the implied DAG.
    """
    # topo_mask[i, j] = 1 means i is an ancestor of j (edge i→j allowed)
    # Build adjacency: parents of each node
    in_deg = {j: 0 for j in range(n_vars)}
    children = {i: [] for i in range(n_vars)}
    for i in range(n_vars):
        for j in range(n_vars):
            if topo_mask[i, j] > 0.5:
                children[i].append(j)
                in_deg[j] += 1

    queue = [v for v in range(n_vars) if in_deg[v] == 0]
    order = []
    while queue:
        v = queue.pop(0)
        order.append(v)
        for u in children[v]:
            in_deg[u] -= 1
            if in_deg[u] == 0:
                queue.append(u)

    # If cycle detected (shouldn't happen), fall back to range
    if len(order) < n_vars:
        return list(range(n_vars))
    return order

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

    def _build_core(self, x_dim, X=None, verbose=True):
        from deepanm.core.gppom_hsic import GPPOMC_lnhsic_Core
        self.x_dim = x_dim

        # --- Phase 1: RESIT-inspired Topological Ordering ---
        causal_order = None
        if X is not None:
            from deepanm.core.toposort import hsic_greedy_order
            if verbose:
                print("[TopoSort] Estimating causal order via HSIC greedy search...")
            causal_order = hsic_greedy_order(X, verbose=False)
            if verbose:
                order_str = " → ".join(f"X{i}" for i in causal_order)
                print(f"[TopoSort] Discovered order: {order_str}")
                print(f"[TopoSort] Only edges consistent with this order will be learned.")

        # --- Phase 2: Build Neural Core with Topo Mask ---
        self.core = GPPOMC_lnhsic_Core(
            x_dim, 0, self.n_clusters, self.hidden_dim, self.lda,
            self.device, causal_order=causal_order
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
            self._build_core(X.shape[1], X=X, verbose=verbose)
        else:
            self._build_core(X.shape[1], X=X, verbose=verbose)

        trainer = DeepANMTrainer(self, lr=lr)
        self.history = trainer.train(X, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return self.history

    def fit_bootstrap(self, X, n_bootstraps=5,
                      epochs=200, batch_size=64, lr=5e-3, verbose=True,
                      apply_quantile=False, apply_isolation=False):
        """
        Stability Selection via Bootstrap resampling.

        Edge detection uses Adaptive LASSO per bootstrap round instead of
        a fixed threshold — results are statistically principled and robust.

        Returns (prob_matrix, avg_ATE_matrix) where prob_matrix[i,j]
        is the fraction of bootstrap runs that confirmed edge i→j.
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

            boot_data = X[np.random.choice(n_samples, n_samples, replace=True)]

            self.core = None
            self.fit(boot_data, epochs=epochs, batch_size=batch_size, lr=lr, verbose=False)

            # Adaptive LASSO edge selection (no fixed threshold)
            ATE, W_bin = self.get_dag_matrix(X=boot_data)
            agg_W   += ATE
            agg_bin += W_bin

        return agg_bin / n_bootstraps, agg_W / n_bootstraps

    def get_dag_matrix(self, X=None):
        """
        Extract the learned causal adjacency matrix.

        If X is provided, runs Adaptive LASSO edge selection using the
        Neural ATE Jacobian combined with the discovered causal order.
        This replaces the brittle fixed threshold with statistically
        principled model selection.

        Returns
        -------
        If X is provided: (ATE_matrix, W_binary_adaptive_lasso)
        If X is None:     (W_raw, W_binary_by_gate_prob)
        """
        with torch.no_grad():
            W_dag_masked = self.core.W_dag * self.core.topo_mask
            W = W_dag_masked.detach().cpu().numpy()

            if X is not None:
                # Neural ATE Jacobian
                ATE = self.core.MLP.get_global_ate_matrix(
                    X, W_dag=W_dag_masked).cpu().numpy()

                # Reconstruct causal order from topo_mask (triangular structure)
                causal_order = _order_from_topo_mask(
                    self.core.topo_mask.cpu().numpy(), self.x_dim)

                # Adaptive LASSO + ATE double gate
                from deepanm.utils.adaptive_lasso import adaptive_lasso_from_ate
                W_bin = adaptive_lasso_from_ate(ATE, X, causal_order)
                return ATE, W_bin
            else:
                # Fallback: return raw W only (no threshold applied)
                return W, (torch.sigmoid(self.core.W_logits).detach().cpu().numpy() > 0.5).astype(float) * self.core.topo_mask.cpu().numpy()

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
