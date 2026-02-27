"""
HSIC-based Greedy Causal Topological Ordering
Inspired by RESIT (Peters et al., 2014) and LiNGAM.

Core idea: The "root" variable in a causal DAG is the one whose residuals,
after regressing on ALL other variables, are most INDEPENDENT from those others.
We greedily peel off roots one by one to build the full causal order.

This runs entirely in NumPy BEFORE neural network training, so it is fast and
constraint-free. The resulting causal order is then used to build a strict
triangular mask that forbids reverse-direction edges inside the neural net.
"""

import numpy as np
from scipy.stats import gamma as scipy_gamma


# ---------------------------------------------------------------------------
# Gaussian (RBF) Kernel HSIC — exact, O(N^2), used on small subsets
# ---------------------------------------------------------------------------

def _rbf_gram(X: np.ndarray, width: float) -> np.ndarray:
    """Centered RBF Gram matrix."""
    n = X.shape[0]
    G = np.sum(X ** 2, axis=1)
    dists = G[:, None] + G[None, :] - 2.0 * X @ X.T
    K = np.exp(-dists / (2.0 * width ** 2))
    # Center
    row = K.mean(axis=1, keepdims=True)
    col = K.mean(axis=0, keepdims=True)
    total = K.mean()
    return K - row - col + total


def _median_bandwidth(X: np.ndarray) -> float:
    """Median-heuristic bandwidth (capped at 100 samples for speed)."""
    X_sub = X[:100] if X.shape[0] > 100 else X
    n = X_sub.shape[0]
    G = np.sum(X_sub ** 2, axis=1)
    dists = G[:, None] + G[None, :] - 2.0 * X_sub @ X_sub.T
    upper = dists[np.triu_indices(n, k=1)]
    med = np.median(upper[upper > 0]) if upper[upper > 0].size > 0 else 1.0
    return max(np.sqrt(0.5 * med), 1e-6)


def _hsic_statistic(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute the HSIC test statistic between X and Y.
    Returns the raw statistic (larger = more dependent).
    """
    X = X.reshape(len(X), -1)
    Y = Y.reshape(len(Y), -1)
    n = X.shape[0]
    if n < 4:
        return 0.0

    wx = _median_bandwidth(X)
    wy = _median_bandwidth(Y)
    Kc = _rbf_gram(X, wx)
    Lc = _rbf_gram(Y, wy)
    return float(np.sum(Kc * Lc) / n)


# ---------------------------------------------------------------------------
# Simple nonlinear regressor (Gradient Boosted Trees) for residual computation
# ---------------------------------------------------------------------------

def _fit_residuals(X_others: np.ndarray, Xk: np.ndarray) -> np.ndarray:
    """
    Fit Xk ~ f(X_others) using a high-capacity nonlinear regressor and return residuals.
    Uses 200 GBM trees with depth 4 to handle Cubic, Sin, Exp compositions.
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        reg = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                        learning_rate=0.05, random_state=42,
                                        subsample=0.8)
        reg.fit(X_others, Xk)
        return Xk - reg.predict(X_others)
    except Exception:
        # Fallback: linear residuals
        w = np.linalg.lstsq(X_others, Xk, rcond=None)[0]
        return Xk - X_others @ w


# ---------------------------------------------------------------------------
# Main: HSIC-based greedy topological sort
# ---------------------------------------------------------------------------

def hsic_greedy_order(X: np.ndarray, verbose: bool = False) -> list:
    """
    Estimate causal topological order using Sink-First HSIC (Bottom-Up RESIT).

    Strategy (Bottom-Up, more numerically robust):
        While |S| > 1:
            For each candidate sink k in S:
                For each other variable i in S:
                    Regress Xi on Xk.
                    Compute residual r_i.
                    Measure HSIC(r_i, Xk).
                Score(k) = sum of HSIC(r_i, Xk) for all i != k.
            Pick k* = argmin Score(k). -> Xk* is the SINK (leaf node).
            Prepend k* to reverse_order. Remove k* from S.

    Returns root-to-leaf causal order (root first).
    """
    n_samples, n_vars = X.shape

    if n_vars == 1:
        return [0]

    # Standardize
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    X_scaled = (X - X.mean(axis=0)) / std

    remaining = list(range(n_vars))
    reverse_order = []  # Will hold leaf → root, reversed at the end

    while len(remaining) > 1:
        sink_scores = []

        for k in remaining:
            others = [i for i in remaining if i != k]
            Xk = X_scaled[:, k].reshape(-1, 1)  # Candidate sink
            
            # For each other variable, regress it ON Xk, measure HSIC(residual, Xk)
            total_hsic = 0.0
            for i in others:
                Xi = X_scaled[:, i]
                # Simple linear regression of Xi on Xk (fast)
                w = float(np.cov(Xi, X_scaled[:, k])[0, 1] / (np.var(X_scaled[:, k]) + 1e-8))
                residual_i = Xi - w * X_scaled[:, k]
                total_hsic += _hsic_statistic(residual_i.reshape(-1, 1), Xk)
            
            sink_scores.append(total_hsic)

        # Smallest total HSIC -> this variable is most likely a sink (leaf)
        sink_idx = int(np.argmin(sink_scores))
        sink_var = remaining[sink_idx]

        if verbose:
            scores_str = ", ".join(
                f"X{remaining[i]}={sink_scores[i]:.4f}" for i in range(len(remaining))
            )
            print(f"  [TopoSort] Remaining={remaining}  SinkScore=[{scores_str}]  → Sink=X{sink_var}")

        reverse_order.append(sink_var)
        remaining.pop(sink_idx)

    reverse_order.append(remaining[0])  # Last one is the root
    # Flip: root first, leaf last
    causal_order = list(reversed(reverse_order))
    return causal_order


def build_topo_mask(causal_order: list, n_vars: int) -> np.ndarray:
    """
    Build a strict topological mask from a causal order.

    mask[i, j] = 1  iff  causal_order.index(i) < causal_order.index(j)
                   i.e., i is an ancestor of j (edge i→j is allowed)
    mask[i, j] = 0  otherwise (forbidden: would create a cycle or reverse edge)

    Returns
    -------
    mask : np.ndarray, shape (n_vars, n_vars), dtype float32
    """
    position = {var: idx for idx, var in enumerate(causal_order)}
    mask = np.zeros((n_vars, n_vars), dtype=np.float32)
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j and position[i] < position[j]:
                mask[i, j] = 1.0
    return mask
