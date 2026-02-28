"""
HSIC-based Greedy Causal Topological Ordering
Inspired by RESIT (Peters et al., 2014) and LiNGAM.

PERF OPTIMIZATIONS v2:
    [A] RFF-HSIC: O(n*D) approximation replaces O(n²) exact RBF Gram matrix
    [F] HistGradientBoostingRegressor: 10-50x faster than GradientBoostingRegressor
        for n > 100 (uses histogram binning instead of sorted enumeration)
"""

import numpy as np


# ---------------------------------------------------------------------------
# [Fix A] RFF-approximated HSIC  — O(n * D) instead of O(n²)
# ---------------------------------------------------------------------------

def _rff_features(Z: np.ndarray, D: int, bw: float, rng: np.random.RandomState) -> np.ndarray:
    """Random Fourier Features for RBF kernel approximation."""
    d = Z.shape[1]
    W = rng.randn(d, D) / max(bw, 1e-6)
    b = rng.uniform(0, 2 * np.pi, D)
    phi = np.sqrt(2.0 / D) * np.cos(Z @ W + b)
    phi -= phi.mean(axis=0)   # center ~ kernel centering
    return phi


def _bw_estimate(Z: np.ndarray) -> float:
    """Fast bandwidth via std of pairwise distances (much cheaper than full median)."""
    sub = Z[:200] if Z.shape[0] > 200 else Z
    dists = np.sqrt(np.sum((sub[:, None] - sub[None, :]) ** 2, axis=-1))
    upper = dists[np.triu_indices(sub.shape[0], k=1)]
    med = np.median(upper) if upper.size > 0 else 1.0
    return max(med, 1e-6)


def _rff_hsic(X: np.ndarray, Y: np.ndarray, D: int = 64, seed: int = 0) -> float:
    """
    RFF-approximated HSIC statistic. O(n*D) time, O(n*D) memory.
    Much faster than exact O(n²) Gram matrix for n > 200.
    """
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    n = X.shape[0]
    if n < 4:
        return 0.0

    rng = np.random.RandomState(seed)
    bw_x = _bw_estimate(X)
    bw_y = _bw_estimate(Y)

    phi_x = _rff_features(X, D, bw_x, rng)   # (n, D)
    phi_y = _rff_features(Y, D, bw_y, rng)   # (n, D)

    # HSIC ≈ ||phi_x.T @ phi_y||_F^2 / n^2
    C = (phi_x.T @ phi_y) / n                 # (D, D)
    return float(np.sum(C ** 2))


# ---------------------------------------------------------------------------
# [Fix F] Fast residual regressor using HistGradientBoosting
# ---------------------------------------------------------------------------

def _fit_residuals(X_others: np.ndarray, Xk: np.ndarray) -> np.ndarray:
    """
    Fit Xk ~ f(X_others) with HistGradientBoostingRegressor.
    10-50x faster than GradientBoostingRegressor for n > 100
    (uses histogram binning instead of sorted feature enumeration).
    """
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        reg = HistGradientBoostingRegressor(
            max_iter=100,         # Equivalent to n_estimators=100
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            early_stopping=False  # Skip validation set overhead
        )
        reg.fit(X_others, Xk)
        return Xk - reg.predict(X_others)
    except Exception:
        # Fallback: linear residuals (no sklearn dependency)
        w = np.linalg.lstsq(X_others, Xk, rcond=None)[0]
        return Xk - X_others @ w


# ---------------------------------------------------------------------------
# Main: Sink-First HSIC greedy topological sort
# ---------------------------------------------------------------------------

def hsic_greedy_order(X: np.ndarray, n_rff: int = 128, verbose: bool = False) -> list:
    """
    Estimate causal topological order using Sink-First HSIC (Bottom-Up RESIT).

    [Fix A]: Uses RFF-approximated HSIC (O(n*D)) instead of exact O(n²) kernel.

    Strategy (Bottom-Up):
        While |S| > 1:
            For each candidate sink k in S:
                For each other variable i in S:
                    Residual r_i = Xi - linear_fit(Xi ~ Xk).
                    Score += RFF-HSIC(r_i, Xk).
            Pick k* = argmin Score → most likely leaf/sink.
            Record k* as next (from the leaf end). Remove from S.
        Reverse to get root-first order.

    Parameters
    ----------
    X      : (n_samples, n_vars) array
    n_rff  : number of Random Fourier Features (D). Default 64.
    verbose: print per-step scores.

    Returns
    -------
    causal_order : list of int, root first (root = most exogenous).
    """
    n_samples, n_vars = X.shape

    if n_vars == 1:
        return [0]

    # Standardize
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    X_scaled = (X - X.mean(axis=0)) / std

    remaining = list(range(n_vars))
    reverse_order = []   # leaf → root, reversed at end

    step = 0
    while len(remaining) > 1:
        sink_scores = []

        for k in remaining:
            Xk = X_scaled[:, k]
            Xk_2d = Xk.reshape(-1, 1)
            var_k = float(np.var(Xk)) + 1e-8

            total = 0.0
            for i in [j for j in remaining if j != k]:
                Xi = X_scaled[:, i]
                # Simple linear residual of Xi on Xk (O(n) — intentionally cheap here)
                w = float(np.dot(Xi, Xk) / (n_samples * var_k))
                res_i = (Xi - w * Xk).reshape(-1, 1)
                # [Fix A]: RFF-HSIC instead of exact RBF Gram
                total += _rff_hsic(res_i, Xk_2d, D=n_rff, seed=step)
            sink_scores.append(total)

        sink_idx = int(np.argmin(sink_scores))
        sink_var = remaining[sink_idx]

        if verbose:
            scores_str = ", ".join(
                f"X{remaining[i]}={sink_scores[i]:.4f}" for i in range(len(remaining))
            )
            print(f"  [TopoSort] Remaining={remaining}  SinkScore=[{scores_str}]  → Sink=X{sink_var}")

        reverse_order.append(sink_var)
        remaining.pop(sink_idx)
        step += 1

    reverse_order.append(remaining[0])   # last = root
    return list(reversed(reverse_order))


def build_topo_mask(causal_order: list, n_vars: int) -> np.ndarray:
    """
    Build a strict topological mask from a causal order.
    mask[i, j] = 1  iff  position(i) < position(j)  (edge i→j is allowed).
    """
    position = {var: idx for idx, var in enumerate(causal_order)}
    mask = np.zeros((n_vars, n_vars), dtype=np.float32)
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j and position[i] < position[j]:
                mask[i, j] = 1.0
    return mask
