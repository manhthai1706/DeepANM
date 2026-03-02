"""
HSIC-based Greedy Causal Topological Ordering
Inspired by RESIT (Peters et al., 2014) and LiNGAM.

Uses RFF-HSIC O(n*D) approximation and nonlinear residuals (HistGBM)
for robust ordering on skewed, heavy-tailed, and nonlinear data.
"""

import numpy as np


# ---------------------------------------------------------------------------
# RFF-approximated HSIC — O(n * D) instead of O(n²)
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
    """Fast bandwidth via median of pairwise distances on subsample."""
    sub = Z[:200] if Z.shape[0] > 200 else Z
    dists = np.sqrt(np.sum((sub[:, None] - sub[None, :]) ** 2, axis=-1))
    upper = dists[np.triu_indices(sub.shape[0], k=1)]
    med = np.median(upper) if upper.size > 0 else 1.0
    return max(med, 1e-6)


def _rff_hsic(X: np.ndarray, Y: np.ndarray, D: int = 128, seed: int = 0) -> float:
    """RFF-approximated HSIC statistic. O(n*D) time."""
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    n = X.shape[0]
    if n < 4:
        return 0.0

    rng = np.random.RandomState(seed)
    bw_x = _bw_estimate(X)
    bw_y = _bw_estimate(Y)

    phi_x = _rff_features(X, D, bw_x, rng)
    phi_y = _rff_features(Y, D, bw_y, rng)

    C = (phi_x.T @ phi_y) / n
    return float(np.sum(C ** 2))


# ---------------------------------------------------------------------------
# Nonlinear residual computation via HistGradientBoosting
# ---------------------------------------------------------------------------

def _nonlinear_residual(X_cause: np.ndarray, Xi: np.ndarray) -> np.ndarray:
    """
    Compute residual r_i = Xi - f(X_cause) using HistGradientBoostingRegressor.
    Handles nonlinear and heavy-tailed relationships much better than OLS.
    Falls back to linear regression if sklearn is unavailable.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        reg = HistGradientBoostingRegressor(
            max_iter=50, max_depth=3, learning_rate=0.1,
            random_state=42, early_stopping=False
        )
        reg.fit(X_cause, Xi)
        return Xi - reg.predict(X_cause)
    except Exception:
        # Fallback: linear regression
        w = np.linalg.lstsq(X_cause, Xi, rcond=None)[0]
        return Xi - X_cause @ w


# ---------------------------------------------------------------------------
# Main: Sink-First HSIC greedy topological sort
# ---------------------------------------------------------------------------

def hsic_greedy_order(X: np.ndarray, n_rff: int = 128, verbose: bool = False) -> list:
    """
    Estimate causal topological order using Sink-First HSIC (Bottom-Up RESIT).

    Uses:
      - RFF-approximated HSIC (O(n*D)) for independence testing
      - QuantileTransform to handle skewed/heavy-tailed data
      - Nonlinear residuals (HistGBM) for inner-loop regression when d_remaining <= 6

    Strategy (Bottom-Up):
        While |S| > 1:
            For each candidate sink k in S:
                For each other variable i in S:
                    Residual r_i = Xi - f(Xk).
                    Score += RFF-HSIC(r_i, Xk).
            Pick k* = argmin Score -> most likely leaf/sink.
            Record k* as next (from the leaf end). Remove from S.
        Reverse to get root-first order.

    Parameters
    ----------
    X      : (n_samples, n_vars) array
    n_rff  : number of Random Fourier Features. Default 128.
    verbose: print per-step scores.

    Returns
    -------
    causal_order : list of int, root first (root = most exogenous).
    """
    n_samples, n_vars = X.shape

    if n_vars == 1:
        return [0]

    # Simple standardization (idempotent if data is already scaled).
    # QuantileTransform is handled by DeepANM._preprocess() before calling this.
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    X_scaled = (X - X.mean(axis=0)) / std

    remaining = list(range(n_vars))
    reverse_order = []

    step = 0
    while len(remaining) > 1:
        sink_scores = []
        # Use nonlinear residuals when few variables remain (affordable)
        use_nonlinear = len(remaining) <= 6

        for k in remaining:
            Xk = X_scaled[:, k]
            Xk_2d = Xk.reshape(-1, 1)

            total = 0.0
            for i in [j for j in remaining if j != k]:
                Xi = X_scaled[:, i]

                if use_nonlinear:
                    # HistGBM nonlinear residual
                    res_i = _nonlinear_residual(Xk_2d, Xi).reshape(-1, 1)
                else:
                    # Fast linear residual (O(n))
                    var_k = float(np.var(Xk)) + 1e-8
                    w = float(np.dot(Xi, Xk) / (n_samples * var_k))
                    res_i = (Xi - w * Xk).reshape(-1, 1)

                total += _rff_hsic(res_i, Xk_2d, D=n_rff, seed=step)
            sink_scores.append(total)

        sink_idx = int(np.argmin(sink_scores))
        sink_var = remaining[sink_idx]

        if verbose:
            scores_str = ", ".join(
                f"X{remaining[i]}={sink_scores[i]:.4f}" for i in range(len(remaining))
            )
            print(f"  [TopoSort] Remaining={remaining}  SinkScore=[{scores_str}]  -> Sink=X{sink_var}")

        reverse_order.append(sink_var)
        remaining.pop(sink_idx)
        step += 1

    reverse_order.append(remaining[0])
    return list(reversed(reverse_order))
