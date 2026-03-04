"""
HSIC-based Greedy Causal Topological Ordering (TopoSort).
Inspired by RESIT (Peters et al., 2014) and LiNGAM.

Uses pairwise ANM asymmetry with RFF-HSIC (O(N·D)) and nonlinear residuals
to robustly order variables in skewed, heavy-tailed, and nonlinear settings.

Sắp xếp Topo Nhân quả Tham lam dựa trên HSIC.
Lấy cảm hứng từ RESIT (Peters et al., 2014) và LiNGAM.

Dùng tính bất đối xứng ANM từng cặp với RFF-HSIC (O(N·D)) và phần dư phi tuyến
để sắp xếp biến bền vững trên dữ liệu lệch, đuôi nặng và phi tuyến.
"""

import numpy as np


# ---------------------------------------------------------------------------
# RFF-approximated HSIC — O(N·D) instead of O(N²)
# HSIC xấp xỉ qua RFF — O(N·D) thay vì O(N²)
# ---------------------------------------------------------------------------

def _rff_features(Z: np.ndarray, D: int, bw: float, rng: np.random.RandomState) -> np.ndarray:
    """Random Fourier Features for RBF kernel approximation.
    Đặc trưng Fourier Ngẫu nhiên để xấp xỉ nhân RBF."""
    d = Z.shape[1]
    W = rng.randn(d, D) / max(bw, 1e-6)
    b = rng.uniform(0, 2 * np.pi, D)
    phi = np.sqrt(2.0 / D) * np.cos(Z @ W + b)
    phi -= phi.mean(axis=0)  # Center ≈ kernel centering / Chuẩn hóa tâm
    return phi


def _bw_estimate(Z: np.ndarray) -> float:
    """Bandwidth via median of pairwise distances on a subsample.
    Ước lượng bandwidth qua trung vị khoảng cách cặp trên mẫu con."""
    sub = Z[:200] if Z.shape[0] > 200 else Z
    dists = np.sqrt(np.sum((sub[:, None] - sub[None, :]) ** 2, axis=-1))
    upper = dists[np.triu_indices(sub.shape[0], k=1)]
    med = np.median(upper) if upper.size > 0 else 1.0
    return max(med, 1e-6)


def _rff_hsic(X: np.ndarray, Y: np.ndarray, D: int = 128, seed: int = 0) -> float:
    """RFF-approximated HSIC statistic, O(N·D) time.
    Thống kê HSIC xấp xỉ qua RFF, độ phức tạp O(N·D)."""
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    n = X.shape[0]
    if n < 4:
        return 0.0

    rng = np.random.RandomState(seed)
    phi_x = _rff_features(X, D, _bw_estimate(X), rng)
    phi_y = _rff_features(Y, D, _bw_estimate(Y), rng)

    C = (phi_x.T @ phi_y) / n
    return float(np.sum(C ** 2))


# ---------------------------------------------------------------------------
# Nonlinear residual via HistGradientBoosting
# Phần dư phi tuyến qua HistGradientBoosting
# ---------------------------------------------------------------------------

def _nonlinear_residual(X_cause: np.ndarray, Xi: np.ndarray) -> np.ndarray:
    """
    Compute residual r = Xi - f(X_cause) using HistGradientBoostingRegressor.
    Falls back to linear OLS if sklearn is unavailable.

    Tính phần dư r = Xi - f(X_cause) dùng HistGradientBoostingRegressor.
    Dự phòng OLS tuyến tính nếu sklearn không có sẵn.
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
        # Linear fallback / Dự phòng tuyến tính
        w = np.linalg.lstsq(X_cause, Xi, rcond=None)[0]
        return Xi - X_cause @ w


# ---------------------------------------------------------------------------
# Main: Sink-First HSIC greedy topological sort
# Chính: Sắp xếp topo tham lam Sink-First theo HSIC
# ---------------------------------------------------------------------------

def hsic_greedy_order(X: np.ndarray, n_rff: int = 128, verbose: bool = False) -> list:
    """
    Estimate the causal topological order using Pairwise ANM Asymmetry.

    For each variable pair (i, j):
        forward_score  = HSIC(Xj - f(Xi), Xi)   ← residual of j given i
        backward_score = HSIC(Xi - g(Xj), Xj)   ← residual of i given j
        M[i, j] = backward_score - forward_score

    If M[i, j] > 0, the direction i → j is favored.
    RootScore_i = Σ_j M[i, j]: higher score → more likely to be a root.
    Variables are sorted by RootScore descending to form the causal order.

    Ước lượng thứ tự nhân quả dùng Tính Bất đối xứng ANM từng cặp.

    Với mỗi cặp (i, j):
        forward_score  = HSIC(Xj - f(Xi), Xi)   ← phần dư của j theo i
        backward_score = HSIC(Xi - g(Xj), Xj)   ← phần dư của i theo j
        M[i, j] = backward_score - forward_score

    M[i, j] > 0 → chiều i → j được ưu tiên.
    RootScore_i = Σ_j M[i, j]: điểm cao hơn → có khả năng là nguồn gốc hơn.
    Biến được sắp xếp giảm dần theo RootScore để tạo thứ tự nhân quả.
    """
    n_samples, n_vars = X.shape

    if n_vars == 1:
        return [0]

    # Standardize for fair comparison / Chuẩn hóa để so sánh công bằng
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    X_scaled = (X - X.mean(axis=0)) / std

    M = np.zeros((n_vars, n_vars))
    use_nonlinear = n_vars <= 20  # Linear fallback for large graphs / Dự phòng tuyến tính cho đồ thị lớn
    if verbose:
        print("[TopoSort] Computing Pairwise Causal Asymmetry matrix...")

    for i in range(n_vars):
        Xi = X_scaled[:, i]
        Xi_2d = Xi.reshape(-1, 1)
        for j in range(i + 1, n_vars):
            Xj = X_scaled[:, j]
            Xj_2d = Xj.reshape(-1, 1)

            if use_nonlinear:
                res_j_given_i = _nonlinear_residual(Xi_2d, Xj)
                res_i_given_j = _nonlinear_residual(Xj_2d, Xi)
            else:
                # Fast linear OLS / OLS tuyến tính nhanh
                var_i = float(np.var(Xi)) + 1e-8
                res_j_given_i = Xj - float(np.dot(Xj, Xi) / (n_samples * var_i)) * Xi
                
                var_j = float(np.var(Xj)) + 1e-8
                res_i_given_j = Xi - float(np.dot(Xi, Xj) / (n_samples * var_j)) * Xj

            score_i_to_j = _rff_hsic(res_j_given_i.reshape(-1, 1), Xi_2d, D=n_rff)
            score_j_to_i = _rff_hsic(res_i_given_j.reshape(-1, 1), Xj_2d, D=n_rff)

            # Positive delta favors i → j / Delta dương ưu tiên i → j
            delta = score_j_to_i - score_i_to_j
            M[i, j] = delta
            M[j, i] = -delta

    # Sort by root score (descending) / Sắp xếp theo điểm gốc (giảm dần)
    root_scores = M.sum(axis=1)
    order = np.argsort(root_scores)[::-1].tolist()
    
    if verbose:
        scores_str = ", ".join(f"X{k}={root_scores[k]:.3f}" for k in order)
        print(f"  [TopoSort] Final Pairwise Root Scores: [{scores_str}]")

    return order
