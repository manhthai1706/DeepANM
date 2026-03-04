"""
HSIC-based Greedy Causal Topological Ordering (TopoSort).
Inspired by RESIT (Peters et al., 2014) and LiNGAM.
Sắp xếp Topo Nhân quả Tham lam dựa trên tiêu chí hsic (Peters et al., 2014).

Uses pairwise ANM asymmetry with RFF-HSIC (O(N·D)) to robustly order variables.
Sử dụng tính bất đối xứng ANM từng cặp với RFF-HSIC (độ phức tạp O(N·D)) để sắp xếp biến ổn định.
"""

import numpy as np


# ---------------------------------------------------------------------------
# RFF-approximated HSIC Logic / Logic tính HSIC xấp xỉ qua RFF
# ---------------------------------------------------------------------------

def _rff_features(Z: np.ndarray, D: int, bw: float, rng: np.random.RandomState) -> np.ndarray:
    """Random Fourier Features for approximating the infinite-dimensional RBF kernel.
    Đặc trưng Fourier ngẫu nhiên để xấp xỉ nhân RBF (Gaussian) vô hạn chiều."""
    d = Z.shape[1]
    # Sample random weight matrix and bias / Lấy mẫu ma trận trọng số và độ lệch ngẫu nhiên
    W = rng.randn(d, D) / max(bw, 1e-6)
    b = rng.uniform(0, 2 * np.pi, D)
    # Cosine map maps data into a space where dot product ≈ RBF kernel
    # Ánh xạ cos đưa dữ liệu vào không gian có tích vô hướng xấp xỉ nhân RBF
    phi = np.sqrt(2.0 / D) * np.cos(Z @ W + b)
    phi -= phi.mean(axis=0) # Implicit kernel centering / Chuẩn hóa tâm nhân ngầm định
    return phi


def _bw_estimate(Z: np.ndarray) -> float:
    """Bandwidth estimation via median distance heuristic (Silverman's rule variant).
    Ước lượng độ rộng băng thông qua heuristic trung vị khoảng cách cặp."""
    sub = Z[:200] if Z.shape[0] > 200 else Z # Sample subsample for speed / Lấy mẫu con để tăng tốc
    # Pairwise Euclidean distances / Khoảng cách Euclid từng cặp
    dists = np.sqrt(np.sum((sub[:, None] - sub[None, :]) ** 2, axis=-1))
    upper = dists[np.triu_indices(sub.shape[0], k=1)] # Upper triangle / Tam giác trên
    med = np.median(upper) if upper.size > 0 else 1.0
    return max(med, 1e-6)


def _rff_hsic(X: np.ndarray, Y: np.ndarray, D: int = 128, seed: int = 0) -> float:
    """Computes linearized HSIC in O(ND) time instead of classic O(N²).
    Tính toán chỉ số độc lập HSIC tuyến tính hóa trong thời gian O(ND)."""
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    n = X.shape[0]
    if n < 4: return 0.0

    rng = np.random.RandomState(seed)
    # Project both variables into kernel feature space / Chiếu cả hai biến vào không gian nhân
    phi_x = _rff_features(X, D, _bw_estimate(X), rng)
    phi_y = _rff_features(Y, D, _bw_estimate(Y), rng)

    # Statistical dependence ≈ cross-covariance of features / Phụ thuộc thống kê ≈ hiệp phương sai chéo đặc trưng
    C = (phi_x.T @ phi_y) / n
    return float(np.sum(C ** 2)) # Frobenius norm as hsic proxy / Chuẩn Frobenius làm đại diện hsic


# ---------------------------------------------------------------------------
# Nonlinear Residual Extraction / Trích xuất phần dư phi tuyến
# ---------------------------------------------------------------------------

def _nonlinear_residual(X_cause: np.ndarray, Xi: np.ndarray) -> np.ndarray:
    """
    Fits Xi = f(X_cause) + noise and returns the residual 'noise'.
    Khớp hàm Xi = f(X_cause) + nhiễu và trả về phần 'nhiễu' (phần dư).
    """
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        # Use Boosted Trees as a universal functional approximator
        # Dùng Boosted Trees làm bộ xấp xỉ hàm vạn năng
        reg = HistGradientBoostingRegressor(
            max_iter=50, max_depth=3, learning_rate=0.1,
            random_state=42, early_stopping=False
        )
        reg.fit(X_cause, Xi)
        return Xi - reg.predict(X_cause)
    except Exception:
        # Fallback to linear least squares / Dự phòng bình phương tối thiểu tuyến tính
        w = np.linalg.lstsq(X_cause, Xi, rcond=None)[0]
        return Xi - X_cause @ w


# ---------------------------------------------------------------------------
# Main TopoSort Algorithm / Thuật toán TopoSort chính
# ---------------------------------------------------------------------------

def hsic_greedy_order(X: np.ndarray, n_rff: int = 128, verbose: bool = False) -> list:
    """
    Greedy search for root nodes using ANM identifiability logic. 
    Tìm kiếm tham lam các nút gốc dựa trên logic định danh của mô hình ANM.

    Asymmetry Principle / Nguyên lý bất đối xứng:
      If X -> Y, then HSIC(Y - f(X), X) < HSIC(X - g(Y), Y).
      Nếu X -> Y, thì hsic của phần dư Y theo X sẽ nhỏ hơn chiều ngược lại.
    """
    n_samples, n_vars = X.shape
    if n_vars == 1: return [0]

    # Pre-scale for numerical stability / Chuẩn hóa trước để ổn định số học
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    X_scaled = (X - X.mean(axis=0)) / std

    M = np.zeros((n_vars, n_vars)) # Direction preference matrix / Ma trận ưu tiên chiều cạnh
    
    # Process small-to-medium graphs with high nonlinear precision
    # Xử lý đồ thị vừa và nhỏ với độ chính xác phi tuyến cao
    use_nonlinear = n_vars <= 20 

    for i in range(n_vars):
        Xi = X_scaled[:, i]
        Xi_2d = Xi.reshape(-1, 1)
        for j in range(i + 1, n_vars):
            Xj = X_scaled[:, j]
            Xj_2d = Xj.reshape(-1, 1)

            # Fit in both directions i -> j and j -> i / Khớp theo cả hai chiều i -> j và j -> i
            if use_nonlinear:
                res_j_given_i = _nonlinear_residual(Xi_2d, Xj)
                res_i_given_j = _nonlinear_residual(Xj_2d, Xi)
            else:
                var_i = float(np.var(Xi)) + 1e-8
                res_j_given_i = Xj - float(np.dot(Xj, Xi) / (n_samples * var_i)) * Xi
                var_j = float(np.var(Xj)) + 1e-8
                res_i_given_j = Xi - float(np.dot(Xi, Xj) / (n_samples * var_j)) * Xj

            # Compare independence of residuals / So sánh tính độc lập của phần dư
            score_i_to_j = _rff_hsic(res_j_given_i.reshape(-1, 1), Xi_2d, D=n_rff)
            score_j_to_i = _rff_hsic(res_i_given_j.reshape(-1, 1), Xj_2d, D=n_rff)

            # Preference delta (> 0 favors i -> j) / Độ lệch ưu tiên (dương ưu tiên i -> j)
            delta = score_j_to_i - score_i_to_j
            M[i, j] = delta
            M[j, i] = -delta

    # Extract global order by aggregating pairwise preferences
    # Trích xuất thứ tự tổng quát bằng cách tổng hợp các ưu tiên từng cặp
    root_scores = M.sum(axis=1) # High score means variable is a likely source / Điểm cao nghĩa là biến dễ là nguồn gốc
    order = np.argsort(root_scores)[::-1].tolist() # Sort descending / Sắp xếp giảm dần
    
    if verbose:
        scores_str = ", ".join(f"X{k}={root_scores[k]:.3f}" for k in order)
        print(f"  [TopoSort] Final Pairwise Root Scores: [{scores_str}]")

    return order
