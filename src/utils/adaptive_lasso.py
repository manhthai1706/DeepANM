"""
DeepANM Edge Selection via Nonlinear Adaptive LASSO.
Replaces brittle fixed-threshold edge selection with:
  1. Random Forest permutation importance (nonlinear edge scoring)
  2. Conditional Independence (CI) pruning to remove indirect paths

DeepANM Chọn Cạnh qua Adaptive LASSO Phi tuyến.
Thay thế lựa chọn cạnh ngưỡng cố định bằng:
  1. Tầm quan trọng hoán vị Random Forest (chấm điểm cạnh phi tuyến)
  2. Kiểm định Độc lập có Điều kiện (CI) để loại bỏ đường gián tiếp
"""

import numpy as np


def _ols_fit(X_parents: np.ndarray, Xj: np.ndarray) -> np.ndarray:
    """
    Linear OLS coefficients for Xj ~ X_parents (mean-centered).
    Dự phòng OLS tuyến tính cho bài toán hồi quy Xj theo X_parents.
    """
    Xc = X_parents - X_parents.mean(axis=0) # Center parents / Chuẩn hóa tâm tập cha
    yc = Xj - Xj.mean() # Center target / Chuẩn hóa tâm tập biến mục tiêu
    try:
        # Solve least squares for linear regression / Giải bình phương tối thiểu
        beta, _, _, _ = np.linalg.lstsq(Xc, yc, rcond=None)
        return beta
    except Exception:
        return np.ones(X_parents.shape[1]) # Fallback to 1s if fails / Trả về 1 nếu lỗi


def _adaptive_lasso_column(X_parents: np.ndarray, Xj: np.ndarray,
                            gamma: float = 1.0) -> np.ndarray:
    """
    Nonlinear edge scoring via Random Forest permutation importance.
    Chấm điểm cạnh phi tuyến qua tầm quan trọng hoán vị của Random Forest.

    Edge Logic / Quy tắc xác nhận cạnh:
      Confirmed if mean R-squared drop > 3% AND signal is > 2x noise std.
      Xác nhận nếu mức giảm R-squared > 3% VÀ tín hiệu > 2 lần độ lệch nhiễu.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
        
        n, p = X_parents.shape
        if p == 0:
            return np.array([])

        # Fit a non-linear forest regressor / Tuyển rừng hồi quy phi tuyến
        reg = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=5,
                                    n_jobs=-1, random_state=42)
        reg.fit(X_parents, Xj)
        
        # Measure feature importance by shuffling each column / Đo mức độ quan trọng bằng cách xáo trộn cột
        result = permutation_importance(reg, X_parents, Xj, n_repeats=5, random_state=42, n_jobs=-1)
        
        mean_drop = result.importances_mean
        std_drop = result.importances_std
        
        # Binary mask: true parents must show significant R2 impact beyond noise
        # Mặt nạ nhị phân: cha thật phải có ảnh hưởng R2 đáng kể vượt qua nhiễu
        mask = (mean_drop > 0.03) & (mean_drop > 2.0 * std_drop)
        return mask.astype(float)

    except Exception:
        return _ols_fit(X_parents, Xj) # Fallback to OLS / Dự phòng OLS


def _partial_correlation_pruning(X: np.ndarray, parent: int, target: int, other_parents: list) -> bool:
    """
    Test if parent is conditionally independent of target given other_parents.
    Sử dụng phần dư phi tuyến (HistGBM) và kiểm định Pearson để phát hiện đường gián tiếp.

    Returns True if the edge is likely redundant (indirect path).
    Trả về True nếu cạnh có khả năng thừa (đường đi gián tiếp).
    """
    if len(other_parents) == 0:
        return False # Nothing to condition on / Không có gì để điều kiện hóa
        
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from scipy.stats import pearsonr
        
        Z = X[:, other_parents] # Conditioning set / Tập điều kiện
        Xi = X[:, parent]       # Candidate parent / Biến cha ứng viên
        Xj = X[:, target]       # Target variable / Biến mục tiêu
        
        # Predict both Xi and Xj from Z and extract residuals
        # Dự báo cả Xi và Xj từ Z và lấy phần dư
        reg_i = HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=42).fit(Z, Xi)
        reg_j = HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=42).fit(Z, Xj)
        
        res_i = Xi - reg_i.predict(Z) # Feature residuals / Phần dư đặc trưng
        res_j = Xj - reg_j.predict(Z) # Target residuals / Phần dư mục tiêu
        
        # Test if residuals are uncorrelated / Kiểm tra xem phần dư có không tương quan không
        corr, p_value = pearsonr(res_i, res_j)
        
        # Pruning condition: low p-value indicates correlation, high p-value indicates independence
        # Điều kiện loại bỏ: p-value cao chỉ ra tính độc lập (kiểm định Pearson)
        return p_value > 0.01 and abs(corr) < 0.1
    except Exception:
        return False


def adaptive_lasso_dag(X: np.ndarray, causal_order: list, layer_constraint=None,
                       use_rf=True, use_ci_pruning=True) -> np.ndarray:
    """
    Assembles a binary DAG matrix using per-node nonlinear parent selection.
    Lắp ráp ma trận DAG nhị phân qua việc chọn cha phi tuyến cho từng biến.

    Parameters / Tham số
    ----------
    causal_order : topological ordering from Phase 1 / thứ tự topo từ Pha 1.
    use_rf       : toggle nonlinear selection / bật/tắt chọn cạnh phi tuyến.
    use_ci_pruning: remove indirect paths via residuals / loại bỏ đường gián tiếp.
    """
    n_vars = X.shape[1]
    W_bin = np.zeros((n_vars, n_vars), dtype=np.float32)

    for step, j in enumerate(causal_order):
        # Potential parents are variables appearing EARLIER in the causal order
        # Cha ứng viên là các biến xuất hiện TRƯỚC trong thứ tự nhân quả
        potential_parents = causal_order[:step]
        if len(potential_parents) == 0:
            continue
            
        # Optional: Apply biological layer constraints (prevent cross-layer violation)
        # Tùy chọn: Áp dụng ràng buộc tầng sinh học (tránh vi phạm thứ tự tầng)
        target_layer = layer_constraint.get(j, 5) if layer_constraint else None
        valid_parents = []
        for p in potential_parents:
            if layer_constraint is None:
                valid_parents.append(p)
            else:
                parent_layer = layer_constraint.get(p, 5)
                # Only allow edges from higher layer to lower or equal layer (sink flow)
                # Chỉ cho phép cạnh từ tầng cao xuống thấp hoặc bằng (dòng chảy hạ nguồn)
                if parent_layer <= target_layer:
                    valid_parents.append(p)
                
        if len(valid_parents) == 0:
            continue

        X_parents = X[:, valid_parents]
        Xj = X[:, j]

        # Phase 2 component: Feature Importance / Pha 2: Chấm điểm đặc trưng quan trọng
        coef = _adaptive_lasso_column(X_parents, Xj) if use_rf else _ols_fit(X_parents, Xj)

        for k, parent in enumerate(valid_parents):
            if abs(coef[k]) > 1e-8:
                # Optional: Secondary pruning to remove transitive paths (A->B, B->C => Prune A->C)
                # Tùy chọn: Lọc lần hai để loại bỏ đường bắc cầu
                if use_ci_pruning:
                    other_parents = [p for p in valid_parents if p != parent]
                    should_prune = _partial_correlation_pruning(X, parent, j, other_parents)
                else:
                    should_prune = False
                
                if not should_prune:
                    W_bin[parent, j] = 1.0 # Confirmed causal edge / Xác nhận cạnh nhân quả

    return W_bin


def adaptive_lasso_from_ate(ATE: np.ndarray, X: np.ndarray,
                             causal_order: list) -> np.ndarray:
    """
    "Double-Gate" Selection: Intersection of RF Importance AND Neural ATE Causal Strength.
    Lọc "Cổng Đôi": Giao của Tầm quan trọng RF VÀ Cường độ nhân quả ATE Neural.

    Ensures edges are both statistically robust and numerically significant.
    Đảm bảo cạnh vừa bền vững về thống kê vừa có ý nghĩa về giá trị số học.
    """
    W_lasso = adaptive_lasso_dag(X, causal_order) # Gate 1: Statistical presence / Cổng 1: Hiện diện thống kê
    # Gate 2: Causal Magnitude (ATE must be > 0.01) / Cổng 2: Cường độ nhân quả (ATE > 0.01)
    ATE_gate = (np.abs(ATE) > 0.01).astype(float)
    return (W_lasso * ATE_gate).astype(float)
