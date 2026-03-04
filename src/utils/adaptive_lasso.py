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
    Used as a fast linear fallback when RF is disabled.

    Hệ số OLS tuyến tính cho Xj ~ X_parents (chuẩn hóa tâm).
    Dùng làm dự phòng tuyến tính nhanh khi tắt RF.
    """
    Xc = X_parents - X_parents.mean(axis=0)
    yc = Xj - Xj.mean()
    try:
        beta, _, _, _ = np.linalg.lstsq(Xc, yc, rcond=None)
        return beta
    except Exception:
        return np.ones(X_parents.shape[1])


def _adaptive_lasso_column(X_parents: np.ndarray, Xj: np.ndarray,
                            gamma: float = 1.0) -> np.ndarray:
    """
    Nonlinear edge scoring via Random Forest permutation importance.
    An edge i→j is confirmed if: mean permutation drop > 3% AND drop > 2σ noise.
    Falls back to OLS if sklearn is unavailable.

    Chấm điểm cạnh phi tuyến qua tầm quan trọng hoán vị Random Forest.
    Cạnh i→j được xác nhận nếu: mức giảm trung bình > 3% VÀ giảm > 2σ nhiễu.
    Dự phòng OLS nếu sklearn không có sẵn.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
        
        n, p = X_parents.shape
        if p == 0:
            return np.array([])

        reg = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=5,
                                    n_jobs=-1, random_state=42)
        reg.fit(X_parents, Xj)
        
        result = permutation_importance(reg, X_parents, Xj, n_repeats=5, random_state=42, n_jobs=-1)
        
        mean_drop = result.importances_mean
        std_drop = result.importances_std
        # Edge confirmed: meaningful R² drop and signal > 2× noise std
        # Cạnh xác nhận: mức giảm R² có ý nghĩa và tín hiệu > 2× độ lệch chuẩn nhiễu
        mask = (mean_drop > 0.03) & (mean_drop > 2.0 * std_drop)
        return mask.astype(float)

    except Exception:
        return _ols_fit(X_parents, Xj)

def _partial_correlation_pruning(X: np.ndarray, parent: int, target: int, other_parents: list) -> bool:
    """
    Test if parent ⊥ target | other_parents (conditional independence).
    Uses nonlinear HistGBM residuals + Pearson correlation test.
    Returns True if the edge should be pruned (indirect path detected).

    Kiểm định parent ⊥ target | other_parents (độc lập có điều kiện).
    Dùng phần dư HistGBM phi tuyến + kiểm định tương quan Pearson.
    Trả về True nếu cạnh cần loại bỏ (phát hiện đường gián tiếp).
    """
    if len(other_parents) == 0:
        return False  # No conditioning set → cannot be indirect / Không có tập điều kiện → không thể gián tiếp
        
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from scipy.stats import pearsonr
        
        Z = X[:, other_parents]
        Xi = X[:, parent]
        Xj = X[:, target]
        
        reg_i = HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=42).fit(Z, Xi)
        reg_j = HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=42).fit(Z, Xj)
        
        res_i = Xi - reg_i.predict(Z)
        res_j = Xj - reg_j.predict(Z)
        
        corr, p_value = pearsonr(res_i, res_j)
        
        # Prune if conditionally independent: p > 0.01 AND |corr| < 0.1
        # Cắt nếu độc lập có điều kiện: p > 0.01 VÀ |corr| < 0.1
        return p_value > 0.01 and abs(corr) < 0.1
    except Exception:
        return False


def adaptive_lasso_dag(X: np.ndarray, causal_order: list, layer_constraint=None,
                       use_rf=True, use_ci_pruning=True) -> np.ndarray:
    """
    Build a binary DAG adjacency matrix using edge selection.

    For each variable j (in causal order), identifies its true parents among
    topological ancestors via RF importance and optional CI pruning.

    Parameters
    ----------
    X               : (n_samples, n_vars) observational data
    causal_order    : root-first variable ordering (from TopoSort)
    layer_constraint: dict mapping node → layer level for feed-forward enforcement
    use_rf          : if True, use RF importance; if False, use linear OLS
    use_ci_pruning  : if True, prune indirect edges via CI test

    Returns
    -------
    W_bin : (n_vars, n_vars) binary matrix; W_bin[i,j]=1 means confirmed edge i→j

    Xây dựng ma trận kề DAG nhị phân qua chọn cạnh.

    Với mỗi biến j (theo thứ tự nhân quả), xác định cha thật trong các tổ tiên
    topo qua tầm quan trọng RF và kiểm định CI tùy chọn.

    Tham số
    -------
    X               : dữ liệu quan sát (n_samples, n_vars)
    causal_order    : thứ tự biến gốc-trước (từ TopoSort)
    layer_constraint: dict ánh xạ node → tầng để ép chiều feed-forward
    use_rf          : True dùng RF, False dùng OLS tuyến tính
    use_ci_pruning  : True loại bỏ cạnh gián tiếp qua kiểm định CI

    Trả về
    ------
    W_bin : ma trận nhị phân (n_vars, n_vars); W_bin[i,j]=1 nghĩa là cạnh i→j được xác nhận
    """
    n_vars = X.shape[1]
    W_bin = np.zeros((n_vars, n_vars), dtype=np.float32)

    for step, j in enumerate(causal_order):
        potential_parents = causal_order[:step]  # All predecessors in causal order / Mọi tiền nhiệm
        if len(potential_parents) == 0:
            continue
            
        # Apply layer constraint if provided / Áp dụng ràng buộc tầng nếu có
        target_layer = layer_constraint.get(j, 5) if layer_constraint else None
        valid_parents = []
        for p in potential_parents:
            if layer_constraint is None:
                valid_parents.append(p)
            else:
                parent_layer = layer_constraint.get(p, 5)
                if parent_layer <= target_layer:  # Feed-forward only / Chỉ chiều thuận
                    valid_parents.append(p)
                
        if len(valid_parents) == 0:
            continue

        X_parents = X[:, valid_parents]
        Xj = X[:, j]

        # Edge scoring: RF (nonlinear) or OLS (linear) / Chấm điểm cạnh: RF hoặc OLS
        coef = _adaptive_lasso_column(X_parents, Xj) if use_rf else _ols_fit(X_parents, Xj)

        for k, parent in enumerate(valid_parents):
            if abs(coef[k]) > 1e-8:
                # CI Pruning: remove edge if parent ⊥ j | other_parents
                # CI Pruning: loại cạnh nếu parent ⊥ j | other_parents
                if use_ci_pruning:
                    other_parents = [p for p in valid_parents if p != parent]
                    should_prune = _partial_correlation_pruning(X, parent, j, other_parents)
                else:
                    should_prune = False
                
                if not should_prune:
                    W_bin[parent, j] = 1.0

    return W_bin


def adaptive_lasso_from_ate(ATE: np.ndarray, X: np.ndarray,
                             causal_order: list) -> np.ndarray:
    """
    Double-gate edge selection: RF LASSO ∩ ATE > threshold.
    Keeps edges that are both statistically nonzero (RF) and causally significant (ATE).

    Chọn cạnh cổng kép: RF LASSO ∩ ATE > ngưỡng.
    Giữ cạnh vừa khác không về thống kê (RF) vừa có ý nghĩa nhân quả (ATE).
    """
    W_lasso = adaptive_lasso_dag(X, causal_order)
    # ATE gate: filter edges with negligible causal magnitude / Cổng ATE: lọc cạnh có cường độ nhân quả nhỏ
    ATE_gate = (np.abs(ATE) > 0.01).astype(float)
    return (W_lasso * ATE_gate).astype(float)
