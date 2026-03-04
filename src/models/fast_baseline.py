"""
FastANM: Two-phase causal discovery baseline.
Phase 1: HSIC Greedy TopSort to establish causal order.
Phase 2: Random Forest + CI Pruning for nonlinear edge selection.

FastANM: Giải pháp baseline khám phá nhân quả hai pha.
Pha 1: Sắp xếp Topo tham lam bằng HSIC để xác định thứ tự nhân quả.
Pha 2: Random Forest + Kiểm định CI để lựa chọn cạnh phi tuyến.
"""

import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import IsolationForest

from src.core.toposort import hsic_greedy_order
from src.utils.adaptive_lasso import adaptive_lasso_dag


class FastANM:
    """
    Non-neural causal discovery engine focused on speed and high structural precision.
    Động cơ khám phá nhân quả không dùng mạng neural, tập trung vào tốc độ và độ chính xác cấu trúc.

    Workflow / Quy trình:
    1. Establish total order via pairwise hsic asymmetry. / Xác định thứ tự tổng thông qua bất đối xứng hsic từng cặp.
    2. Prune edges within that order via non-linear feature selection. / Lọc cạnh trong thứ tự đó qua chọn đặc trưng phi tuyến.
    """

    def __init__(self):
        """Initialize state variables. / Khởi tạo biến trạng thái."""
        self.causal_order_ = None # Stores the topological ordering / Lưu trữ thứ tự topological
        self.W_ = None            # Stores the final binary adjacency matrix / Lưu ma trận kề nhị phân cuối cùng

    def _preprocess(self, X, apply_isolation=False, apply_quantile=False):
        """
        Data cleaning: removes outliers and maps distribution to normal space.
        Làm sạch dữ liệu: loại bỏ ngoại lệ và ánh xạ phân phối về không gian chuẩn.
        """
        if apply_isolation:
            # Use Isolation Forest to drop anomalies / Dùng Isolation Forest để loại bỏ bất thường
            mask = IsolationForest(contamination=0.05, random_state=42).fit_predict(X) == 1
            X = X[mask]
        if apply_quantile:
            # Normalize to Gaussian (required for kernel performance) / Chuẩn hóa về Gaussian (cần cho hiệu năng nhân)
            X = QuantileTransformer(output_distribution='normal').fit_transform(X)
        return X

    def fit(self, X, apply_quantile=False, apply_isolation=False, verbose=True,
            layer_constraint=None, use_rf=True, use_ci_pruning=True):
        """
        Run the FastANM discovery pipeline on dataset X.
        Chạy pipeline khám phá FastANM trên tập dữ liệu X.

        Parameters / Tham số
        ----------
        layer_constraint : prior knowledge about variable levels. / tri thức trước về các tầng biến.
        use_rf           : toggles Random Forest importance. / bật/tắt tầm quan trọng Random Forest.
        use_ci_pruning   : toggles secondary statistical pruning. / bật/tắt lọc thống kê lần hai.

        Returns / Trả về
        -------
        W_bin : (n_vars, n_vars) discovered binary DAG. / ma trận DAG nhị phân đã khám phá.
        """
        # 0. Clean the data / Làm sạch dữ liệu
        X_p = self._preprocess(X, apply_isolation=apply_isolation, apply_quantile=apply_quantile)
        
        # 1. Phase 1: Topological Sort / Pha 1: Sắp xếp Topological
        if verbose:
            print("[FastMode] Step 1/2: Running TopoSort (HSIC Sink-First)...")
        
        # Determine root-to-leaf flow / Xác định dòng chảy từ gốc tới lá
        self.causal_order_ = hsic_greedy_order(X_p, verbose=verbose)
        
        if verbose:
            order_str = " → ".join(f"X{i}" for i in self.causal_order_)
            print(f"[FastMode] Causal order: {order_str}")
        
        # 2. Phase 2: Structural Link Discovery / Pha 2: Khám phá liên kết cấu trúc
        if verbose:
            print("[FastMode] Step 2/2: Running Adaptive LASSO for edge selection...")
            
        # Select best edges respecting the topological flow / Chọn các cạnh tốt nhất tuân thủ dòng chảy topo
        self.W_ = adaptive_lasso_dag(X_p, self.causal_order_, layer_constraint=layer_constraint,
                                     use_rf=use_rf, use_ci_pruning=use_ci_pruning)
        
        if verbose:
            print(f"[FastMode] Done! Discovered {int(self.W_.sum())} edges.")
        
        return self.W_ # Final DAG structure matrix / Ma trận cấu trúc DAG cuối cùng
