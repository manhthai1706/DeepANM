"""
FastANM: Two-phase causal discovery baseline.
Phase 1: HSIC Greedy TopSort to establish causal order.
Phase 2: Random Forest + CI Pruning for nonlinear edge selection.

FastANM: Baseline khám phá nhân quả hai pha.
Pha 1: HSIC Greedy TopoSort để xác định thứ tự nhân quả.
Pha 2: Random Forest + CI Pruning để chọn cạnh phi tuyến.
"""

import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import IsolationForest

from src.core.toposort import hsic_greedy_order
from src.utils.adaptive_lasso import adaptive_lasso_dag


class FastANM:
    """
    Lightweight causal discovery model: TopoSort + Adaptive Edge Selection.
    No neural network — designed for speed and interpretability.
    Used as Phase 1+2 of the full DeepANM pipeline.

    Mô hình khám phá nhân quả nhẹ: TopoSort + Chọn Cạnh Thích nghi.
    Không dùng mạng neural — thiết kế cho tốc độ và khả năng giải thích.
    Dùng làm Pha 1+2 trong pipeline DeepANM đầy đủ.
    """

    def __init__(self):
        self.causal_order_ = None
        self.W_ = None

    def _preprocess(self, X, apply_isolation=False, apply_quantile=False):
        """Remove outliers and/or normalize data.
        Loại bỏ ngoại lệ và/hoặc chuẩn hóa dữ liệu."""
        if apply_isolation:
            mask = IsolationForest(contamination=0.05, random_state=42).fit_predict(X) == 1
            X = X[mask]
        if apply_quantile:
            X = QuantileTransformer(output_distribution='normal').fit_transform(X)
        return X

    def fit(self, X, apply_quantile=False, apply_isolation=False, verbose=True,
            layer_constraint=None, use_rf=True, use_ci_pruning=True):
        """
        Run FastANM on data X and return the discovered binary DAG matrix.

        Parameters
        ----------
        layer_constraint : dict mapping node index → layer level (optional prior)
        use_rf           : use Random Forest importance (else linear OLS)
        use_ci_pruning   : apply conditional independence pruning after RF

        Returns
        -------
        W_bin : (n_vars, n_vars) binary adjacency matrix

        Chạy FastANM trên dữ liệu X và trả về ma trận DAG nhị phân đã khám phá.

        Tham số
        -------
        layer_constraint : dict ánh xạ chỉ số node → tầng (tri thức trước tùy chọn)
        use_rf           : dùng tầm quan trọng Random Forest (ngược lại OLS tuyến tính)
        use_ci_pruning   : áp dụng kiểm định CI sau RF

        Trả về
        ------
        W_bin : ma trận kề nhị phân (n_vars, n_vars)
        """
        X_p = self._preprocess(X, apply_isolation=apply_isolation, apply_quantile=apply_quantile)
        
        if verbose:
            print("[FastMode] Step 1/2: Running TopoSort (HSIC Sink-First)...")
        
        self.causal_order_ = hsic_greedy_order(X_p, verbose=verbose)
        
        if verbose:
            order_str = " → ".join(f"X{i}" for i in self.causal_order_)
            print(f"[FastMode] Causal order: {order_str}")
        
        if verbose:
            print("[FastMode] Step 2/2: Running Adaptive LASSO for edge selection...")
            
        self.W_ = adaptive_lasso_dag(X_p, self.causal_order_, layer_constraint=layer_constraint,
                                     use_rf=use_rf, use_ci_pruning=use_ci_pruning)
        
        if verbose:
            print(f"[FastMode] Done! Discovered {int(self.W_.sum())} edges.")
        
        return self.W_
