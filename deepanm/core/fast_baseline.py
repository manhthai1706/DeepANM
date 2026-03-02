"""
Fast Baseline Logic (Lightweight Mode)
Bypasses Deep Learning entirely. Uses only Phase 1 (TopoSort) + Phase 3 (Adaptive LASSO).
"""
import numpy as np
from deepanm.core.toposort import hsic_greedy_order
from deepanm.utils.adaptive_lasso import adaptive_lasso_dag

class FastANM:
    """
    Lightweight Mode / Tốc độ Ánh sáng.
    Chỉ chạy Phase 1 (TopoSort) + Phase 3 (Adaptive LASSO).
    Bỏ qua hoàn toàn Neural Network (Phase 2), không cần PyTorch.
    Phù hợp để test nhanh (sanity check) hoặc làm baseline so sánh trên bộ dữ liệu mới.
    """
    def __init__(self):
        self.causal_order_ = None
        self.W_ = None
        
    def _preprocess(self, X, apply_isolation=False, apply_quantile=False):
        if apply_isolation:
            from sklearn.ensemble import IsolationForest
            mask = IsolationForest(contamination=0.05, random_state=42).fit_predict(X) == 1
            X = X[mask]
        if apply_quantile:
            from sklearn.preprocessing import QuantileTransformer
            X = QuantileTransformer(output_distribution='normal').fit_transform(X)
        return X

    def fit(self, X, apply_quantile=False, apply_isolation=False, verbose=True):
        """
        Train the fast baseline model and return the discovered DAG matrix.
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
            
        self.W_ = adaptive_lasso_dag(X_p, self.causal_order_)
        
        if verbose:
            print(f"[FastMode] Done! Discovered {int(self.W_.sum())} edges.")
            
        return self.W_
