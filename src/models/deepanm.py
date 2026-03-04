"""
DeepANM: Deep Additive Noise Model for Causal Discovery.
Three-phase pipeline: TopoSort → Edge Selection → Neural SCM Fitting.

DeepANM: Mô hình Nhiễu Cộng Sâu cho Khám phá Nhân quả.
Pipeline ba pha: Sắp xếp Topo → Chọn Cạnh → Khớp SCM Neural.
"""

import numpy as np
import torch
import torch.nn as nn


def _order_from_topo_mask(topo_mask: np.ndarray, n_vars: int) -> list:
    """
    Reconstruct root-first causal order from a stored topo_mask using Kahn's algorithm.
    topo_mask[i, j] = 1 means edge i → j is allowed (i is an ancestor of j).

    Khôi phục thứ tự nhân quả gốc-trước từ topo_mask bằng thuật toán Kahn.
    topo_mask[i, j] = 1 nghĩa là cạnh i → j được phép (i là tổ tiên của j).
    """
    in_deg = {j: 0 for j in range(n_vars)}
    children = {i: [] for i in range(n_vars)}
    for i in range(n_vars):
        for j in range(n_vars):
            if topo_mask[i, j] > 0.5:
                children[i].append(j)
                in_deg[j] += 1

    queue = [v for v in range(n_vars) if in_deg[v] == 0]
    order = []
    while queue:
        v = queue.pop(0)
        order.append(v)
        for u in children[v]:
            in_deg[u] -= 1
            if in_deg[u] == 0:
                queue.append(u)

    # Fallback if a cycle is detected (should not happen with valid DAG)
    # Dự phòng nếu có chu trình (không nên xảy ra với DAG hợp lệ)
    return order if len(order) == n_vars else list(range(n_vars))

class DeepANM(nn.Module):
    """
    Deep Additive Noise Model for multivariate causal graph discovery.

    Three discovery modes:
        'fast' : FastANM (TopoSort + RF/CI LASSO) → graph structure,
                 then Neural SCM Fitter computes precise ATE. (ALM Disabled)
        'topo' : TopoSort constrains order; Neural Net finds edges. (ALM Disabled)
        'alm'  : Free Deep Learning exploration. (ALM Enabled)

    Mô hình Nhiễu Cộng Sâu cho khám phá đồ thị nhân quả đa biến.

    Ba chế độ khám phá:
        'fast' : FastANM (TopoSort + RF/CI LASSO) → cấu trúc đồ thị,
                 sau đó Neural SCM Fitter tính ATE chính xác. (Tắt ALM)
        'topo' : TopoSort ràng buộc thứ tự; Mạng Neural tìm cạnh. (Tắt ALM)
        'alm'  : Khám phá tự do bằng Deep Learning. (Bật ALM)
    """

    def __init__(self, x_dim=None, n_clusters=2, hidden_dim=32, lda=1.0, device=None):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.lda = lda
        self.x_dim = x_dim
        self.n_clusters = n_clusters
        self.hidden_dim = hidden_dim
        self.core = None   # Built lazily on first fit() / Xây dựng lười biếng lần đầu fit()
        self.history = None

        if x_dim is not None:
            # Pre-build with default (no-prior) mask / Xây dựng trước với mặt nạ mặc định
            self._build_core(x_dim, X=None, causal_order=None, verbose=False)

    # ------------------------------------------------------------------
    # Internal helpers / Các hàm hỗ trợ nội bộ
    # ------------------------------------------------------------------

    def _build_core(self, x_dim, X=None, causal_order=None, causal_graph=None, verbose=True):
        """
        Build the GPPOM core engine with appropriate structural constraints.

        Priority order:
          1. causal_graph provided → SCM Fitter mode (fix structure, no ALM, no TopoSort)
          2. causal_order provided → skip TopoSort (no ALM)
          3. X provided, no order → run TopoSort now (no ALM)
          4. Neither              → free exploration (ALM Enabled)

        Xây dựng động cơ GPPOM lõi với ràng buộc cấu trúc phù hợp.

        Thứ tự ưu tiên:
          1. Có causal_graph → Chế độ SCM Fitter (cố định cấu trúc, tắt ALM)
          2. Có causal_order → bỏ qua TopoSort (tắt ALM)
          3. Có X, không có thứ tự → chạy TopoSort ngay (tắt ALM)
          4. Không có gì → khám phá tự do (bật ALM)
        """
        from src.core.gppom_hsic import GPPOMC_lnhsic_Core
        self.x_dim = x_dim

        if causal_graph is not None:
            order = None
        elif causal_order is not None:
            order = causal_order
            if verbose:
                print(f"[Discovery] Using cached TopoSort order. ALM Disabled.")
        elif X is not None:
            from src.core.toposort import hsic_greedy_order
            if verbose:
                print("[Discovery] Discovering TopoSort order (Sink-First)... ALM Disabled.")
            order = hsic_greedy_order(X, verbose=False)
        else:
            order = None
            if verbose:
                print("[Discovery] No prior knowledge. ALM Enabled.")

        self.core = GPPOMC_lnhsic_Core(
            x_dim, 0, self.n_clusters, self.hidden_dim, self.lda,
            self.device, causal_order=order, causal_graph=causal_graph
        )
        self.to(self.device)

    def _preprocess(self, X, apply_isolation=False, apply_quantile=False):
        """
        Optional preprocessing: outlier removal and quantile normalization.
        Tiền xử lý tùy chọn: loại ngoại lệ và chuẩn hóa phân vị.
        """
        if apply_isolation:
            from sklearn.ensemble import IsolationForest
            mask = IsolationForest(contamination=0.05, random_state=42).fit_predict(X) == 1
            X = X[mask]
        if apply_quantile:
            from sklearn.preprocessing import QuantileTransformer
            X = QuantileTransformer(output_distribution='normal').fit_transform(X)
        return X

    # ------------------------------------------------------------------
    # Public API / Giao diện công khai
    # ------------------------------------------------------------------

    def fit(self, X, epochs=200, batch_size=64, lr=2e-3, verbose=True,
            apply_quantile=False, apply_isolation=False,
            discovery_mode="fast", layer_constraint=None, _precomputed_order=None,
            use_rf=True, use_ci_pruning=True):
        """
        Train the DeepANM pipeline on observational data X.

        Parameters
        ----------
        discovery_mode : 'fast' | 'topo' | 'alm'  (see class docstring)
        layer_constraint : dict mapping node index → layer level (optional prior knowledge)
        use_rf          : use Random Forest for edge selection (else linear OLS)
        use_ci_pruning  : apply CI-based pruning after RF selection

        Huấn luyện pipeline DeepANM trên dữ liệu quan sát X.

        Tham số
        -------
        discovery_mode  : 'fast' | 'topo' | 'alm'  (xem docstring lớp)
        layer_constraint: dict ánh xạ chỉ số node → tầng (tri thức trước tùy chọn)
        use_rf          : dùng Random Forest để chọn cạnh (ngược lại dùng OLS tuyến tính)
        use_ci_pruning  : áp dụng kiểm định CI sau khi chọn cạnh RF
        """
        from src.utils.trainer import DeepANMTrainer
        X_p = self._preprocess(X, apply_isolation=apply_isolation, apply_quantile=apply_quantile)
        self.discovery_mode = discovery_mode

        causal_graph = None
        causal_order = _precomputed_order
        X_init = None
        
        if discovery_mode == "fast":
            from src.models.fast_baseline import FastANM
            if verbose: print("\n[DeepANM] Phase 1+2: FastANM Graph Discovery...")
            fast_model = FastANM()
            causal_graph = fast_model.fit(X, apply_quantile=apply_quantile, apply_isolation=apply_isolation,
                                          verbose=verbose, layer_constraint=layer_constraint,
                                          use_rf=use_rf, use_ci_pruning=use_ci_pruning)
            if verbose: print("\n[DeepANM] Phase 3: SCM Neural Fitting...")
        elif discovery_mode == "topo":
            X_init = X_p if causal_order is None else None
        elif discovery_mode == "alm":
            causal_order = None
            
        self._build_core(X_p.shape[1], X=X_init,
                         causal_order=causal_order, causal_graph=causal_graph, verbose=verbose)

        trainer = DeepANMTrainer(self, lr=lr)
        self.history = trainer.train(X_p, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return self.history

    def fit_bootstrap(self, X, n_bootstraps=5,
                      epochs=200, batch_size=64, lr=5e-3, verbose=True,
                      apply_quantile=False, apply_isolation=False, discovery_mode="fast",
                      layer_constraint=None, use_rf=True, use_ci_pruning=True, use_scm_filter=True):
        """
        Stability Selection via bootstrap resampling.
        Runs fit() multiple times on resampled data and aggregates results.
        In 'topo' mode, TopoSort runs ONCE on the full dataset before the loop.
        Returns (prob_matrix, avg_ATE_matrix).

        Lựa chọn ổn định qua lấy mẫu bootstrap.
        Chạy fit() nhiều lần trên dữ liệu lấy mẫu lại và tổng hợp kết quả.
        Trong chế độ 'topo', TopoSort chạy MỘT LẦN trên toàn bộ dữ liệu trước vòng lặp.
        Trả về (prob_matrix, avg_ATE_matrix).
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        X = self._preprocess(X, apply_isolation=apply_isolation, apply_quantile=apply_quantile)

        n_samples, n_vars = X.shape
        agg_W   = np.zeros((n_vars, n_vars))
        agg_bin = np.zeros((n_vars, n_vars))

        # Run TopoSort once for 'topo' mode to avoid redundant computation
        # Chạy TopoSort một lần cho chế độ 'topo' để tránh tính toán lặp lại
        self._causal_order = None
        if discovery_mode == "topo":
            from src.core.toposort import hsic_greedy_order
            if verbose:
                print("[TopoSort] Running ONCE on full dataset before bootstrap...")
            self._causal_order = hsic_greedy_order(X, verbose=False)
            if verbose:
                order_str = " → ".join(f"X{i}" for i in self._causal_order)
                print(f"[TopoSort] Causal order: {order_str}")
                print(f"[TopoSort] This order will be reused for all {n_bootstraps} bootstrap rounds.")

        for b in range(n_bootstraps):
            if verbose:
                print(f"[Bootstrap] Round {b+1}/{n_bootstraps}...")

            boot_data = X[np.random.choice(n_samples, n_samples, replace=True)]

            self.core = None
            self.fit(boot_data, epochs=epochs, batch_size=batch_size, lr=lr,
                     verbose=False, discovery_mode=discovery_mode,
                     layer_constraint=layer_constraint, _precomputed_order=self._causal_order,
                     use_rf=use_rf, use_ci_pruning=use_ci_pruning)

            ATE, W_bin = self.get_dag_matrix(X=boot_data, use_scm_filter=use_scm_filter)
            agg_W   += ATE
            agg_bin += W_bin

        return agg_bin / n_bootstraps, agg_W / n_bootstraps

    def get_dag_matrix(self, X=None, use_scm_filter=True):
        """
        Extract the learned causal adjacency matrix.

        Parameters
        ----------
        X              : data for ATE Jacobian computation (optional)
        use_scm_filter : if True, apply Adaptive ATE Gate (P15) in fast mode.
                         if False, return raw FastANM edges.

        Returns
        -------
        (ATE_matrix, W_binary) if X is provided
        (W_raw, W_binary_from_gate) otherwise

        Trích xuất ma trận kề nhân quả đã học.

        Tham số
        -------
        X              : dữ liệu để tính Jacobian ATE (tùy chọn)
        use_scm_filter : nếu True, áp dụng Cổng ATE Thích nghi (P15) ở chế độ fast.
                         nếu False, trả về cạnh thô từ FastANM.

        Trả về
        ------
        (ATE_matrix, W_binary) nếu có X
        (W_raw, W_binary_from_gate) nếu không có X
        """
        with torch.no_grad():
            W_dag_masked = self.core.W_dag * self.core.topo_mask
            W = W_dag_masked.detach().cpu().numpy()

            if X is not None:
                # Compute Neural ATE Jacobian / Tính Jacobian ATE Neural
                ATE = self.core.MLP.get_global_ate_matrix(
                    X, W_dag=W_dag_masked).cpu().numpy()

                if getattr(self, 'discovery_mode', None) == "fast":
                    W_lasso = self.core.topo_mask.cpu().numpy()
                    
                    if not use_scm_filter:
                        # Return raw FastANM edges without neural filtering
                        # Trả về cạnh thô FastANM không qua lọc neural
                        return ATE, W_lasso
                    
                    # Adaptive ATE Gate: prune bottom 15th percentile of edge strengths
                    # Cổng ATE Thích nghi: cắt 15% cạnh yếu nhất theo cường độ
                    ate_abs = np.abs(ATE)
                    edge_ates = ate_abs[W_lasso > 0.5]
                    
                    if len(edge_ates) > 0 and edge_ates.max() > 0:
                        adaptive_threshold = np.percentile(edge_ates, 15)
                        ATE_gate = (ate_abs > adaptive_threshold).astype(float)
                    else:
                        ATE_gate = (ate_abs > 1e-4).astype(float)
                    
                    return ATE, W_lasso * ATE_gate
                else:
                    # Reconstruct causal order, then apply Adaptive LASSO + ATE double-gate
                    # Khôi phục thứ tự nhân quả, áp dụng Adaptive LASSO + cổng ATE kép
                    causal_order = _order_from_topo_mask(
                        self.core.topo_mask.cpu().numpy(), self.x_dim)
                    from src.utils.adaptive_lasso import adaptive_lasso_from_ate
                    W_bin = adaptive_lasso_from_ate(ATE, X, causal_order)
                    return ATE, W_bin
            else:
                # No X: return raw weights and gate-thresholded binary matrix
                # Không có X: trả về trọng số thô và ma trận nhị phân qua cổng
                return W, (torch.sigmoid(self.core.W_logits).detach().cpu().numpy() > 0.5).astype(float) * self.core.topo_mask.cpu().numpy()

    def set_exogenous(self, exog_indices):
        """
        Mark variables as exogenous: block all incoming edges to these nodes.
        Call after _build_core() or fit() to take effect.

        Đánh dấu biến ngoại sinh: chặn mọi cạnh đến các node này.
        Gọi sau _build_core() hoặc fit() để có hiệu lực.
        """
        self.exog_indices = exog_indices
        if self.core is not None:
            for idx in exog_indices:
                self.core.topo_mask[:, idx] = 0.0
                self.core.constraint_mask[:, idx] = 0.0

    def estimate_ate(self, X: np.ndarray, from_idx: int, to_idx: int) -> float:
        """
        Estimate the ATE of variable from_idx on to_idx via unit do-intervention:
            ATE = E[Y_to | do(X_from + 1)] - E[Y_to | do(X_from)]

        Ước tính ATE của biến from_idx lên to_idx qua can thiệp đơn vị:
            ATE = E[Y_to | do(X_from + 1)] - E[Y_to | do(X_from)]
        """
        self.eval()
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        X_control = X.copy()
        X_treatment = X.copy()
        X_treatment[:, from_idx] += 1.0  # Unit intervention / Can thiệp đơn vị
        return self.core.MLP.estimate_ate(X_control, X_treatment)

    def predict_clusters(self, X):
        """Return the most likely mechanism cluster for each sample.
        Trả về cụm cơ chế có khả năng nhất cho từng mẫu."""
        self.eval()
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        with torch.no_grad():
            out = self.core.MLP(X.to(self.device))
            return out['z_soft'].argmax(dim=1).cpu().numpy()

    def forward(self, x, temperature=1.0):
        """Standard PyTorch forward — delegates to Core engine.
        Lan truyền tiến PyTorch chuẩn — ủy thác cho động cơ Core."""
        return self.core(x, temperature=temperature)
