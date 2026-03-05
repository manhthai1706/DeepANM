"""
DeepANM: Deep Additive Noise Model for Causal Discovery.
Integrated Three-Phase Pipeline: 
  - Phase 1: Topological Sort (Causal Order)
  - Phase 2: Graph Selection (Non-linear LASSO)
  - Phase 3: Neural Fitting (Mechanism Learning & ATE Estimation)

Default configuration uses lean/lite settings (n_clusters=1, hidden_dim=16)
for optimal speed-accuracy tradeoff on real-world datasets.

DeepANM: Mô hình Nhiễu Cộng Sâu cho Khám phá Nhân quả.
Pipeline Tích hợp Ba Pha:
  - Pha 1: Sắp xếp Topo (Xác định thứ tự nhân quả)
  - Pha 2: Chọn Đồ thị (LASSO phi tuyến)
  - Pha 3: Khớp Neural (Học cơ chế & Ước tính ATE)

Cấu hình mặc định dùng thông số tinh gọn (n_clusters=1, hidden_dim=16)
để đạt cân bằng tối ưu giữa tốc độ và độ chính xác.
"""

import numpy as np
import torch
import torch.nn as nn


def _order_from_topo_mask(topo_mask: np.ndarray, n_vars: int) -> list:
    """
    Reconstruct causal order from a binary DAG mask using Kahn's topological sort.
    Khôi phục thứ tự nhân quả từ mặt nạ DAG nhị phân dùng thuật toán Kahn.
    """
    in_deg = {j: 0 for j in range(n_vars)} # Track incoming edges / Theo dõi số cạnh đi vào
    children = {i: [] for i in range(n_vars)} # Track outgoing edges / Theo dõi cạnh đi ra
    for i in range(n_vars):
        for j in range(n_vars):
            if topo_mask[i, j] > 0.5:
                children[i].append(j)
                in_deg[j] += 1

    # Start with nodes having zero in-degree (roots) / Bắt đầu từ node không có cạnh vào (nút gốc)
    queue = [v for v in range(n_vars) if in_deg[v] == 0]
    order = []
    while queue:
        v = queue.pop(0)
        order.append(v)
        for u in children[v]:
            in_deg[u] -= 1
            if in_deg[u] == 0:
                queue.append(u)

    # Return order if valid; fallback otherwise / Trả về thứ tự nếu hợp lệ; nếu không dùng mặc định
    return order if len(order) == n_vars else list(range(n_vars))

class DeepANM(nn.Module):
    """
    DeepANM Implementation for Multivariate Nonlinear Causal Discovery.
    Cấu trúc mạng neural cho bài toán khám phá quan hệ nhân quả đa biến phi tuyến.

    Supported Discovery Modes / Các chế độ khám phá:
    - 'fast' : Hybrid mode using FastANM for structure + Neural for ATE. / Chế độ lai: dùng FastANM tìm cấu trúc + Neural tính ATE.
    - 'topo' : Constrains ordering via TopoSort but lets Neural find links. / Ràng buộc thứ tự bằng TopoSort nhưng để Neural tìm liên kết.
    - 'alm'  : Pure deep learning exploration via Augmented Lagrangian. / Khám phá thuần học sâu qua Lagrangian Tăng cường.
    """

    def __init__(self, x_dim=None, n_clusters=1, hidden_dim=16, lda=0.5, device=None):
        """Initialize hyperparameters and internal components. / Khởi tạo siêu tham số và các thành phần nội bộ."""
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.lda = lda # HSIC independence loss weight / Trọng số mất mát HSIC
        self.x_dim = x_dim
        self.n_clusters = n_clusters # Number of latent noise mechanisms / Số lượng cơ chế nhiễu ẩn
        self.hidden_dim = hidden_dim # Hidden layer size / Kích thước lớp ẩn
        self.core = None   # Initialized during fit() / Khởi tạo trong quá trình fit()
        self.history = None

        if x_dim is not None:
            # Pre-warm core with default structure / Khởi động lõi với cấu trúc mặc định
            self._build_core(x_dim, X=None, causal_order=None, verbose=False)

    # ------------------------------------------------------------------
    # Internal Logic / Logic nội bộ
    # ------------------------------------------------------------------

    def _build_core(self, x_dim, X=None, causal_order=None, causal_graph=None, verbose=True):
        """
        Builds the GPPOM Core engine with dynamic structural constraints.
        Xây dựng động cơ lõi GPPOM với các ràng buộc cấu trúc động.
        """
        from src.core.gppom_hsic import GPPOMC_lnhsic_Core
        self.x_dim = x_dim

        if causal_graph is not None:
            # Fixed adjacency provided / Có sẵn ma trận kề cố định
            order = None 
        elif causal_order is not None:
            # Order provided, skip Phase 1 / Có sẵn thứ tự, bỏ qua Pha 1
            order = causal_order
            if verbose: print(f"[Discovery] Using cached TopoSort order. ALM Disabled.")
        elif X is not None:
            # Discover order via HSIC Greedy TopSort / Khám phá thứ tự qua HSIC Greedy TopSort
            from src.core.toposort import hsic_greedy_order
            if verbose: print("[Discovery] Discovering TopoSort order (Sink-First)... ALM Disabled.")
            order = hsic_greedy_order(X, verbose=False)
        else:
            # No prior info, full ALM exploration / Không có thông tin trước, khám phá tự do bằng ALM
            order = None
            if verbose: print("[Discovery] No prior knowledge. ALM Enabled.")

        # Construct the core processing unit / Xây dựng bộ xử lý lõi
        self.core = GPPOMC_lnhsic_Core(
            x_dim, 0, self.n_clusters, self.hidden_dim, self.lda,
            self.device, causal_order=order, causal_graph=causal_graph
        )
        self.to(self.device)

    def _preprocess(self, X, apply_isolation=False, apply_quantile=False):
        """Optional data cleaning and normalization. / Tiền xử lý dữ liệu tùy chọn."""
        if apply_isolation:
            from sklearn.ensemble import IsolationForest
            mask = IsolationForest(contamination=0.05, random_state=42).fit_predict(X) == 1
            X = X[mask]
        if apply_quantile:
            from sklearn.preprocessing import QuantileTransformer
            X = QuantileTransformer(output_distribution='normal').fit_transform(X)
        return X

    # ------------------------------------------------------------------
    # High-level API / Giao diện người dùng
    # ------------------------------------------------------------------

    def fit(self, X, epochs=50, batch_size=128, lr=5e-3, verbose=True,
            apply_quantile=False, apply_isolation=False,
            discovery_mode="fast", layer_constraint=None, _precomputed_order=None,
            use_rf=True, use_ci_pruning=True):
        """
        Main entry point for model training.
        Điểm bắt đầu huấn luyện mô hình chính.

        Modes:
        - 'fast': Highly recommended for structure accuracy. / Khuyên dùng để đảm bảo cấu trúc chính xác.
        """
        from src.utils.trainer import DeepANMTrainer
        X_p = self._preprocess(X, apply_isolation=apply_isolation, apply_quantile=apply_quantile)
        self.discovery_mode = discovery_mode

        causal_graph = None
        causal_order = _precomputed_order
        X_init = None
        
        if discovery_mode == "fast":
            # Run Phase 1+2 (FastANM) to fix the binary structure / Chạy Pha 1+2 tìm cấu trúc nhị phân
            from src.models.fast_baseline import FastANM
            if verbose: print("\n[DeepANM] Phase 1+2: FastANM Graph Discovery...")
            fast_model = FastANM()
            causal_graph = fast_model.fit(X, apply_quantile=apply_quantile, apply_isolation=apply_isolation,
                                          verbose=verbose, layer_constraint=layer_constraint,
                                          use_rf=use_rf, use_ci_pruning=use_ci_pruning)
            if verbose: print("\n[DeepANM] Phase 3: SCM Neural Fitting (Refining effects)...")
        elif discovery_mode == "topo":
            X_init = X_p if causal_order is None else None # Prepare for TopoSort / Chuẩn bị chạy TopoSort
        elif discovery_mode == "alm":
            causal_order = None # Full ALM: no topological ordering prior / Khám phá tự do bằng ALM
            
        # 1. Build the network architecture / Xây dựng kiến trúc mạng
        self._build_core(X_p.shape[1], X=X_init,
                         causal_order=causal_order, causal_graph=causal_graph, verbose=verbose)

        # 2. Run the actual training loop / Tiến hành vòng lặp huấn luyện thực tế
        trainer = DeepANMTrainer(self, lr=lr)
        self.history = trainer.train(X_p, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return self.history

    def fit_bootstrap(self, X, n_bootstraps=5,
                      epochs=50, batch_size=128, lr=5e-3, verbose=True,
                      apply_quantile=False, apply_isolation=False, discovery_mode="fast",
                      layer_constraint=None, use_rf=True, use_ci_pruning=True, use_scm_filter=True):
        """
        Stability Selection using resampled data batches. 
        Highly recommended to eliminate spurious edges in small datasets.
        Lựa chọn ổn định qua lấy mẫu lại dữ liệu. Rất khuyên dùng để loại bỏ cạnh giả.
        """
        if isinstance(X, torch.Tensor): X = X.cpu().numpy()
        X = self._preprocess(X, apply_isolation=apply_isolation, apply_quantile=apply_quantile)

        n_samples, n_vars = X.shape
        agg_W = np.zeros((n_vars, n_vars))
        agg_bin = np.zeros((n_vars, n_vars))

        # Optimization: run TopoSort once on full data for consistency / Chạy TopoSort một lần để nhất quán
        self._causal_order = None
        if discovery_mode == "topo":
            from src.core.toposort import hsic_greedy_order
            if verbose: print("[TopoSort] Running ONCE on full dataset for stable ordering...")
            self._causal_order = hsic_greedy_order(X, verbose=False)

        for b in range(n_bootstraps):
            if verbose: print(f"[Bootstrap] Round {b+1}/{n_bootstraps}...")
            # Sample with replacement / Lấy mẫu có lặp lại
            boot_data = X[np.random.choice(n_samples, n_samples, replace=True)]
            self.core = None # Reset core for each round / Làm mới lõi mỗi vòng lặp
            self.fit(boot_data, epochs=epochs, batch_size=batch_size, lr=lr,
                     verbose=False, discovery_mode=discovery_mode,
                     layer_constraint=layer_constraint, _precomputed_order=self._causal_order,
                     use_rf=use_rf, use_ci_pruning=use_ci_pruning)

            # Accumulate discovered edges and their weights / Tích lũy các cạnh và trọng số đã khám phá
            ATE, W_bin = self.get_dag_matrix(X=boot_data, use_scm_filter=use_scm_filter)
            agg_W   += ATE
            agg_bin += W_bin

        # Data-driven edge probability / Xác suất cạnh dựa trên dữ liệu
        return agg_bin / n_bootstraps, agg_W / n_bootstraps

    def get_dag_matrix(self, X=None, use_scm_filter=True):
        """
        Retrieves the learned DAG structure and causal effect matrix.
        Trích xuất cấu trúc DAG đã học và ma trận tác động nhân quả.
        """
        with torch.no_grad():
            W_dag_masked = self.core.W_dag * self.core.topo_mask
            W = W_dag_masked.detach().cpu().numpy()

            if X is not None:
                # 1. Estimate Direct Effects via Neural Jacobian / Ước lượng tác động trực tiếp qua Jacobian Neural
                ATE = self.core.MLP.get_global_ate_matrix(X, W_dag=W_dag_masked).cpu().numpy()

                if getattr(self, 'discovery_mode', None) == "fast":
                    W_lasso = self.core.topo_mask.cpu().numpy()
                    if not use_scm_filter: return ATE, W_lasso
                    
                    # Phase 3 Refinement: Adaptive ATE Gate (prunes weak causal signals)
                    # Tinh lọc Pha 3: Cổng ATE Thích nghi (lọc bỏ các tín hiệu nhân quả yếu)
                    ate_abs = np.abs(ATE)
                    edge_ates = ate_abs[W_lasso > 0.5]
                    if len(edge_ates) > 0 and edge_ates.max() > 0:
                        adaptive_threshold = np.percentile(edge_ates, 15) # Remove bottom 15% / Lọc bỏ 15% yếu nhất
                        ATE_gate = (ate_abs > adaptive_threshold).astype(float)
                    else:
                        ATE_gate = (ate_abs > 1e-4).astype(float)
                    return ATE, W_lasso * ATE_gate
                else:
                    # Generic mode link selection / Lựa chọn liên kết chế độ chung
                    causal_order = _order_from_topo_mask(self.core.topo_mask.cpu().numpy(), self.x_dim)
                    from src.utils.adaptive_lasso import adaptive_lasso_from_ate
                    W_bin = adaptive_lasso_from_ate(ATE, X, causal_order)
                    return ATE, W_bin
            else:
                return W, (torch.sigmoid(self.core.W_logits).detach().cpu().numpy() > 0.5).astype(float) * self.core.topo_mask.cpu().numpy()

    def set_exogenous(self, exog_indices):
        """Utility to enforce specific nodes as root causes (no parents allowed). / Ép các nút trở thành nguyên nhân gốc."""
        self.exog_indices = exog_indices
        if self.core is not None:
            for idx in exog_indices:
                self.core.topo_mask[:, idx] = 0.0 # Clear all incoming columns / Xóa toàn bộ cột cạnh đi vào
                self.core.constraint_mask[:, idx] = 0.0

    def estimate_ate(self, X: np.ndarray, from_idx: int, to_idx: int) -> float:
        """Computes intervention effect: how much Y changes if X is shifted. / Tính toán tác động can thiệp."""
        self.eval()
        if isinstance(X, torch.Tensor): X = X.cpu().numpy()
        X_control = X.copy()
        X_treatment = X.copy()
        X_treatment[:, from_idx] += 1.0 # Simulate hypothetical intervention / Mô phỏng can thiệp giả định
        return self.core.MLP.estimate_ate(X_control, X_treatment)

    def predict_clusters(self, X):
        """Identify which latent causal mechanism governs each sample. / Xác định cơ chế nhân quả ẩn ứng với mỗi mẫu."""
        self.eval()
        if isinstance(X, np.ndarray): X = torch.from_numpy(X).float()
        with torch.no_grad():
            out = self.core.MLP(X.to(self.device))
            return out['z_soft'].argmax(dim=1).cpu().numpy() # Multi-mechanism switching / Chuyển đổi đa cơ chế

    def forward(self, x, temperature=1.0):
        """Routing forward pass to Core engine. / Điều hướng lan truyền tiến tới động cơ Core."""
        return self.core(x, temperature=temperature)
