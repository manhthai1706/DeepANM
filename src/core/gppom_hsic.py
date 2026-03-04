"""
DeepANM Core: GPPOM-HSIC Module
Combines Random Fourier Features, fast HSIC independence testing,
Gumbel gate, and DAG penalty into a unified causal learning model.

Mô-đun lõi của DeepANM: GPPOM-HSIC
Kết hợp Random Fourier Features, kiểm định độc lập HSIC nhanh,
cổng Gumbel và phạt DAG thành một mô hình học nhân quả thống nhất.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core import mlp

class RFFGPLayer(nn.Module):
    """
    Approximates Gaussian kernel using Random Fourier Features (Bochner's theorem).
    Reduces complexity from O(N²) to O(N·D) for kernel computations.

    Xấp xỉ Gaussian kernel bằng Random Fourier Features (định lý Bochner).
    Giảm độ phức tạp từ O(N²) xuống O(N·D).
    """
    def __init__(self, input_dim, n_features=256, ARD=False):
        super().__init__()
        self.n_features = n_features
        # Fixed random frequencies W and phase shifts b (not learned, live on device)
        # Tần số ngẫu nhiên W và pha b cố định (không học, tự di chuyển theo device)
        self.register_buffer("W", torch.randn(input_dim, n_features))
        self.register_buffer("b", torch.rand(n_features) * 2 * np.pi)
        
        # Learnable output scale / Hệ số khuếch đại đầu ra (tham số học được)
        self.log_alpha = nn.Parameter(torch.zeros(1))
        # ARD: per-dimension length-scale (if ARD=True, each input dim gets its own scale)
        # ARD: hệ số co giãn riêng từng chiều (nếu ARD=True)
        self.log_gamma = nn.Parameter(torch.zeros(input_dim if ARD else 1))
        
        # Precomputed normalization constant / Hằng số chuẩn hóa tính trước
        self.scale = np.sqrt(2.0 / n_features)

    def forward(self, x):
        gamma = torch.exp(self.log_gamma)
        alpha = torch.exp(self.log_alpha)
        # Project scaled input into random feature space / Chiếu đầu vào vào không gian đặc trưng ngẫu nhiên
        projection = (x * gamma) @ self.W + self.b
        # Cosine mapping approximates infinite-dimensional RBF kernel
        # Ánh xạ cosine xấp xỉ nhân RBF vô chiều
        phi = self.scale * torch.cos(projection)
        return phi * torch.sqrt(alpha)

class FastHSIC(nn.Module):
    """
    Linear-time HSIC independence test using RFF features, O(N·D) complexity.
    Used as a loss term to enforce noise independence in the causal model.

    Kiểm định độc lập HSIC thời gian tuyến tính dùng RFF, độ phức tạp O(N·D).
    Dùng làm hàm mất mát để ép nhiễu độc lập với đầu vào trong mô hình nhân quả.
    """
    def __init__(self, x_dim, z_dim, n_features=128):
        super().__init__()
        # ARD helps detect dependency in each dimension separately
        # ARD giúp phát hiện phụ thuộc riêng từng chiều
        self.phi_x = RFFGPLayer(x_dim, n_features, ARD=True)
        self.phi_z = RFFGPLayer(z_dim, n_features, ARD=True)

    def forward(self, X, Z):
        n = X.shape[0]
        if n < 2: return torch.tensor(0.0, device=X.device)
        
        # Extract RFF embeddings / Trích xuất đặc trưng ngẫu nhiên
        feat_x = self.phi_x(X)
        feat_z = self.phi_z(Z)
        
        # Center features (equivalent to kernel matrix centering)
        # Chuẩn hóa tâm đặc trưng (tương đương chuẩn hóa tâm ma trận kernel)
        feat_x = feat_x - feat_x.mean(dim=0, keepdim=True)
        feat_z = feat_z - feat_z.mean(dim=0, keepdim=True)
        
        # Cross-covariance matrix / Ma trận hiệp phương sai chéo
        covariance = (feat_x.T @ feat_z) / (n - 1)
        
        # Frobenius norm² ≈ linearized HSIC statistic
        # Chuẩn Frobenius² ≈ thống kê HSIC tuyến tính hóa
        return torch.sum(covariance**2)

class GPPOMC_lnhsic_Core(nn.Module):
    """
    Core module of DeepANM. Manages the DAG structure (topo mask, Gumbel gate),
    MLP-based SEM, GP regression head, and all loss components.

    Mô-đun lõi của DeepANM. Quản lý cấu trúc DAG (mặt nạ topo, cổng Gumbel),
    SEM dựa trên MLP, đầu hồi quy GP và tất cả các thành phần mất mát.
    """
    def __init__(self, x_dim, y_dim, n_clusters, hidden_dim, lda, device, causal_order=None, causal_graph=None):
        super().__init__()
        self.lda = lda
        self.device = device
        self.d = x_dim + y_dim

        # --- Edge weight values and soft gate logits / Giá trị trọng số cạnh và logit cổng mềm ---
        self.W_val = nn.Parameter(torch.empty(self.d, self.d).uniform_(-0.01, 0.01))
        self.W_logits = nn.Parameter(torch.full((self.d, self.d), -3.0))

        # --- Topological mask & ALM control / Mặt nạ topo & điều khiển ALM ---
        self.use_alm = True
        
        if causal_graph is not None:
            # Mode 1: Full graph known (from FastANM) → fix structure, disable ALM
            # Chế độ 1: Đồ thị đã biết đầy đủ (từ FastANM) → cố định cấu trúc, tắt ALM
            self.register_buffer('topo_mask', torch.tensor(causal_graph, dtype=torch.float32, device=device))
            self.use_alm = False
            with torch.no_grad():
                self.W_logits.copy_(self.topo_mask * 10.0 - (1 - self.topo_mask) * 10.0)
                self.W_logits.requires_grad = False
                
        elif causal_order is not None and len(causal_order) == self.d:
            # Mode 2: Causal order known → upper-triangular mask, disable ALM
            # Chế độ 2: Thứ tự nhân quả đã biết → mặt nạ tam giác trên, tắt ALM
            position = {var: idx for idx, var in enumerate(causal_order)}
            topo = torch.zeros(self.d, self.d)
            for i in range(self.d):
                for j in range(self.d):
                    if i != j and position[i] < position[j]:
                        topo[i, j] = 1.0
            self.register_buffer('topo_mask', topo.to(device))
            self.use_alm = False
            
        else:
            # Mode 3: No prior knowledge → full mask, enable ALM for free exploration
            # Chế độ 3: Không có tri thức trước → mặt nạ đầy đủ, bật ALM để khám phá tự do
            self.register_buffer('topo_mask', (1 - torch.eye(self.d)).to(device))
            self.use_alm = True

        # Kept for backward compatibility with get_dag_penalty
        # Giữ lại để tương thích ngược với get_dag_penalty
        self.register_buffer('constraint_mask', self.topo_mask.clone())

        # Core SEM neural network / Mạng neural SEM lõi
        self.MLP = mlp.MLP(input_dim=self.d, hidden_dim=hidden_dim,
                           output_dim=self.d, n_clusters=n_clusters, device=device)

        # GP regression heads for x and z / Đầu hồi quy GP cho x và z
        self.gp_phi_z = RFFGPLayer(n_clusters, n_features=64)
        self.gp_phi_x = RFFGPLayer(self.d, n_features=64)
        self.linear_head = nn.Linear(64, self.d, bias=False)

        # HSIC independence testers / Bộ kiểm định độc lập HSIC
        self.fast_hsic = FastHSIC(self.d, n_clusters, n_features=64)
        self.pnl_hsic = FastHSIC(self.d, self.d, n_features=64)

    @property
    def W_dag(self):
        """Reconstructs W_dag for external API compatibility (visualize, get_dag_matrix).
        Tái tạo W_dag để tương thích với API bên ngoài (visualize, get_dag_matrix)."""
        return torch.sigmoid(self.W_logits) * self.W_val

    def get_dag_penalty(self, W_mask):
        """
        DAGMA DAG acyclicity penalty: h(W) = -log det(I - W²).
        h(W) = 0 iff W is a DAG. Uses PyTorch's slogdet for efficiency.

        Phạt tính phi chu trình DAGMA: h(W) = -log det(I - W²).
        h(W) = 0 khi và chỉ khi W là DAG. Dùng slogdet của PyTorch để hiệu quả.
        """
        W_dag_masked = W_mask * self.constraint_mask
        A = W_dag_masked * W_dag_masked
        I = torch.eye(self.d, device=A.device)
        # slogdet returns (sign, log|det|); we take index [1]
        # slogdet trả về (dấu, log|det|); lấy phần tử [1]
        h = -torch.linalg.slogdet(I - A)[1]
        return h

    def forward(self, batch_data, temperature=1.0):
        # --- Gumbel-Sigmoid Hard Masking (Straight-Through Estimator) ---
        # Stochastic binary gate during training; deterministic at inference
        # Cổng nhị phân ngẫu nhiên lúc huấn luyện; tất định lúc suy diễn
        W_prob = torch.sigmoid(self.W_logits)
        
        if self.training:
            # Sample Gumbel noise for Relaxed Bernoulli / Lấy mẫu nhiễu Gumbel (Bernoulli nới lỏng)
            U = torch.rand_like(self.W_logits)
            Z = torch.log(U + 1e-10) - torch.log(1 - U + 1e-10)
            W_soft_mask = torch.sigmoid((self.W_logits + Z) / temperature)
            # Hard binarization / Nhị phân hóa cứng
            W_hard_mask = (W_soft_mask > 0.5).float()
            # Straight-Through: forward uses hard mask, backward passes soft gradient
            # Straight-Through: tiến dùng mặt nạ cứng, lùi truyền gradient mềm
            W_mask = W_hard_mask.detach() - W_soft_mask.detach() + W_soft_mask
        else:
            # At inference: keep edges with gate probability > 50%
            # Lúc suy diễn: giữ cạnh có xác suất cổng > 50%
            W_mask = (W_prob > 0.5).float()

        # Final weight matrix: values × binary gate × topological mask
        # Ma trận trọng số cuối: giá trị × cổng nhị phân × mặt nạ topo
        W_dag_masked = self.W_val * W_mask * self.topo_mask
        
        # DAG-masked input: each variable receives only its allowed parents
        # Đầu vào DAG-masked: mỗi biến chỉ nhận từ các cha được phép
        masked_input = batch_data @ W_dag_masked
        
        # Forward through MLP / Lan truyền qua MLP
        out = self.MLP(masked_input, temperature=temperature)
        z_soft, kl_loss = out['z_soft'], out['kl_loss']
        
        # GP regression: combine nonlinear (GP) + linear (mu) predictions
        # Hồi quy GP: kết hợp dự đoán phi tuyến (GP) + tuyến tính (mu)
        phi = self.gp_phi_z(z_soft) * self.gp_phi_x(masked_input)
        y_pred_gp = self.linear_head(phi) + out['mu']

        # Structural regression loss (MSE) / Mất mát hồi quy cấu trúc (MSE)
        loss_reg = F.mse_loss(y_pred_gp, batch_data)
        
        # PNL noise independence: HSIC(residual, causes) should be 0
        # Độc lập nhiễu PNL: HSIC(phần dư, nguyên nhân) phải = 0
        h_y = self.MLP.pnl_transform(batch_data)
        res_pnl = h_y - y_pred_gp
        loss_hsic_pnl = self.pnl_hsic(masked_input, res_pnl)
        
        # Mechanism clustering: z must correlate with X
        # Phân cụm cơ chế: z phải tương quan với X
        loss_hsic_clu = self.fast_hsic(batch_data, z_soft)
        
        # Sparsity regularization / Điều chuẩn độ thưa
        l1_loss = torch.sum(W_prob * self.topo_mask)
        l2_loss = 0.5 * torch.sum(W_dag_masked ** 2)
        
        # GMM noise log-likelihood / Log-likelihood nhiễu GMM
        log_prob = out.get('log_prob_noise', torch.zeros_like(kl_loss))
        loss_nll = -torch.mean(log_prob) 
        
        # Total multi-objective loss / Tổng mất mát đa mục tiêu
        base_loss = (loss_reg + 
                     loss_nll * 0.1 + 
                     self.lda * loss_hsic_clu + 
                     self.lda * loss_hsic_pnl + 
                     0.1 * l1_loss +
                     0.02 * l2_loss +
                     0.1 * kl_loss)
        
        return base_loss, loss_reg, loss_hsic_clu, loss_nll
