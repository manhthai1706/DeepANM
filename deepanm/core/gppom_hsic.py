"""
Full-Spectrum Deep Learning GPPOM / Mô hình GPPOM học sâu toàn diện
Integrated kernels and DAG learning / Tích hợp hàm nhân và học DAG
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepanm.core import mlp

class RFFGPLayer(nn.Module):
    """
    Đặc trưng Fourier Ngẫu nhiên (Random Fourier Features - RFF) xấp xỉ Gaussian Kernel.
    Tối giản hóa quy trình chuyển tensor vào GPU/CPU, chống suy hao tốc độ (Bottleneck).
    """
    def __init__(self, input_dim, n_features=256, ARD=False):
        super().__init__()
        self.n_features = n_features
        # Cố định ngẫu nhiên tần số W và góc pha b ngay trong bộ đệm (mặc định sẽ chạy cùng device)
        self.register_buffer("W", torch.randn(input_dim, n_features))
        self.register_buffer("b", torch.rand(n_features) * 2 * np.pi)
        
        self.log_alpha = nn.Parameter(torch.zeros(1))
        # Automatic Relevance Determination (ARD): Chiều nào quan trọng sẽ tự thu phóng riêng biệt
        self.log_gamma = nn.Parameter(torch.zeros(input_dim if ARD else 1))
        
        # Tham số nhân hằng số tính trước, không sử dụng Tensor khai báo động tránh tốn overhead
        self.scale = np.sqrt(2.0 / n_features)

    def forward(self, x):
        gamma = torch.exp(self.log_gamma)
        alpha = torch.exp(self.log_alpha)
        
        # Phép chiếu dữ liệu thu phóng vào không gian ngẫu nhiên
        projection = (x * gamma) @ self.W + self.b
        
        # Ánh xạ cosin biến dữ liệu tĩnh thành vô hạn chiều
        phi = self.scale * torch.cos(projection)
        return phi * torch.sqrt(alpha)

class FastHSIC(nn.Module):
    """
    Kiểm định Độc lập Siêu Tốc (O(d * N) thay vì O(N^2)). 
    Tuyệt đối bắt buộc dùng làm Loss Function cho batch data liên tục.
    Phiên bản này được trang bị ARD để phát hiện chiều phụ thuộc tốt hơn.
    """
    def __init__(self, x_dim, z_dim, n_features=128):
        super().__init__()
        # ARD sẽ giúp HSIC hiểu mô hình nhiễu phức đa chiều
        self.phi_x = RFFGPLayer(x_dim, n_features, ARD=True)
        self.phi_z = RFFGPLayer(z_dim, n_features, ARD=True)

    def forward(self, X, Z):
        n = X.shape[0]
        if n < 2: return torch.tensor(0.0, device=X.device)
        
        # 1. Trích xuất lưới Fourier ngẫu nhiên
        feat_x = self.phi_x(X)
        feat_z = self.phi_z(Z)
        
        # 2. Centering Features (Tương đương việc chuẩn hóa tâm ma trận K nhưng siêu nhanh qua mean)
        feat_x = feat_x - feat_x.mean(dim=0, keepdim=True)
        feat_z = feat_z - feat_z.mean(dim=0, keepdim=True)
        
        # 3. Tính toán Hiệp phương sai chéo C_xz
        covariance = (feat_x.T @ feat_z) / (n - 1)
        
        # 4. Trả về bình phương chuẩn Frobenius (Chính là xấp xỉ tuyến tính của Loss HSIC RBF)
        return torch.sum(covariance**2)

class GPPOMC_lnhsic_Core(nn.Module):
    """DeepANM Core Logic / Logic cốt lõi của DeepANM"""
    def __init__(self, x_dim, y_dim, n_clusters, hidden_dim, lda, device, causal_order=None):
        super().__init__()
        self.lda = lda
        self.device = device
        self.d = x_dim + y_dim

        # --- Gumbel Gate Parameters ---
        self.W_val = nn.Parameter(torch.empty(self.d, self.d).uniform_(-0.01, 0.01))
        self.W_logits = nn.Parameter(torch.full((self.d, self.d), -1.5))

        # --- Topological Mask (Pha 1 RESIT-inspired) ---
        # Nếu có causal_order: chỉ cho phép cạnh đúng hướng tự tiên → hậu (tạo ra Strict Triangular Mask)
        # Nếu không: dùng mask cơ bản (không self-loop)
        if causal_order is not None and len(causal_order) == self.d:
            # position[var] = vị trí trong thứ tự tô-pô
            position = {var: idx for idx, var in enumerate(causal_order)}
            topo = torch.zeros(self.d, self.d)
            for i in range(self.d):
                for j in range(self.d):
                    if i != j and position[i] < position[j]:
                        topo[i, j] = 1.0
            self.register_buffer('topo_mask', topo.to(device))
        else:
            # Fallback: chỉ cấm self-loop (giống cũ)
            self.register_buffer('topo_mask', (1 - torch.eye(self.d)).to(device))

        # Giữ constraint_mask để tương thích ngược với get_dag_penalty
        self.register_buffer('constraint_mask', self.topo_mask.clone())

        # Backbone MLP
        self.MLP = mlp.MLP(input_dim=self.d, hidden_dim=hidden_dim,
                           output_dim=self.d, n_clusters=n_clusters, device=device)

        self.gp_phi_z = RFFGPLayer(n_clusters, n_features=64)
        self.gp_phi_x = RFFGPLayer(self.d, n_features=64)
        self.linear_head = nn.Linear(64, self.d, bias=False)

        self.fast_hsic = FastHSIC(self.d, n_clusters, n_features=64)
        self.pnl_hsic = FastHSIC(self.d, self.d, n_features=64)
    @property
    def W_dag(self):
        """Khôi phục W_dag ảo để tương thích ngược với API bên ngoài (visualize, get_dag_matrix)"""
        return torch.sigmoid(self.W_logits) * self.W_val

    def get_dag_penalty(self, W_mask):
        r"""
        Tối ưu hóa: Thay vì chạy vòng lặp nhân ma trận O(d) lần trong Python, 
        ta giải tích trực tiếp hàm Log-Determinant của DAGMA bằng lõi C++ của PyTorch.
        Tốc độ tính toán nhanh hơn cực kì nhiều lần, không tốn RAM overhead.
        """
        W_dag_masked = W_mask * self.constraint_mask
        A = W_dag_masked * W_dag_masked
        
        I = torch.eye(self.d, device=A.device)
        # Hàm slogdet trả về (sign, logabsdet), ta lấy phần thứ [1] 
        h = -torch.linalg.slogdet(I - A)[1]
        
        return h

    def forward(self, batch_data, temperature=1.0):
        # ----------------------------------------------------
        # GUMBEL-SIGMOID HARD MASKING (Straight-Through Estimator)
        # Giết chết hoàn toàn Cạnh rác (False Positives)
        # ----------------------------------------------------
        # Phân phối xác suất biên
        W_prob = torch.sigmoid(self.W_logits)
        
        if self.training:
            # Tung đồng xu Gumbel (Relaxed Bernoulli)
            U = torch.rand_like(self.W_logits)
            Z = torch.log(U + 1e-10) - torch.log(1 - U + 1e-10)
            W_soft_mask = torch.sigmoid((self.W_logits + Z) / temperature)
            
            # Ép cứng về 0.0 hoặc 1.0 (Hard Mask)
            W_hard_mask = (W_soft_mask > 0.5).float()
            
            # Straight-Through Estimator: Trả về Hard Mask nhưng Gradient đi qua Soft Mask
            W_mask = W_hard_mask.detach() - W_soft_mask.detach() + W_soft_mask
        else:
            # Khi Inference, chỉ những cạnh có xác suất > 50% mới được mở cổng
            W_mask = (W_prob > 0.5).float()

        # Ma trận chung cuộc: Trọng_số_thực_tế * Mask_Nhị_phân * TOPO_MASK (Strict DAG)
        W_dag_masked = self.W_val * W_mask * self.topo_mask
        
        # Mask inputs with DAG matrix / Che đầu vào bằng ma trận DAG (Chắc chắn loại bỏ đường chéo)
        masked_input = batch_data @ W_dag_masked
        
        # Pass qua MLP với dữ liệu đã được DAG-masked
        out = self.MLP(masked_input, temperature=temperature)
        z_soft, kl_loss = out['z_soft'], out['kl_loss']
        
        # Chuyển Z và X qua lưới đặc trưng Gaussian Process (GP) để tìm tương quan ẩn
        phi = self.gp_phi_z(z_soft) * self.gp_phi_x(masked_input) # X đã được filter bởi graph liên hệ
        y_pred_gp = self.linear_head(phi) + out['mu'] # Hợp nhất Nonlinear (GP) + Linear Baseline (mu)
        
        # DAG penalty is added by trainer (ALM loop) — not duplicated here
        
        # Lỗi MSE Hồi Quy Cấu Trúc
        loss_reg = F.mse_loss(y_pred_gp, batch_data)
        
        # Đoạn tính HSIC cho Noise (Kiểm định độc lập: N_y độc lập hoàn toàn với X)
        h_y = self.MLP.pnl_transform(batch_data)  # g(Y)
        res_pnl = h_y - y_pred_gp                 # Nhiễu ước tính N = g(Y) - f(X)
        
        # Ràng buộc độc lập: Hàm HSIC đánh giá độc lập giữa "nhiễu" và "nguyên nhân đầu vào"
        loss_hsic_pnl = self.pnl_hsic(masked_input, res_pnl)
        
        # Ràng buộc cơ chế: Cơ chế Z (z_soft) phải phụ thuộc chặt chẽ với dữ liệu X
        loss_hsic_clu = self.fast_hsic(batch_data, z_soft)
        
        # Sparsity penalty: Ɛp logits xuống âm sâu trong khộ tô-pô đã xác định
        l1_loss = torch.sum(W_prob * self.topo_mask)
        l2_loss = 0.5 * torch.sum(W_dag_masked ** 2)
        
        # Log-Likelihood dựa trên luồng Nhiễu GMM của kiến trúc Causica/DECI
        log_prob = out.get('log_prob_noise', torch.zeros_like(kl_loss))
        loss_nll = -torch.mean(log_prob) 
        
        # Hàm loss gốc tối ưu hóa đa tầng (Multi-objective Augmented Lagrangian Base)
        base_loss = (loss_reg + 
                     loss_nll * 0.015 + 
                     self.lda * loss_hsic_clu + 
                     self.lda * loss_hsic_pnl + 
                     0.02 * l1_loss + # Chuẩn L1 Thưa thớt cạnh
                     0.02 * l2_loss + # Chuẩn L2 Tăng mượt bề mặt hàm Loss (NOTEARS)
                     0.1 * kl_loss) # Ràng buộc VAE
        
        return base_loss, loss_reg, loss_hsic_clu, loss_nll # Return all / Trả về tất cả
