"""
DeepANM Core: GPPOM-HSIC Module
Combines Random Fourier Features, fast HSIC independence testing,
Gumbel gate, and DAG penalty into a unified causal learning model.

Mô-đun lõi của DeepANM: GPPOM-HSIC
Kết hợp RFF, kiểm định độc lập HSIC nhanh, cổng Gumbel và phạt DAG.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core import mlp

class RFFGPLayer(nn.Module):
    """
    Approximates kernel mappings using Random Fourier Features (Bochner's theorem).
    Xấp xỉ ánh xạ nhân dùng RFF (định lý Bochner), biến kernel vô chiều thành hữu hạn chiều.
    """
    def __init__(self, input_dim, n_features=256, ARD=False):
        super().__init__()
        self.n_features = n_features
        # Static buffers for frequency and phase / Bộ đệm tĩnh cho tần số và pha
        self.register_buffer("W", torch.randn(input_dim, n_features))
        self.register_buffer("b", torch.rand(n_features) * 2 * np.pi)
        
        # Learnable scale parameters / Các thông số co giãn có thể học được
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.log_gamma = nn.Parameter(torch.zeros(input_dim if ARD else 1))
        
        self.scale = np.sqrt(2.0 / n_features)

    def forward(self, x):
        gamma = torch.exp(self.log_gamma)
        alpha = torch.exp(self.log_alpha)
        # Random projection into high-dimensional space / Chiếu ngẫu nhiên vào không gian cao chiều
        projection = (x * gamma) @ self.W + self.b
        phi = self.scale * torch.cos(projection) # Cosine encoding / Mã hóa Cosine
        return phi * torch.sqrt(alpha)

class FastHSIC(nn.Module):
    """
    Linear-time HSIC dependency estimation via random feature cross-covariance.
    Ước lượng phụ thuộc HSIC thời gian tuyến tính qua hiệp phương sai chéo đặc trưng ngẫu nhiên.
    """
    def __init__(self, x_dim, z_dim, n_features=128):
        super().__init__()
        # ARD mapping for both input and noise proxy / Ánh xạ ARD cho cả đầu vào và nhiễu đại diện
        self.phi_x = RFFGPLayer(x_dim, n_features, ARD=True)
        self.phi_z = RFFGPLayer(z_dim, n_features, ARD=True)

    def forward(self, X, Z):
        n = X.shape[0]
        if n < 2: return torch.tensor(0.0, device=X.device)
        
        feat_x = self.phi_x(X)
        feat_z = self.phi_z(Z)
        
        # Center the features for kernel covariance / Chuẩn hóa tâm đặc trưng cho hiệp phương sai nhân
        feat_x = feat_x - feat_x.mean(dim=0, keepdim=True)
        feat_z = feat_z - feat_z.mean(dim=0, keepdim=True)
        
        covariance = (feat_x.T @ feat_z) / (n - 1)
        return torch.sum(covariance**2) # Squared Frobenius Norm / Chuẩn Frobenius bình phương

class GPPOMC_lnhsic_Core(nn.Module):
    """
    Main causal engine: orchestrates graph weight learning, ALM constraints, and structural losses.
    Động cơ nhân quả chính: điều phối học trọng số đồ thị, ràng buộc ALM và tổn thất cấu trúc.
    """
    def __init__(self, x_dim, y_dim, n_clusters, hidden_dim, lda, device, causal_order=None, causal_graph=None):
        super().__init__()
        self.lda = lda
        self.device = device
        self.d = x_dim + y_dim

        # Adjacency parameters / Tham số ma trận kề
        self.W_val = nn.Parameter(torch.empty(self.d, self.d).uniform_(-0.01, 0.01))
        self.W_logits = nn.Parameter(torch.full((self.d, self.d), -3.0))

        # Handle structural constraints / Xử lý các ràng buộc cấu trúc
        self.use_alm = True
        
        if causal_graph is not None:
            # Fixed structure from FastANM / Cấu trúc cố định từ FastANM
            self.register_buffer('topo_mask', torch.tensor(causal_graph, dtype=torch.float32, device=device))
            self.use_alm = False
            with torch.no_grad():
                # Fix gate logits to high/low / Cố định logit cổng ở mức cao/thấp
                self.W_logits.copy_(self.topo_mask * 10.0 - (1 - self.topo_mask) * 10.0)
                self.W_logits.requires_grad = False
                
        elif causal_order is not None and len(causal_order) == self.d:
            # Order constraint / Ràng buộc thứ tự
            position = {var: idx for idx, var in enumerate(causal_order)}
            topo = torch.zeros(self.d, self.d)
            for i in range(self.d):
                for j in range(self.d):
                    if i != j and position[i] < position[j]:
                        topo[i, j] = 1.0
            self.register_buffer('topo_mask', topo.to(device))
            self.use_alm = False
            
        else:
            # Free discovery / Khám phá tự do
            self.register_buffer('topo_mask', (1 - torch.eye(self.d)).to(device))
            self.use_alm = True

        self.register_buffer('constraint_mask', self.topo_mask.clone())

        # Integrated SEM MLP / Mạng SEM MLP tích hợp
        self.MLP = mlp.MLP(input_dim=self.d, hidden_dim=hidden_dim,
                           output_dim=self.d, n_clusters=n_clusters, device=device)

        # Regression and Independence testers / Các bộ hồi quy và kiểm định độc lập
        self.gp_phi_z = RFFGPLayer(n_clusters, n_features=64)
        self.gp_phi_x = RFFGPLayer(self.d, n_features=64)
        self.linear_head = nn.Linear(64, self.d, bias=False)

        self.fast_hsic = FastHSIC(self.d, n_clusters, n_features=64)
        self.pnl_hsic = FastHSIC(self.d, self.d, n_features=64)

    @property
    def W_dag(self):
        """Effective graph weights / Trọng số đồ thị thực dụng."""
        return torch.sigmoid(self.W_logits) * self.W_val

    def get_dag_penalty(self, W_mask):
        """DAGMA acyclicity penalty h(W). / Hàm phạt tính chu trình DAGMA h(W)."""
        W_dag_masked = W_mask * self.constraint_mask
        A = W_dag_masked * W_dag_masked
        I = torch.eye(self.d, device=A.device)
        h = -torch.linalg.slogdet(I - A)[1] # Log-determinant DAG penalty / Phạt DAG qua Log-determinant
        return h

    def forward(self, batch_data, temperature=1.0):
        # 1. Gumbel Gate: probabilistic edge sampling / Cổng Gumbel: lấy mẫu cạnh xác suất
        W_prob = torch.sigmoid(self.W_logits)
        if self.training:
            U = torch.rand_like(self.W_logits)
            Z = torch.log(U + 1e-10) - torch.log(1 - U + 1e-10)
            W_soft_mask = torch.sigmoid((self.W_logits + Z) / temperature)
            W_hard_mask = (W_soft_mask > 0.5).float()
            # Straight-through gradient pass / Truyền gradient qua (ST estimator)
            W_mask = W_hard_mask.detach() - W_soft_mask.detach() + W_soft_mask
        else:
            W_mask = (W_prob > 0.5).float()

        W_dag_masked = self.W_val * W_mask * self.topo_mask
        masked_input = batch_data @ W_dag_masked
        
        # 2. Causality and Mechansim inference / Suy luận nhân quả và cơ chế
        out = self.MLP(masked_input, temperature=temperature)
        z_soft, kl_loss = out['z_soft'], out['kl_loss']
        
        # 3. Predict outcome using hybrid GP-MLP head / Dự đoán kết quả dùng đầu lai GP-MLP
        phi = self.gp_phi_z(z_soft) * self.gp_phi_x(masked_input)
        y_pred = self.linear_head(phi) + out['mu']

        # 4. Losses / Các hàm tổn thất
        loss_reg = F.mse_loss(y_pred, batch_data) # Reconstruction MSE / MSE tái cấu trúc
        
        # Independence Loss: Res ⊥ Causes / Tổn thất độc lập: Phần dư ⊥ Nguyên nhân
        res_pnl = self.MLP.pnl_transform(batch_data) - y_pred
        loss_hsic_pnl = self.pnl_hsic(masked_input, res_pnl)
        
        # Clustering Loss (Mechanism sensitivity) / Tổn thất phân cụm (độ nhạy cơ chế)
        loss_hsic_clu = self.fast_hsic(batch_data, z_soft)
        
        l1_loss = torch.sum(W_prob * self.topo_mask) # Sparsity / Độ thưa
        l2_loss = 0.5 * torch.sum(W_dag_masked ** 2)
        
        log_prob = out.get('log_prob_noise', torch.zeros_like(kl_loss))
        loss_nll = -torch.mean(log_prob) # GMM Noise likelihood / Khả năng xảy ra nhiễu GMM
        
        # Combined multi-objective optimization / Tối ưu hóa đa mục tiêu kết hợp
        base_loss = (loss_reg + 
                     loss_nll * 0.1 + 
                     self.lda * loss_hsic_clu + 
                     self.lda * loss_hsic_pnl + 
                     0.1 * l1_loss +
                     0.02 * l2_loss +
                     0.1 * kl_loss)
        
        return base_loss, loss_reg, loss_hsic_clu, loss_nll
