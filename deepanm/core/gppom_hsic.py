"""
Full-Spectrum Deep Learning GPPOM / Mô hình GPPOM học sâu toàn diện
Integrated kernels and DAG learning / Tích hợp hàm nhân và học DAG
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from deepanm.core.kernels import RBFKernel
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
    def __init__(self, x_dim, y_dim, n_clusters, hidden_dim, lda, device):
        super().__init__()
        self.lda = lda # HSIC weight / Trọng số HSIC
        self.device = device # Device / Thiết bị
        self.d = x_dim + y_dim # Total vars / Tổng số biến
        
        # Learnable Adjacency (DAG NOTEARS) / Ma trận kề học được, khởi tạo phân phối U(-0.01, 0.01) để phá vỡ tính đối xứng
        self.W_dag = nn.Parameter(torch.empty(self.d, self.d).uniform_(-0.01, 0.01))
        
        # Mask bảo vệ cấu trúc (Cấm Self-loop và cho phép thêm luật Exogenous Prior)
        self.register_buffer('constraint_mask', 1 - torch.eye(self.d, device=device))
        
        # Backbone MLP / Khung mạng MLP
        self.MLP = mlp.MLP(input_dim=self.d, hidden_dim=hidden_dim, 
                          output_dim=self.d, n_clusters=n_clusters, device=device)
        
        self.gp_phi_z = RFFGPLayer(n_clusters, n_features=64) # Z mapping / Ánh xạ Z siêu nhẹ
        self.gp_phi_x = RFFGPLayer(self.d, n_features=64) # X mapping / Ánh xạ X siêu nhẹ
        self.linear_head = nn.Linear(64, self.d, bias=False) # Head / Đầu ra tuyến tính
        
        self.fast_hsic = FastHSIC(self.d, n_clusters, n_features=64) # Cluster HSIC / HSIC phân cụm
        self.pnl_hsic = FastHSIC(self.d, self.d, n_features=64) # PNL noise HSIC / HSIC nhiễu PNL

    def get_dag_penalty(self):
        r"""
        Tối ưu hóa: Thay vì chạy vòng lặp nhân ma trận O(d) lần trong Python, 
        ta giải tích trực tiếp hàm Log-Determinant của DAGMA bằng lõi C++ của PyTorch.
        Tốc độ tính toán nhanh hơn cực kì nhiều lần, không tốn RAM overhead.
        """
        W_dag_masked = self.W_dag * self.constraint_mask
        A = W_dag_masked * W_dag_masked
        
        I = torch.eye(self.d, device=A.device)
        # Hàm slogdet trả về (sign, logabsdet), ta lấy phần thứ [1] 
        h = -torch.linalg.slogdet(I - A)[1]
        
        return h

    def forward(self, batch_data, temperature=1.0):
        # Thiết luật A_ii = 0 (Bắt buộc không có Self-Loop cho DAG) và Cấm ngược chiểu vào Exogenous
        W_dag_masked = self.W_dag * self.constraint_mask
        
        # Mask inputs with DAG matrix / Che đầu vào bằng ma trận DAG (Chắc chắn loại bỏ đường chéo)
        masked_input = batch_data @ W_dag_masked
        
        # Pass qua MLP với dữ liệu đã được DAG-masked
        out = self.MLP(masked_input, temperature=temperature)
        z_soft, kl_loss = out['z_soft'], out['kl_loss']
        
        # Chuyển Z và X qua lưới đặc trưng Gaussian Process (GP) để tìm tương quan ẩn
        phi = self.gp_phi_z(z_soft) * self.gp_phi_x(masked_input) # X đã được filter bởi graph liên hệ
        y_pred_gp = self.linear_head(phi) + out['mu'] # Hợp nhất Nonlinear (GP) + Linear Baseline (mu)
        
        loss_dag = self.get_dag_penalty()
        
        # Lỗi MSE Hồi Quy Cấu Trúc
        loss_reg = F.mse_loss(y_pred_gp, batch_data)
        
        # Đoạn tính HSIC cho Noise (Kiểm định độc lập: N_y độc lập hoàn toàn với X)
        h_y = self.MLP.pnl_transform(batch_data)  # g(Y)
        res_pnl = h_y - y_pred_gp                 # Nhiễu ước tính N = g(Y) - f(X)
        
        # Ràng buộc độc lập: Hàm HSIC đánh giá độc lập giữa "nhiễu" và "nguyên nhân đầu vào"
        loss_hsic_pnl = self.pnl_hsic(masked_input, res_pnl)
        
        # Ràng buộc cơ chế: Cơ chế Z (z_soft) phải phụ thuộc chặt chẽ với dữ liệu X
        loss_hsic_clu = self.fast_hsic(batch_data, z_soft)
        
        # L1 Regularization: Giải thuật LASSO làm cho ma trận DAG cực kì thưa thớt (Sparse) 
        # L2 Regularization: Đóng góp của NOTEARS, chống quá ngưỡng Gradient của từng trọng số đơn
        l1_loss = torch.sum(torch.abs(W_dag_masked))
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
