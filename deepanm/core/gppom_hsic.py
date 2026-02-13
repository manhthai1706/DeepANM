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
    """Random Fourier Features for GP / Đặc trưng Fourier ngẫu nhiên cho GP"""
    def __init__(self, input_dim, n_features=256):
        super().__init__()
        self.n_features = n_features # Num features / Số lượng đặc trưng
        # Fixed random weights / Trọng số ngẫu nhiên cố định
        self.register_buffer("W", torch.randn(input_dim, n_features))
        # Fixed random bias / Độ chệch ngẫu nhiên cố định
        self.register_buffer("b", torch.rand(n_features) * 2 * np.pi)
        self.log_alpha = nn.Parameter(torch.zeros(1)) # Scale param / Tham số tỉ lệ
        self.log_gamma = nn.Parameter(torch.zeros(1)) # Bandwidth param / Tham số độ rộng

    def forward(self, x):
        gamma = torch.exp(self.log_gamma) # Bandwidth / Độ rộng
        alpha = torch.exp(self.log_alpha) # Scale / Tỉ lệ
        projection = (x * gamma) @ self.W + self.b # Random projection / Chiếu ngẫu nhiên
        # Cosine transform (RFF formula) / Biến đổi cosin (Công thức RFF)
        phi = torch.sqrt(torch.tensor(2.0 / self.n_features)) * torch.cos(projection)
        return phi * torch.sqrt(alpha) # Final features / Đặc trưng cuối cùng

class FastHSIC(nn.Module):
    """O(N) Fast HSIC using RFF / HSIC tốc độ cao O(N) thông qua RFF"""
    def __init__(self, x_dim, z_dim, n_features=128):
        super().__init__()
        self.phi_x = RFFGPLayer(x_dim, n_features) # Features for X / Đặc trưng cho X
        self.phi_z = RFFGPLayer(z_dim, n_features) # Features for Z / Đặc trưng cho Z

    def forward(self, X, Z):
        n = X.shape[0] # Num samples / Số lượng mẫu
        if n < 2: return torch.tensor(0.0, device=X.device) # Fallback / Trường hợp lỗi
        
        feat_x = self.phi_x(X) # X features / Đặc trưng X
        feat_z = self.phi_z(Z) # Z features / Đặc trưng Z
        
        # Center features / Chuẩn hóa tâm đặc trưng
        feat_x = feat_x - feat_x.mean(dim=0, keepdim=True)
        feat_z = feat_z - feat_z.mean(dim=0, keepdim=True)
        
        # Matrix covariance sum / Tổng hiệp phương sai ma trận
        covariance = (feat_x.T @ feat_z) / (n - 1)
        return torch.sum(covariance**2) # HSIC proxy / Chỉ số HSIC đại diện

class GPPOMC_lnhsic_Core(nn.Module):
    """DeepANM Core Logic / Logic cốt lõi của DeepANM"""
    def __init__(self, x_dim, y_dim, n_clusters, hidden_dim, lda, device):
        super().__init__()
        self.lda = lda # HSIC weight / Trọng số HSIC
        self.device = device # Device / Thiết bị
        self.d = x_dim + y_dim # Total vars / Tổng số biến
        
        # Learnable Adjacency (DAG NOTEARS) / Ma trận kề học được
        self.W_dag = nn.Parameter(torch.zeros(self.d, self.d))
        
        # Backbone MLP / Khung mạng MLP
        self.MLP = mlp.MLP(input_dim=self.d, hidden_dim=hidden_dim, 
                          output_dim=self.d, n_clusters=n_clusters, device=device)
        
        self.gp_phi_z = RFFGPLayer(n_clusters, n_features=128) # Z mapping / Ánh xạ Z
        self.gp_phi_x = RFFGPLayer(self.d, n_features=128) # X mapping / Ánh xạ X
        self.linear_head = nn.Linear(128, self.d, bias=False) # Head / Đầu ra tuyến tính
        
        self.fast_hsic = FastHSIC(self.d, n_clusters, n_features=128) # Cluster HSIC / HSIC phân cụm
        self.pnl_hsic = FastHSIC(self.d, self.d, n_features=128) # PNL noise HSIC / HSIC nhiễu PNL

    def get_dag_penalty(self):
        """NOTEARS acyclicity constraint h(W) / Ràng buộc không chu trình h(W)"""
        W_sq = self.W_dag * self.W_dag # Squared weights / Bình phương trọng số
        E = torch.matrix_exp(W_sq) # Matrix exponential / Expo ma trận
        h = torch.trace(E) - self.d # Trace minus d / Vết trừ cho d
        return h # Penalty score / Điểm phạt h(W)

    def forward(self, batch_data, temperature=1.0):
        # Pass through MLP / Đi qua MLP
        out = self.MLP(batch_data, temperature=temperature)
        z_soft, kl_loss = out['z_soft'], out['kl_loss']
        
        # Mask inputs with DAG matrix / Che đầu vào bằng ma trận DAG
        masked_input = batch_data @ torch.abs(self.W_dag)
        
        # Predict using Z (mechanism) and masked X / Dự báo từ Z và X đã lọc
        phi = self.gp_phi_z(z_soft) * self.gp_phi_x(masked_input)
        y_pred_gp = self.linear_head(phi) # GP prediction / Dự báo từ GP
        
        loss_dag = self.get_dag_penalty() # DAG score / Ràng buộc DAG
        loss_reg = F.mse_loss(y_pred_gp, batch_data) # Regression MSE / Sai số hồi quy
        
        # Independence for PNL / Tính độc lập cho PNL
        h_y = self.MLP.pnl_transform(batch_data) # Transform data / Biến đổi dữ liệu
        res_pnl = h_y - y_pred_gp # Residuals / Phần dư
        loss_hsic_pnl = self.pnl_hsic(batch_data, res_pnl) # HSIC test / Kiểm định HSIC
        
        loss_hsic_clu = self.fast_hsic(batch_data, z_soft) # Cluster mapping / Ánh xạ cụm
        
        # Composite Loss / Hàm mất mát tổng hợp
        total_loss = (loss_reg + 
                      2.0 * loss_dag + # DAG Constraint / Ràng buộc DAG
                      self.lda * torch.log(loss_hsic_clu + 1e-8) + # Cluster HSIC / Tối ưu phân cụm
                      3.0 * torch.log(loss_hsic_pnl + 1e-8) + # Noise HSIC / Tối ưu nhiễu
                      0.2 * kl_loss) # VAE Latent / Ràng buộc VAE
        
        return total_loss, loss_reg, loss_hsic_clu # Return all / Trả về tất cả
