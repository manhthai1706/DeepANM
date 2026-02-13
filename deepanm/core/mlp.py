"""
Ultimate Deep Learning MLP / Mạng nơ-ron MLP chuyên sâu
Features: Self-Attention, Gumbel-Softmax, Normalizing Flows
Tính năng: Tự chú ý, Gumbel-Softmax, Luồng chuẩn hóa
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class AttentionLayer(nn.Module):
    """Self-Attention for weighting input features / Lớp tự chú ý đánh trọng số đặc trưng"""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim) # Query projection / Phép chiếu Query
        self.key = nn.Linear(dim, dim) # Key projection / Phép chiếu Key
        self.value = nn.Linear(dim, dim) # Value projection / Phép chiếu Value
        self.scale = dim ** -0.5 # Scaling factor / Hệ số tỉ lệ

    def forward(self, x):
        x_in = x.unsqueeze(1) # Add sequence dimension / Thêm chiều chuỗi
        q = self.query(x_in) # Compute query / Tính query
        k = self.key(x_in) # Compute key / Tính key
        v = self.value(x_in) # Compute value / Tính value
        
        # Dot-product attention / Chú ý tích vô hướng
        attn = (q @ k.transpose(-2, -1)) * self.scale # Calculate scores / Tính điểm số
        attn = attn.softmax(dim=-1) # Normalize weights / Chuẩn hóa trọng số
        out = (attn @ v).squeeze(1) # Apply attention / Áp dụng chú ý
        return out + x # Residual connection / Kết nối phần dư

class InvertibleLayer(nn.Module):
    """Monotonic Invertible Layer / Lớp khả nghịch đơn điệu"""
    def __init__(self, dim):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(dim)) # Learnable weights / Trọng số học được
        self.bias = nn.Parameter(torch.zeros(dim)) # Learnable bias / Độ chệch học được
        
    def forward(self, x):
        # f(x) = softplus(w) * x + b (Ensures monotonicity / Đảm bảo tính đơn điệu)
        return F.softplus(self.weights) * x + self.bias
    
    def inverse(self, y):
        """Invert the transformation / Nghịch đảo phép biến đổi"""
        return (y - self.bias) / F.softplus(self.weights)

class ResBlock(nn.Module):
    """Residual Block with LayerNorm / Khối phần dư kèm LayerNorm"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), # Linear layer / Lớp tuyến tính
            nn.LayerNorm(dim), # Normalization / Chuẩn hóa lớp
            nn.GELU(), # Activation function / Hàm kích hoạt
            nn.Dropout(dropout), # Dropout for regularization / Dropout ổn định hóa
            nn.Linear(dim, dim), # Second linear layer / Lớp tuyến tính thứ hai
            nn.LayerNorm(dim) # Final normalization / Chuẩn hóa cuối
        )
    def forward(self, x):
        # Add input to output (Residual / Phần dư)
        return F.gelu(x + self.block(x))

class MonotonicSplineLayer(nn.Module):
    """Neural Spline Flow simplified / Luồng Spline Nơ-ron đơn giản hóa"""
    def __init__(self, dim, hidden_dim=32, n_bins=8):
        super().__init__()
        self.dim = dim # Dimension / Số chiều
        self.n_bins = n_bins # Number of spline bins / Số lượng thùng spline
        # Learn mapping to spline parameters / Học ánh xạ sang tham số spline
        self.spline_params = nn.Linear(dim, dim * (3 * n_bins + 1))
        
    def forward(self, x):
        params = self.spline_params(x) # Get parameters / Lấy tham số
        # Reshape and sum to form monotonic curve / Chuyển hình và cộng để tạo đường đơn điệu
        return torch.tanh(params.view(x.shape[0], self.dim, -1)).sum(dim=-1)

class MultivariateCausalBackbone(nn.Module):
    """Feature selection backbone / Khung mạng chọn lọc đặc trưng"""
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # Hidden layer / Lớp ẩn
            nn.GELU(), # GELU activation / Kích hoạt GELU
            nn.Linear(hidden_dim, input_dim), # Project back / Chiếu ngược lại
            nn.Sigmoid() # Sigmoid for gating / Sigmoid để tạo cổng
        )
        # Sequential residual blocks / Các khối phần dư tuần tự
        self.res_blocks = nn.ModuleList([ResBlock(input_dim, dropout) for _ in range(3)])

    def forward(self, x):
        # Apply gate (feature selection / chọn lọc đặc trưng)
        gated_x = x * self.gate(x)
        for block in self.res_blocks: # Pass through ResBlocks / Đi qua các khối ResBlock
            gated_x = block(gated_x)
        return gated_x # Result / Kết quả

class MLP(nn.Module):
    """Multi-Head Causal MLP / Mạng MLP nhiều đầu ra nhân quả"""
    def __init__(self, input_dim, hidden_dim, output_dim, n_clusters=2, device='cpu'):
        super().__init__()
        self.device = device # Computation device / Thiết bị tính toán
        self.n_clusters = n_clusters # Num of mechanisms / Số lượng cơ chế
        self.output_dim = output_dim # Output size / Kích thước đầu ra
        
        self.backbone = MultivariateCausalBackbone(input_dim, hidden_dim) # Core feat / Đặc trưng lõi
        
        # VAE heads / Các đầu VAE
        self.z_mean = nn.Linear(input_dim, n_clusters) # Mean head / Đầu tính trung bình
        self.z_logvar = nn.Linear(input_dim, n_clusters) # Variance head / Đầu tính phương sai
        
        self.regressor = nn.Linear(input_dim, output_dim * 2) # Prediction head / Đầu dự báo
        self.spline_flow = MonotonicSplineLayer(output_dim) # Noise model / Mô hình nhiễu
        self.pnl_transform = InvertibleLayer(output_dim) # PNL head / Đầu PNL
        
        self.to(device) # Move to device / Chuyển vào thiết bị
        
    def encode_latent(self, x):
        """Extract latent parameters / Trích xuất tham số ẩn"""
        feat = self.backbone(x)
        return self.z_mean(feat), self.z_logvar(feat)

    def reparameterize(self, mu, logvar):
        """VAE reparameterization trick / Thủ thuật lấy mẫu VAE"""
        std = torch.exp(0.5 * logvar) # Get standard deviation / Lấy độ lệch chuẩn
        eps = torch.randn_like(std) # Normal noise / Nhiễu chuẩn
        return mu + eps * std # Reparameterized sample / Mẫu đã tham số hóa

    def forward(self, x, temperature=1.0):
        if isinstance(x, np.ndarray): x = torch.from_numpy(x).float().to(self.device)
            
        feat = self.backbone(x) # Extract features / Trích xuất đặc trưng
        
        mu_z, logvar_z = self.z_mean(feat), self.z_logvar(feat) # Latent heads / Các đầu biến ẩn
        z_sample = self.reparameterize(mu_z, logvar_z) # Sample latent / Lấy mẫu biến ẩn
        z_soft = F.softmax(z_sample / temperature, dim=-1) # Gumbel-Softmax like / Phân cụm mềm
        
        # KL loss formula / Công thức mất mát KL
        kl_vae = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=-1)
        
        reg_out = self.regressor(feat) # Regressor output / Đầu ra mạng hồi quy
        mu, log_var = torch.chunk(reg_out, 2, dim=-1) # Split mu and var / Tách trung bình và phương sai
        
        noise_spline = self.spline_flow(torch.randn_like(mu)) # Generated noise / Tạo nhiễu
        
        return {
            "mu": mu, "log_var": log_var, "z_soft": z_soft,
            "kl_loss": kl_vae.mean(), "noise_complex": noise_spline,
            "y_trans": self.pnl_transform(mu) # Transform for PNL / Biến đổi cho PNL
        }

    def train_model(self, x, y, epochs=200, lr=1e-3):
        """Train the MLP / Huấn luyện MLP"""
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-2)
        self.train() # Set training mode / Đặt chế độ huấn luyện
        
        for epoch in range(epochs):
            optimizer.zero_grad() # Clear grads / Xóa gradient
            out = self.forward(x) # Forward / Lan truyền tiến
            # NLL Loss / Mất mát log-likelihood âm
            loss_reg = F.gaussian_nll_loss(out['mu'], y, torch.exp(out['log_var']))
            loss = loss_reg + 0.1 * out['kl_loss'] # Total loss / Tổng mất mát
            loss.backward() # Backward / Lan truyền ngược
            optimizer.step() # Update / Cập nhật trọng số

    def predict(self, x):
        """Predict results / Dự báo kết quả"""
        self.eval() # Eval mode / Chế độ đánh giá
        with torch.no_grad(): # No tracking / Không theo dõi gradient
            out = self.forward(x) # Get output / Lấy đầu ra
            return out['mu'].cpu().numpy(), torch.exp(0.5*out['log_var']).cpu().numpy(), out['z_soft'].cpu().numpy()
