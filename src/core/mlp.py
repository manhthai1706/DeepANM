"""
DeepANM Neural Components: Encoder, SEM, Noise Model, Decoder, and MLP.
Implements a unified causal network for nonlinear structure learning (ANM/PNL).

Các thành phần Neural của DeepANM: Encoder, SEM, Mô hình Nhiễu, Decoder và MLP.
Xây dựng mạng nhân quả thống nhất cho học cấu trúc phi tuyến (ANM/PNL).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Maps input X to a latent cluster assignment using Gumbel-Softmax.
    Identifies which causal mechanism generated each sample.

    Ánh xạ đầu vào X thành phân cụm cơ chế ẩn dùng Gumbel-Softmax.
    Xác định cơ chế nhân quả nào đã sinh ra từng mẫu dữ liệu.
    """
    def __init__(self, input_dim, hidden_dim, n_mechanisms=2, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        # Logits for mechanism assignment / Logit cho phân loại cơ chế
        self.z_logits = nn.Linear(hidden_dim, n_mechanisms)
        
    def forward(self, x, temperature=1.0):
        feat = self.net(x)
        logits = self.z_logits(feat)
        
        # Soft mechanism probabilities / Xác suất cơ chế mềm
        q_y = F.softmax(logits, dim=-1)
        # KL divergence from uniform prior to encourage entropy / KL phân kỳ từ prior đều để khuyến khích entropy
        log_q_y = torch.log(q_y + 1e-10)
        kl_loss = torch.sum(q_y * (log_q_y - np.log(1.0 / q_y.shape[-1])), dim=-1)
        
        # Gumbel-Softmax: differentiable discrete sampling / Lấy mẫu rời rạc có thể vi phân
        if self.training:
            z_soft = F.gumbel_softmax(logits, tau=temperature, hard=False)
        else:
            z_soft = q_y
            
        return feat, z_soft, kl_loss.mean()

class ANM_SEM(nn.Module):
    """
    Structural Equation Model (SEM): learns the nonlinear causal function f(X).
    Residual MLP architecture for stable training and expressive power.

    Mô hình Phương trình Cấu trúc (SEM): học hàm nhân quả phi tuyến f(X).
    Kiến trúc MLP thặng dư để huấn luyện ổn định và khả năng biểu đạt mạnh.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, dropout=0.05):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
            
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = F.gelu(h + block(h))  # Residual connection / Kết nối thặng dư (Skip connection)
        return self.output_proj(h)

class HeterogeneousNoiseModel(nn.Module):
    """
    Gaussian Mixture Model (GMM) noise model (DECI-inspired).
    Fits non-Gaussian, heavy-tailed, or multimodal noise distributions.

    Mô hình nhiễu hỗn hợp Gaussian (GMM), lấy cảm hứng từ DECI.
    Khớp các phân phối nhiễu phi Gaussian, đuôi nặng hoặc đa đỉnh.
    """
    def __init__(self, dim, n_components=5):
        super().__init__()
        self.dim = dim
        self.n_components = n_components
        
        # GMM parameters: learning mixture weights, means, and variances
        # Tham số GMM: học trọng số hỗn hợp, trung bình và phương sai
        self.logits = nn.Parameter(torch.zeros(dim, n_components))
        self.means = nn.Parameter(torch.randn(dim, n_components) * 0.05)
        self.log_vars = nn.Parameter(torch.zeros(dim, n_components) - 1.0)

    def compute_log_prob(self, noise):
        """Vectorized log-likelihood computation for the GMM. / Tính toán log-likelihood vector hóa cho GMM."""
        noise_expanded = noise.unsqueeze(-1) # (batch, dim, 1)
        
        weights = F.softmax(self.logits, dim=-1)  # (dim, n_components)
        vars = torch.exp(self.log_vars) + 1e-6
        
        # Log-likelihood of each Gaussian component / Log-likelihood của từng thành phần Gaussian
        log_probs_components = -0.5 * (np.log(2 * np.pi) + self.log_vars + ((noise_expanded - self.means) ** 2) / vars)
        log_weights = torch.log(weights + 1e-10)
        
        # Log-sum-exp trick for numerical stability / Thủ thuật log-sum-exp để đảm bảo ổn định số học
        log_prob = torch.logsumexp(log_weights + log_probs_components, dim=-1)  # (batch, dim)
        return log_prob.sum(dim=-1)  # Sum across variables / Tổng trên các biến (batch,)

class Decoder(nn.Module):
    """
    Post-Nonlinear (PNL) transformation: g(.) in Y = g(f(X) + N).
    Monotone affine transform (via Softplus) to ensure invertibility.

    Biến đổi hậu phi tuyến (PNL): g(.) trong Y = g(f(X) + N).
    Biến đổi đơn điệu (qua Softplus) để đảm bảo tính khả nghịch.
    """
    def __init__(self, output_dim):
        super().__init__()
        # Softplus ensures weight > 0, which guarantees monotonicity
        # Softplus đảm bảo trọng số > 0, bảo đảm tính đơn điệu của hàm
        self.weight = nn.Parameter(torch.ones(output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x):
        return F.softplus(self.weight) * x + self.bias
        
    def inverse(self, y_trans):
        """Inverse g⁻¹(.) to recover noise. / Hàm nghịch đảo g⁻¹(.) để khôi phục nhiễu."""
        return (y_trans - self.bias) / (F.softplus(self.weight) + 1e-6)

class MLP(nn.Module):
    """
    Unified Causal Neural Network integrating Encoder, SEM, PNL Decoder, and GMM Noise.
    Mạng Neural Nhân quả thống nhất: tích hợp Encoder, SEM, Decoder PNL và Nhiễu GMM.
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1, n_clusters=2, device='cpu'):
        super().__init__()
        self.device = device
        self.n_clusters = n_clusters
        
        # 1. Mechanism classifier (Encoder) / Bộ phân loại cơ chế (Encoder)
        self.encoder = Encoder(input_dim, hidden_dim, n_clusters)
        
        # 2. Structural Equation Model f(X) / Mô hình phương trình cấu trúc f(X)
        self.sem = ANM_SEM(input_dim, hidden_dim, output_dim)
        
        # 3. GMM noise distribution head / Đầu phân phối nhiễu GMM
        self.noise_model = HeterogeneousNoiseModel(output_dim, n_components=5)
            
        # 4. Post-Nonlinear decoder g(.) / Bộ giải mã hậu phi tuyến g(.)
        self.pnl_transform = Decoder(output_dim)
        
        self.to(device)

    def forward(self, x, y=None, temperature=1.0):
        """Standard Forward pass for mechanism learning. / Lan truyền tiến chuẩn cho việc học cơ chế."""
        if isinstance(x, np.ndarray): x = torch.from_numpy(x).float().to(self.device)
        if y is not None and isinstance(y, np.ndarray): y = torch.from_numpy(y).float().to(self.device)
        
        # 1. Classify mechanism / Phân loại cơ chế mẫu
        feat, z_soft, kl_loss = self.encoder(x, temperature)
        
        # 2. Predict causal component f(X) / Dự đoán thành phần nhân quả f(X)
        mu = self.sem(x)
        
        # 3. Calculate noise proxy: N = g(X) - f(X) / Tính toán đại diện nhiễu: N = g(X) - f(X)
        noise_proxy = self.pnl_transform(x) - mu

        results = {
            "z_soft": z_soft,
            "kl_loss": kl_loss,
            "mu": mu,
            "log_prob_noise": self.noise_model.compute_log_prob(noise_proxy),
        }
        
        if y is not None:
            # If target y is provided, compute noise from target / Nếu có y thực, tính nhiễu từ mục tiêu
            exact_noise = y - mu
            results["log_prob_noise"] = self.noise_model.compute_log_prob(exact_noise)

        return results

    def estimate_ate(self, X_control, X_treatment):
        """
        Interventional ATE: Expectation difference under do-intervention.
        ATE can thiệp: Sự khác biệt kỳ vọng dưới can thiệp do(.).
        """
        self.eval()
        with torch.no_grad():
            if isinstance(X_control, np.ndarray): 
                X_control = torch.from_numpy(X_control).float().to(self.device)
            if isinstance(X_treatment, np.ndarray): 
                X_treatment = torch.from_numpy(X_treatment).float().to(self.device)
            
            # Predict outcome in control vs treatment state / Dự đoán kết quả trong trạng thái đối chứng vs can thiệp
            y_control = self.pnl_transform.inverse(self.sem(X_control))
            y_treatment = self.pnl_transform.inverse(self.sem(X_treatment))
            ate = torch.mean(y_treatment - y_control).item()
            
        return ate

    def get_global_ate_matrix(self, X, W_dag=None, eps=1e-3):
        """
        Computes the complete ATE Jacobian matrix across all variable pairs.
        Tính toán ma trận Jacobian ATE hoàn chỉnh trên tất cả các cặp biến.
        """
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_ten = torch.from_numpy(X).float().to(self.device)
            else:
                X_ten = X.clone().float().to(self.device)

            n_samples, n_vars = X_ten.shape

            # Compute baseline output / Tính đầu ra cơ sở
            base_input = X_ten @ W_dag if W_dag is not None else X_ten
            y_base = self.pnl_transform.inverse(self.sem(base_input))

            # Batch intervention calculation / Tính toán can thiệp theo lô
            treat_batch = base_input.unsqueeze(0).repeat(n_vars, 1, 1)  # (v, n, v)
            perturbation = torch.eye(n_vars, device=self.device).unsqueeze(1) * eps
            treat_batch += perturbation

            treat_flat = treat_batch.view(n_vars * n_samples, n_vars)
            y_treat = self.pnl_transform.inverse(self.sem(treat_flat)).view(n_vars, n_samples, n_vars)

            # Sensitivity: rate of change across variables / Độ nhạy: tỷ lệ thay đổi giữa các biến
            sensitivity = torch.zeros(n_vars, device=self.device)
            for j in range(n_vars):
                sensitivity[j] = (y_treat[j, :, j] - y_base[:, j]).mean() / eps

            # Direct Effects = Jacobian sensitivity weighted by binary graph
            # Tác động trực tiếp = Jacobian của độ nhạy có trọng số bởi đồ thị nhị phân
            if W_dag is not None:
                ate_matrix = sensitivity.unsqueeze(0) * W_dag
            else:
                ate_matrix = torch.mean(y_treat - y_base.unsqueeze(0), dim=1) / eps

            ate_matrix.fill_diagonal_(0.0) # Self-causality is zero / Tự nhân quả bằng 0

        return ate_matrix
