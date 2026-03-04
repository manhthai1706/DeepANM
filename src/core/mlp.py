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
        # KL divergence from uniform prior / KL phân kỳ từ prior đều
        log_q_y = torch.log(q_y + 1e-10)
        kl_loss = torch.sum(q_y * (log_q_y - np.log(1.0 / q_y.shape[-1])), dim=-1)
        
        # Gumbel-Softmax: differentiable discrete sampling
        # Gumbel-Softmax: lấy mẫu rời rạc có thể vi phân
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
    Kiến trúc MLP thặng dư để huấn luyện ổn định và biểu đạt mạnh.
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
            h = F.gelu(h + block(h))  # Residual connection / Kết nối thặng dư
        return self.output_proj(h)

class HeterogeneousNoiseModel(nn.Module):
    """
    Gaussian Mixture Model (GMM) noise model (DECI-inspired).
    Fits non-Gaussian, heavy-tailed, or multimodal noise distributions.
    Provides a robust log-likelihood target for variational inference.

    Mô hình nhiễu hỗn hợp Gaussian (GMM), lấy cảm hứng từ DECI.
    Khớp các phân phối nhiễu phi Gaussian, đuôi nặng hoặc đa đỉnh.
    Cung cấp log-likelihood mục tiêu bền vững cho suy diễn biến phân.
    """
    def __init__(self, dim, n_components=5):
        super().__init__()
        self.dim = dim
        self.n_components = n_components
        
        # GMM parameters: mixture weights, means, log-variances
        # Tham số GMM: trọng số hỗn hợp, trung bình, log-phương sai
        self.logits = nn.Parameter(torch.zeros(dim, n_components))
        self.means = nn.Parameter(torch.randn(dim, n_components) * 0.05)
        self.log_vars = nn.Parameter(torch.zeros(dim, n_components) - 1.0)

    def compute_log_prob(self, noise):
        # noise: (batch, dim) → expand for vectorized GMM: (batch, dim, 1)
        # noise: (batch, dim) → mở rộng để vector hóa GMM: (batch, dim, 1)
        noise_expanded = noise.unsqueeze(-1)
        
        weights = F.softmax(self.logits, dim=-1)  # (dim, n_components)
        vars = torch.exp(self.log_vars) + 1e-6
        
        # Log-likelihood of each Gaussian component / Log-likelihood từng thành phần Gaussian
        log_probs_components = -0.5 * (np.log(2 * np.pi) + self.log_vars + ((noise_expanded - self.means) ** 2) / vars)
        log_weights = torch.log(weights + 1e-10)
        
        # Log-sum-exp trick for numerical stability / Thủ thuật log-sum-exp để ổn định số học
        log_prob = torch.logsumexp(log_weights + log_probs_components, dim=-1)  # (batch, dim)
        return log_prob.sum(dim=-1)  # (batch,)

class Decoder(nn.Module):
    """
    Post-Nonlinear (PNL) transformation: g(.) in Y = g(f(X) + N).
    Monotone affine transform (via Softplus) to ensure invertibility.

    Biến đổi hậu phi tuyến (PNL): g(.) trong Y = g(f(X) + N).
    Biến đổi affine đơn điệu (qua Softplus) để đảm bảo khả nghịch.
    """
    def __init__(self, output_dim):
        super().__init__()
        # Softplus ensures weight > 0, guaranteeing monotonicity
        # Softplus đảm bảo weight > 0, bảo đảm tính đơn điệu
        self.weight = nn.Parameter(torch.ones(output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x):
        return F.softplus(self.weight) * x + self.bias
        
    def inverse(self, y_trans):
        """Inverse g⁻¹(.) to recover noise N = g⁻¹(Y) - f(X).
        Hàm nghịch g⁻¹(.) để khôi phục nhiễu N = g⁻¹(Y) - f(X)."""
        return (y_trans - self.bias) / (F.softplus(self.weight) + 1e-6)

class MLP(nn.Module):
    """
    Unified Causal Neural Network integrating Encoder, SEM, PNL Decoder, and GMM Noise.
    Handles multi-variable causal structure learning and Direct Effect estimation (ATE).

    Mạng Neural Nhân quả thống nhất: tích hợp Encoder, SEM, Decoder PNL và Nhiễu GMM.
    Xử lý học cấu trúc nhân quả đa biến và ước tính tác động nhân quả trực tiếp (ATE).
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1, n_clusters=2, device='cpu'):
        super().__init__()
        self.device = device
        self.n_clusters = n_clusters
        
        # 1. Mechanism classifier / Bộ phân loại cơ chế
        self.encoder = Encoder(input_dim, hidden_dim, n_clusters)
        
        # 2. Structural Equation Model f(X) / Mô hình phương trình cấu trúc f(X)
        self.sem = ANM_SEM(input_dim, hidden_dim, output_dim)
        
        # 3. GMM noise distribution / Phân phối nhiễu GMM
        self.noise_model = HeterogeneousNoiseModel(output_dim, n_components=5)
            
        # 4. Post-Nonlinear decoder g(.) / Decoder hậu phi tuyến g(.)
        self.pnl_transform = Decoder(output_dim)
        
        self.to(device)

    def forward(self, x, y=None, temperature=1.0):
        """Forward pass computing structural causal equations.
        Lan truyền tiến tính toán phương trình nhân quả cấu trúc."""
        if isinstance(x, np.ndarray): x = torch.from_numpy(x).float().to(self.device)
        if y is not None and isinstance(y, np.ndarray): y = torch.from_numpy(y).float().to(self.device)
        
        # 1. Classify mechanism / Phân loại cơ chế
        feat, z_soft, kl_loss = self.encoder(x, temperature)
        
        # 2. Predict f(X) / Dự đoán f(X)
        mu = self.sem(x)
        
        # 3. Self-supervised noise proxy: N̂ = g(X) - f(X)
        # Ước tính nhiễu tự giám sát: N̂ = g(X) - f(X)
        noise_proxy = self.pnl_transform(x) - mu

        results = {
            "z_soft": z_soft,
            "kl_loss": kl_loss,
            "mu": mu,
            "log_prob_noise": self.noise_model.compute_log_prob(noise_proxy),
        }
        
        if y is not None:
            # If ground truth y available, use exact noise / Nếu có y thực, dùng nhiễu chính xác
            exact_noise = y - mu
            results["log_prob_noise"] = self.noise_model.compute_log_prob(exact_noise)

        return results

    def estimate_ate(self, X_control, X_treatment):
        """
        Binary ATE: E[Y | do(X_treatment)] - E[Y | do(X_control)].
        Measures the causal effect magnitude from a do-intervention.

        ATE nhị phân: đo cường độ tác động nhân quả từ can thiệp do(.).
        """
        self.eval()
        with torch.no_grad():
            if isinstance(X_control, np.ndarray): 
                X_control = torch.from_numpy(X_control).float().to(self.device)
            if isinstance(X_treatment, np.ndarray): 
                X_treatment = torch.from_numpy(X_treatment).float().to(self.device)
            
            y_control = self.pnl_transform.inverse(self.sem(X_control))
            y_treatment = self.pnl_transform.inverse(self.sem(X_treatment))
            ate = torch.mean(y_treatment - y_control).item()
            
        return ate

    def get_global_ate_matrix(self, X, W_dag=None, eps=1e-3):
        """
        Compute the Direct Causal Effect (ATE) matrix via batched neural Jacobian.
        For each edge i→j: ATE[i,j] = sensitivity_j × W_dag[i,j].

        Tính ma trận Tác động Nhân quả Trực tiếp (ATE) qua Jacobian Neural theo lô.
        Với mỗi cạnh i→j: ATE[i,j] = độ_nhạy_j × W_dag[i,j].
        """
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_ten = torch.from_numpy(X).float().to(self.device)
            else:
                X_ten = X.clone().float().to(self.device)

            n_samples, n_vars = X_ten.shape

            # DAG-masked base input / Đầu vào cơ sở được che bởi DAG
            base_input = X_ten @ W_dag if W_dag is not None else X_ten
            y_base = self.pnl_transform.inverse(self.sem(base_input))

            # Batch perturbation: perturb each variable's input column by eps
            # Can thiệp theo lô: nhiễu từng cột đầu vào một khoảng eps
            treat_batch = base_input.unsqueeze(0).repeat(n_vars, 1, 1)  # (d, n, d)
            perturbation = torch.eye(n_vars, device=self.device).unsqueeze(1) * eps
            treat_batch += perturbation

            treat_flat = treat_batch.view(n_vars * n_samples, n_vars)
            y_treat = self.pnl_transform.inverse(self.sem(treat_flat)).view(n_vars, n_samples, n_vars)

            # Diagonal sensitivity: how much output j responds to perturbation in j's input
            # Độ nhạy đường chéo: đầu ra j phản ứng bao nhiêu khi đầu vào j bị nhiễu
            sensitivity = torch.zeros(n_vars, device=self.device)
            for j in range(n_vars):
                sensitivity[j] = (y_treat[j, :, j] - y_base[:, j]).mean() / eps

            # Direct ATE = sensitivity × edge weight / ATE trực tiếp = độ nhạy × trọng số cạnh
            if W_dag is not None:
                ate_matrix = sensitivity.unsqueeze(0) * W_dag
            else:
                ate_matrix = torch.mean(y_treat - y_base.unsqueeze(0), dim=1) / eps

            ate_matrix.fill_diagonal_(0.0)

        return ate_matrix
