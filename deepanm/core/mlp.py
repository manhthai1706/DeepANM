"""
Deep Causal Network - Encoders, SEMs, and Normalizing Flows
Optimized for Causal Discovery (ANM, PNL, Heterogeneous Mechanisms)
Kiến trúc mô hình tối ưu cho Phát hiện Nhân quả phi tuyến
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder: Ánh xạ không gian đầu vào X thành biểu diễn ẩn và chỉ định cơ chế (Mechanisms).
    Dùng để xử lý dữ liệu hỗn hợp (heterogeneous data) có thể sinh ra từ nhiều cơ chế nhân quả khác nhau.
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
        # Sinh ra tham số phân phối ẩn (Cụm cơ chế)
        self.z_logits = nn.Linear(hidden_dim, n_mechanisms)
        
    def forward(self, x, temperature=1.0):
        feat = self.net(x)
        logits = self.z_logits(feat)
        
        # Xác suất phân loại cơ chế
        q_y = F.softmax(logits, dim=-1)
        # Tính toán KL Divergence từ Prior Uniform (1/K) -> Tránh Error VAE
        log_q_y = torch.log(q_y + 1e-10)
        kl_loss = torch.sum(q_y * (log_q_y - np.log(1.0 / q_y.shape[-1])), dim=-1)
        
        # Sử dụng Gumbel-Softmax để lấy mẫu cơ chế tự động và có thể vi phân
        if self.training:
            z_soft = F.gumbel_softmax(logits, tau=temperature, hard=False)
        else:
            z_soft = q_y
            
        return feat, z_soft, kl_loss.mean()

class ANM_SEM(nn.Module):
    """
    Structural Equation Model (SEM): Mô hình hóa phương trình nhân quả f(X).
    Sử dụng mạng phần dư (Residual Network) để xấp xỉ các hàm phi tuyến phức tạp nhất.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ))
            
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = F.gelu(h + block(h)) # Residual connection
        y_pred = self.output_proj(h)
        return y_pred

class HeterogeneousNoiseModel(nn.Module):
    """
    Mô hình Nhiễu Dị Thể (DECI-inspired Non-Gaussian Noise).
    Sử dụng Gaussian Mixture Model (GMM) linh hoạt theo dòng Causal Flows để 
    khớp chính xác các loại nhiễu lệch, đa đỉnh cực đoan (Long-tail, Bimodal).
    Phá vỡ hoàn toàn hạn chế của phân phối chuẩn Gaussian truyền thống, 
    cung cấp Likelihood cực chuẩn cho Variational Inference ghép nối DAG.
    """
    def __init__(self, dim, n_components=5):
        super().__init__()
        self.dim = dim
        self.n_components = n_components
        
        # GMM Parameters: Phân phối trọng lượng hỗn hợp (Logits), Trọng tâm (Means) và Biên độ uốn (Log-Vars)
        self.logits = nn.Parameter(torch.zeros(dim, n_components))
        self.means = nn.Parameter(torch.randn(dim, n_components) * 0.05)
        self.log_vars = nn.Parameter(torch.zeros(dim, n_components) - 1.0)

    def compute_log_prob(self, noise):
        # noise shape: (batch, dim) => mở rộng để Vector hóa GMM: (batch, dim, 1)
        noise_expanded = noise.unsqueeze(-1) 
        
        # Softmax để lấy Normalization Weights của từng Cụm nhiễu
        weights = F.softmax(self.logits, dim=-1) # (dim, n_components)
        vars = torch.exp(self.log_vars) + 1e-6
        
        # Log-Likelihood của nội tại chuẩn bị trộn (log N(x))
        log_probs_components = -0.5 * (np.log(2 * np.pi) + self.log_vars + ((noise_expanded - self.means) ** 2) / vars)
        log_weights = torch.log(weights + 1e-10)
        
        # Ghép nối trọng số
        weighted_log_probs = log_weights + log_probs_components
        
        # Log-Sum-Exp Trick: Kỹ thuật Flow chống nổ Gradient (Gradient Explosion) của Pytorch
        log_prob = torch.logsumexp(weighted_log_probs, dim=-1) # (batch, dim)
        
        # Trả về Tổng Độ Tiết Lộ Nhiễu (NLL Target)
        return log_prob.sum(dim=-1) # (batch,)

class Decoder(nn.Module):
    """
    Decoder cho phép đảo ngược trạng thái: Cực kì hữu ích cho mô hình PNL (Post-Nonlinear).
    Ánh xạ dạng Y = g(f(X) + N). Trong đó Decoder tương đương với hàm g(.) đơn điệu.
    """
    def __init__(self, output_dim):
        super().__init__()
        # Trọng số qua Softplus bảo đảm đạo hàm luôn dương tính -> tính đơn điệu (Monotonicity)
        self.weight = nn.Parameter(torch.ones(output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x):
        return F.softplus(self.weight) * x + self.bias
        
    def inverse(self, y_trans):
        """Khả nghịch lại g^-1(.) cho việc tính nhiễu N = g^-1(Y) - f(X)"""
        return (y_trans - self.bias) / (F.softplus(self.weight) + 1e-6)

class MLP(nn.Module):
    """
    Mạng Causal MLP hợp nhất toàn bộ vòng đời: Encoder -> SEM -> Decoder + Flow/Gaussian Noise.
    Tối ưu hóa để phát hiện nhân quả song biến (X -> Y hoặc mạng đa biến).
    * Giữ lại module tên MLP để tương thích với các script gọi vào của bạn.
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1, n_clusters=2, use_spline=True, device='cpu'):
        super().__init__()
        self.device = device
        self.n_clusters = n_clusters
        
        # 1. Tách đặc trưng và cụm cơ chế (Gumbel Mechanism Selection)
        self.encoder = Encoder(input_dim, hidden_dim, n_clusters)
        
        # 2. Xấp xỉ phương trình cấu trúc f(X) qua Additive Noise Model (ANM-SEM)
        self.sem = ANM_SEM(input_dim, hidden_dim, output_dim)
        
        # 3. Mô hình hóa phân phối nhiễu phức hợp qua mạng GMM lai DECI
        self.noise_model = HeterogeneousNoiseModel(output_dim, n_components=5)
            
        # 4. Post-Nonlinear Decoder
        self.pnl_transform = Decoder(output_dim)
        
        self.to(device)

    def forward(self, x, y=None, temperature=1.0):
        """Lan truyền tiến tính toán phương trình nhân quả"""
        if isinstance(x, np.ndarray): x = torch.from_numpy(x).float().to(self.device)
        if y is not None and isinstance(y, np.ndarray): y = torch.from_numpy(y).float().to(self.device)
        
        # 1. Phân loại cơ chế sinh học
        feat, z_soft, kl_loss = self.encoder(x, temperature)
        
        # 2. Suy diễn f(X)
        mu = self.sem(x)
        
        # 3. Chuyển vị hậu hình học (Post-nonlinear)
        y_trans = self.pnl_transform(mu)
        
        results = {
            "feat": feat,
            "z_soft": z_soft,
            "kl_loss": kl_loss, # Khôi phục biến này để tích hợp Core GPPOM_HSIC
            "mu": mu,           # Giá trị dự báo của hàm nguyên trạng thái
            "y_trans": y_trans, # Giá trị sau biển đổi
        }
        
        if y is not None:
            # Nhiễu sinh ra
            noise = y - mu
            results["noise"] = noise
            
            results["log_prob_noise"] = self.noise_model.compute_log_prob(noise)

        return results

    def estimate_ate(self, X_control, X_treatment):
        """
        Tính toán Average Treatment Effect (ATE)
        ATE = E[ Y | do(X_treatment) ] - E[ Y | do(X_control) ]
        Rất ý nghĩa khi muốn đo kích thước tác động nhân quả, bên cạnh việc chỉ tìm hướng.
        """
        self.eval() # Chuyển sang chế độ đánh giá
        with torch.no_grad():
            if isinstance(X_control, np.ndarray): 
                X_control = torch.from_numpy(X_control).float().to(self.device)
            if isinstance(X_treatment, np.ndarray): 
                X_treatment = torch.from_numpy(X_treatment).float().to(self.device)
            
            # Suy diễn giá trị Y qua cấu trúc Structural Equation f(X)
            y_control = self.sem(X_control)
            y_treatment = self.sem(X_treatment)
            
            # Decode nếu mô hình đang hoạt động ở chế độ PNL
            y_control_decoded = self.pnl_transform.inverse(y_control)
            y_treatment_decoded = self.pnl_transform.inverse(y_treatment)
            
            # Tính Individual Treatment Effect (ITE) và lấy giá trị trung bình ATE
            ite = y_treatment_decoded - y_control_decoded
            ate = torch.mean(ite).item()
            
        return ate

    def get_global_ate_matrix(self, X, W_dag=None, eps=1e-3):
        """
        Sử dụng ATE để khám phá Cấu trúc Toàn cục (Global Discovery Assist).
        Chạy mô phỏng do-calculus (can thiệp) trên toàn bộ không gian biến:
        Đo lường độ thay đổi của toàn bộ hệ thống (Y_j) khi ta tác động eps vào X_i.
        Jacobian Matrix ATE_ij = partial f_j / partial x_i.
        """
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_ten = torch.from_numpy(X).float().to(self.device)
            else:
                X_ten = X.clone().float().to(self.device)
                
            n_samples, n_vars = X_ten.shape
            ate_matrix = torch.zeros((n_vars, n_vars), device=self.device)
            
            # Tính Control State (Giá trị dự đoán mạng SCM Gốc)
            if W_dag is not None:
                # Hệ thống DAG: Biến j chỉ nhận input từ Parents của nó (X @ W_dag)
                base_input = X_ten @ W_dag
            else:
                base_input = X_ten
                
            y_base = self.sem(base_input)
            y_base = self.pnl_transform.inverse(y_base)
            
            # Tính Treatment State lần lượt cho từng biến i
            for i in range(n_vars):
                # Intervene (Treatment) trên biến X_i hiện tại
                X_treat = X_ten.clone()
                X_treat[:, i] += eps # Bơm nhiễu can thiệp vào riêng biến i
                
                # Biến i thay đổi, mạng nhân quả DAG sẽ truyền tác động này xuống các children của nó thông qua W_dag
                if W_dag is not None:
                    treat_input = X_treat @ W_dag
                else:
                    treat_input = X_treat
                    
                y_treat = self.sem(treat_input)
                y_treat = self.pnl_transform.inverse(y_treat)
                
                # ATE = E [ Y_treat - Y_control ] / eps -> Causal Derivative
                ate_i = torch.mean(y_treat - y_base, dim=0) / eps
                ate_matrix[i, :] = ate_i
                
            # Hệ thống nhân quả (DAG) cấm 1 biến tạo tác động lên chính nó
            ate_matrix.fill_diagonal_(0.0)
            
        return ate_matrix
