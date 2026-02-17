"""
DeepANM - Model Architecture / Kiến trúc mô hình DeepANM
"""

import numpy as np
import torch
import torch.nn as nn

class DeepANM(nn.Module):
    """Deep Additive Noise Model implementation / Triển khai mô hình nhiễu cộng sâu"""
    def __init__(self, x_dim=None, y_dim=0, n_clusters=2, hidden_dim=64, lda=1.0, device=None, data=None, **kwargs):
        """Initialize and optionally train / Khởi tạo và huấn luyện (tùy chọn)"""
        super().__init__()
        
        # Auto-detect dimension from data / Tự động xác định số chiều từ dữ liệu
        if data is not None and x_dim is None:
            x_dim = data.shape[1]
            
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu') # GPU/CPU select / Chọn GPU/CPU
        self.lda = lda # HSIC weight / Trọng số HSIC
        self.x_dim = x_dim # Input dim / Chiều đầu vào
        self.y_dim = y_dim # Target dim / Chiều đầu ra
        self.n_clusters = n_clusters # Mechanisms / Số cơ chế
        self.hidden_dim = hidden_dim # MLP width / Độ rộng mạng MLP
        self.d = x_dim + y_dim if x_dim is not None else 0 # Total vars / Tổng số biến
        
        # Import core engine late to avoid loops / Import module cốt lõi sau để tránh lỗi vòng lặp
        from deepanm.core.gppom_hsic import GPPOMC_lnhsic_Core
        
        if x_dim is not None: # Build core / Xây dựng nhân mô hình
            self.core = GPPOMC_lnhsic_Core(x_dim, y_dim, n_clusters, hidden_dim, lda, self.device)
        else:
            self.core = None # Lazy init / Khởi tạo lười
            
        self.history = None # Log history / Lịch sử huấn luyện
        self.trainer = None # Trainer instance / Thực thể huấn luyện
        self.to(self.device) # Move to GPU/CPU / Chuyển vào GPU/CPU
        
        if data is not None: # Immediate training / Huấn luyện ngay lập tức
            self.fit(data, **kwargs)

    def __call__(self, X=None, Y=None, **kwargs):
        """Enable functional call pattern / Cho phép gọi hàm trực tiếp"""
        if kwargs.get('train', False) or (X is not None and not torch.is_tensor(X)):
            kwargs.pop('train', None)
            return self.fit(X, Y, **kwargs) # Auto-fit / Tự huấn luyện
        return super().__call__(X, **kwargs) # Standard forward / Lan truyền tiến chuẩn
        
    def fit(self, X, Y=None, epochs=200, batch_size=64, lr=2e-3, verbose=True):
        """Train DeepANM / Huấn luyện DeepANM"""
        from deepanm.models.trainer import DeepANMTrainer
        
        if Y is not None: # Combine bivariate / Gộp dữ liệu song biến
            X_all = np.hstack([X, Y]) if isinstance(X, np.ndarray) else torch.cat([X, Y], dim=1)
        else:
            X_all = X # Multivariate input / Đầu vào đa biến
            
        if self.core is None: # Dynamic re-init / Khởi tạo động
            from deepanm.core.gppom_hsic import GPPOMC_lnhsic_Core
            self.x_dim = X_all.shape[1]
            self.core = GPPOMC_lnhsic_Core(self.x_dim, self.y_dim, self.n_clusters, 
                                          self.hidden_dim, self.lda, self.device)
            self.to(self.device)

        self.trainer = DeepANMTrainer(self, lr=lr) # Create trainer / Tạo bộ huấn luyện
        self.history = self.trainer.train(X_all, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return self.history # Training log / Nhật ký huấn luyện
        
    def get_dag_matrix(self, threshold=0.1):
        """Get adjacency matrix / Lấy ma trận kề DAG"""
        with torch.no_grad():
            W = self.core.W_dag.detach().cpu().numpy() # Raw weights / Trọng số thô
            W_bin = (np.abs(W) > threshold).astype(float) # Binary mask / Mặt nạ nhị phân
            return W, W_bin # Returns both / Trả về cả hai
    
    def predict_clusters(self, X):
        """Map points to mechanisms / Phân điểm vào các cụm cơ chế"""
        self.eval() # Eval mode / Chế độ đánh giá
        if isinstance(X, np.ndarray): X = torch.from_numpy(X).float()
        with torch.no_grad():
            X = X.to(self.device)
            out = self.core.MLP(X) # Pass to MLP / Qua mạng MLP
            return out['z_soft'].argmax(dim=1).cpu().numpy() # Predicted labels / Nhãn dự báo

    def forward(self, x, temperature=1.0):
        """Forward pass through the core engine / Lan truyền tiến qua bộ xử lý cốt lõi"""
        return self.core(x, temperature=temperature)

    def get_residuals(self, X, use_pnl=True):
        """Compute residuals (Noise) / Tính toán phần dư (Nhiễu)"""
        self.eval()
        if isinstance(X, np.ndarray): X = torch.from_numpy(X).float()
        with torch.no_grad():
            X = X.to(self.device)
            out = self.core.MLP(X)
            z = out['z_soft'] # Get cluster soft labels / Lấy nhãn cụm mềm
            masked_input = X @ torch.abs(self.core.W_dag) # Structural focus / Tập trung cấu trúc
            phi = self.core.gp_phi_z(z) * self.core.gp_phi_x(masked_input) # Latent feature / Đặc trưng ẩn
            y_pred = self.core.linear_head(phi) # Prediction / Dự báo
            # Subtraction for noise / Phép trừ tính nhiễu
            residuals = (self.core.MLP.pnl_transform(X) if use_pnl else X) - y_pred
            return residuals.cpu().numpy() # Residuals / Phần dư

    def check_stability(self, X, n_splits=3):
        """Check mechanism stability / Kiểm tra sự ổn định cơ chế"""
        if isinstance(X, np.ndarray): X = torch.from_numpy(X).float()
        indices = np.arange(len(X))
        np.random.shuffle(indices) # Shuffle / Xáo trộn
        splits = np.array_split(indices, n_splits) # Split segments / Chia phân đoạn
        
        losses = []
        self.eval()
        with torch.no_grad():
            for split in splits:
                batch_x = X[split].to(self.device)
                total_loss, _, _ = self.core(batch_x) # Compute loss / Tính mất mát
                losses.append(total_loss.item())
        
        stability = np.std(losses) / (np.abs(np.mean(losses)) + 1e-8) # Variation ratio / Tỉ lệ biến thiên
        return stability, losses # Lower is better / Thấp hơn là tốt hơn

    def predict_counterfactual(self, x_orig, y_orig, x_new):
        """Counterfactual inference / Suy luận phản thực tế"""
        self.eval()
        with torch.no_grad():
            def get_y_pred(x_val, y_val_for_z):
                # Prepare tensors / Chuẩn bị tensor
                xt = torch.tensor([[x_val]]).float().to(self.device)
                yt = torch.tensor([[y_val_for_z]]).float().to(self.device)
                xy = torch.cat([xt, yt], dim=1)
                z = self.core.MLP(xy)['z_soft'] # Find mechanism / Tìm cơ chế
                phi = self.core.gp_phi_z(z) * self.core.gp_phi_x(xt) # GP features / Đặc trưng GP
                return self.core.linear_head(phi).item() # GP pred / Dự báo GP

            y_pred_orig = get_y_pred(x_orig, y_orig) # Pred at original / Dự báo ở gốc
            y_pred_new = get_y_pred(x_new, y_orig) # Pred at counterfactual / Dự báo phản thực tế
            
            y_cf = y_orig - y_pred_orig + y_pred_new # Abduction-Action-Prediction / Lý giải-Can thiệp-Dự báo
            return y_cf

    def predict_direction(self, data=None, lda=None):
        """Predict causal direction / Dự báo hướng nhân quả"""
        if data is None: # Use learned DAG weights / Dùng trọng số DAG đã học
            W, _ = self.get_dag_matrix()
            return 1 if W[0, 1] > W[1, 0] else -1
            
        # Hypothesis testing logic / Logic kiểm định giả thuyết
        if lda is None: lda = self.lda
        from deepanm.models.analysis import ANMMM_cd
        direction, _ = ANMMM_cd(data, lda=lda) # Run full analysis / Chạy phân tích đầy đủ
        return direction

