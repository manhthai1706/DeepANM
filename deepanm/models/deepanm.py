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
            if hasattr(self, 'exog_indices') and self.exog_indices is not None:
                self.set_exogenous(self.exog_indices)
            self.to(self.device)

        self.trainer = DeepANMTrainer(self, lr=lr) # Create trainer / Tạo bộ huấn luyện
        self.history = self.trainer.train(X_all, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return self.history # Training log / Nhật ký huấn luyện
        
    def get_dag_matrix(self, threshold=0.1, X=None):
        """Get adjacency matrix / Lấy ma trận kề DAG. Nếu truyền vào X, ta sẽ dùng thuật toán Neural ATE (Jacobian) để đo tác động thật"""
        with torch.no_grad():
            W_dag_masked = self.core.W_dag * self.core.constraint_mask
            W = W_dag_masked.detach().cpu().numpy() # Mảng W cơ bản
            
            if X is not None:
                # Ứng dụng ATE vào Khám phá toàn cục 
                ATE_matrix = self.core.MLP.get_global_ate_matrix(X, W_dag=W_dag_masked)
                ATE_np = ATE_matrix.cpu().numpy()
                
                # Biến i tác động lên j => cần Phép AND giữa Topological W_dag_MASK (Logic đồ thị) và ATE >= threshold
                # Bởi vì Neural Net luôn có ATE nhiễu rò rỉ, ta chỉ xác nhận nó là Cạnh nếu W_dag của NOTEARS đã mở cổng đó.
                W_bin = ((np.abs(W) > threshold) & (np.abs(ATE_np) > 1e-4)).astype(float)
                
                return ATE_np, W_bin # Xuất Mảng tác động Causal ATE Matrix đại diện cho W_raw
            else:
                W_bin = (np.abs(W) > threshold).astype(float) 
                return W, W_bin # Returns both / Trả về cả hai

    def set_exogenous(self, exog_indices):
        """Đánh dấu các biến thuộc nhóm ngoại sinh (Exogenous), cấm nhận cạnh hướng vào."""
        self.exog_indices = exog_indices
        if self.core is not None:
            # Set cột tại index về 0 (đóng mọi input/parent vào biến này)
            for idx in exog_indices:
                self.core.constraint_mask[:, idx] = 0.0
    
    def predict_clusters(self, X):
        """Map points to mechanisms / Phân điểm vào các cụm cơ chế"""
        self.eval() # Eval mode / Chế độ đánh giá
        if isinstance(X, np.ndarray): X = torch.from_numpy(X).float()
        with torch.no_grad():
            X = X.to(self.device)
            out = self.core.MLP(X) # Pass to MLP / Qua mạng MLP
            return out['z_soft'].argmax(dim=1).cpu().numpy() # Predicted labels / Nhãn dự báo

    def fit_bootstrap(self, X, n_bootstraps=5, threshold=0.015, epochs=250, batch_size=64, lr=5e-3, verbose=True):
        """Train DeepANM using Bootstrap Stability Selection / Huấn luyện DeepANM với Bootstrap để tìm cơ chế vững"""
        if isinstance(X, torch.Tensor):
            X_numpy = X.cpu().numpy()
        else:
            X_numpy = X
            
        n_samples, n_vars = X_numpy.shape
        self.aggregated_W = np.zeros((n_vars, n_vars))
        self.aggregated_bin = np.zeros((n_vars, n_vars))
        
        for b in range(n_bootstraps):
            if verbose: print(f"[DeepANM Bootstrap] Đang tiến hành Re-sampling lấy mẫu và mô phỏng vũ trụ {b+1}/{n_bootstraps}...")
            
            # Re-sampling có hoàn lại (mô phỏng thế giới song song)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_data = X_numpy[indices]
            
            # Tái tạo hoàn toàn Core Engine để không Overfit Prior
            self.core = None 
            self.fit(boot_data, epochs=epochs, batch_size=batch_size, lr=lr, verbose=False)
            
            # Khởi chạy phân tích ATE Matrix (Global Discovery) bằng cách truyền boot_data vào
            W_raw, W_bin = self.get_dag_matrix(threshold=threshold, X=boot_data)
            self.aggregated_W += W_raw
            self.aggregated_bin += W_bin
            
        prob_matrix = self.aggregated_bin / n_bootstraps
        avg_weight_matrix = self.aggregated_W / n_bootstraps
        
        return prob_matrix, avg_weight_matrix

    def forward(self, x, temperature=1.0):
        """Forward pass through the core engine / Lan truyền tiến qua bộ xử lý cốt lõi"""
        return self.core(x, temperature=temperature)

    def get_residuals(self, X, use_pnl=True):
        """Compute residuals (Noise) / Tính toán phần dư (Nhiễu)"""
        self.eval()
        if not torch.is_tensor(X): 
            X = torch.as_tensor(X, dtype=torch.float32)
        
        with torch.no_grad():
            X = X.to(self.device)
            out = self.core.MLP(X)
            
            # Áp dụng chuẩn Masking đã tối ưu từ Core Constraint
            W_dag_masked = self.core.W_dag * self.core.constraint_mask
            masked_input = X @ torch.abs(W_dag_masked)
            
            phi = self.core.gp_phi_z(out['z_soft']) * self.core.gp_phi_x(masked_input)
            y_pred = self.core.linear_head(phi)
            
            # Tính phần dư theo phương trình PNL
            h_y = self.core.MLP.pnl_transform(X) if use_pnl else X
            residuals = h_y - y_pred
            
            return residuals.cpu().numpy()

    def check_stability(self, X, n_splits=3):
        """Check mechanism stability / Kiểm tra sự ổn định cơ chế"""
        if not torch.is_tensor(X): 
            X = torch.as_tensor(X, dtype=torch.float32)
            
        n_samples = X.shape[0]
        # Xáo trộn index trên RAM nhẹ thay vì chia và copy  tensor
        indices = torch.randperm(n_samples)
        splits = torch.tensor_split(indices, n_splits)
        
        losses = []
        self.eval()
        with torch.no_grad():
            X_dev = X.to(self.device) # Đưa toàn bộ lên Device 1 lần duy nhất
            for split in splits:
                batch_x = X_dev[split]
                total_loss, _, _ = self.core(batch_x)
                losses.append(total_loss.item())
        
        losses = np.array(losses)
        stability = np.std(losses) / (np.abs(np.mean(losses)) + 1e-8)
        return stability, losses

    def predict_counterfactual(self, x_orig, y_orig, x_new):
        """Counterfactual inference / Suy luận phản thực tế"""
        self.eval()
        with torch.no_grad():
            def get_y_pred(x_val, y_val_for_z):
                # Prepare tensors / Chuẩn bị tensor (Tránh tạo node đạo hàm)
                xt = torch.tensor([[x_val]], dtype=torch.float32, device=self.device)
                yt = torch.tensor([[y_val_for_z]], dtype=torch.float32, device=self.device)
                xy = torch.cat([xt, yt], dim=1)
                
                z = self.core.MLP(xy)['z_soft']
                
                # Áp dụng đúng công thức Masking để đảm bảo tính toán đồng nhất
                W_dag_masked = self.core.W_dag * self.core.constraint_mask
                masked_input = xy @ torch.abs(W_dag_masked)
                
                phi = self.core.gp_phi_z(z) * self.core.gp_phi_x(masked_input)
                return self.core.linear_head(phi).item()

            y_pred_orig = get_y_pred(x_orig, y_orig)
            y_pred_new = get_y_pred(x_new, y_orig)
            
            y_cf = y_orig - y_pred_orig + y_pred_new
            return y_cf

    def estimate_ate(self, X_control, X_treatment):
        """Wrapper tính toán Average Treatment Effect cho người dùng."""
        return self.core.MLP.estimate_ate(X_control, X_treatment)

    def predict_direction(self, data=None):
        """Predict causal direction based on DAG weights for 2-variable systems / Dự báo hướng nhân quả song biến"""
        W, _ = self.get_dag_matrix(X=data)
        return 1 if W[0, 1] > W[1, 0] else -1

