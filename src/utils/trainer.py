"""
DeepANM Trainer: Training loop with Augmented Lagrangian Method (ALM).
Handles DAG acyclicity enforcement, Gumbel temperature annealing,
and gradient clipping for stable optimization.

Bộ Huấn luyện DeepANM: Vòng lặp huấn luyện với Phương pháp Lagrangian Tăng cường (ALM).
Xử lý ép phi chu trình DAG, giảm nhiệt độ Gumbel và cắt gradient để tối ưu hóa ổn định.
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class DeepANMTrainer:
    """
    Training orchestrator for DeepANM using AdamW and Augmented Lagrangian constraints.
    Bộ điều phối huấn luyện cho DeepANM sử dụng AdamW và ràng buộc Lagrangian Tăng cường.
    """
    def __init__(self, model, lr=2e-3, weight_decay=1e-2):
        """Initialize optimizer and history metrics. / Khởi tạo optimizer và bộ ghi chỉ số."""
        self.model = model
        # AdamW for robust optimization with weight decay / AdamW giúp tối ưu hóa bền vững kèm điều chuẩn
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.history = {"loss": [], "nll": [], "hsic": [], "reg": [], "h_val": []}
        
    def train(self, X, epochs=200, batch_size=64, verbose=True):
        """
        Executes mini-batch training with dynamic ALM constraint updates.
        Thực hiện huấn luyện mini-batch với cập nhật ràng buộc ALM động.

        ALM Scheduling / Lịch trình ALM:
          - Every 10 epochs: update rho (penalty factor) or alpha (multiplier).
          - Mỗi 10 epochs: cập nhật rho (hệ số phạt) hoặc alpha (nhân tử).
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float() # Convert numpy to torch tensor / Chuyển numpy sang torch tensor
        
        dataset = TensorDataset(X) # Create dataset / Tạo tập dữ liệu
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # Data loader for batching / Bộ nạp dữ liệu theo lô
        
        if verbose:
            print(f">> Training DeepANM on {self.model.device}")
            print(f"   Variables: {X.shape[1]}, Samples: {X.shape[0]}")
            print(f"   Mode: Augmented Lagrangian DAG (DAGMA) + SplineFlow + VAE")
            print("-" * 60)
        
        # Initialize ALM penalty variables / Khởi tạo các biến phạt ALM
        rho, alpha, h_val = 1.0, 0.0, float('inf')
        rho_max = 1e8 # Maximum ceiling for penalty / Trần tối đa cho hệ số phạt
        
        self.model.train() # Set model to training mode / Chuyển mô hình sang chế độ huấn luyện
        for epoch in range(epochs):
            epoch_loss = epoch_reg = epoch_hsic = epoch_nll = 0.0
            
            # Gumbel temperature annealing (reduces stochasticity over time)
            # Giảm nhiệt độ Gumbel (giảm tính ngẫu nhiên theo thời gian)
            temperature = max(0.1, 1.0 - epoch / epochs)
            
            # Augmented Lagrangian schedule: check acyclicity h(W) progress
            # Lịch trình ALM: kiểm tra tiến độ của độ đo chu trình h(W)
            if self.model.core.use_alm:
                # Current DAG penalty value / Giá trị phạt DAG hiện tại
                curr_h_val = self.model.core.get_dag_penalty(self.model.core.W_dag).item()
                if epoch > 0 and epoch % 10 == 0:
                    # If penalty hasn't decreased sufficiently, tighten the constraint
                    # Nếu giá trị phạt không giảm đủ nhanh, hãy siết chặt ràng buộc
                    if curr_h_val > 0.25 * h_val:
                        rho = min(rho * 2.0, rho_max) # Increase penalty factor / Tăng hệ số phạt
                    else:
                        alpha += rho * curr_h_val      # Update Lagrange multiplier / Cập nhật nhân tử Lagrange
                    h_val = curr_h_val # Update best observed h(W) / Cập nhật h(W) tốt nhất từng thấy
            else:
                curr_h_val = 0.0
            
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.model.device) # Move data to GPU/CPU / Chuyển dữ liệu lên thiết bị
                self.optimizer.zero_grad() # Clear previous gradients / Xóa gradient cũ
                
                # Forward pass: compute losses / Lan truyền tiến: tính toán các tổn thất
                out_loss, reg_loss, hsic_loss, nll_loss = self.model(batch_x, temperature=temperature)
                
                # Total loss with ALM constraint: Loss + Alpha * h(W) + 0.5 * Rho * h(W)^2
                # Tổng tổn thất kèm ràng buộc ALM: Loss + Alpha * h(W) + 0.5 * Rho * h(W)^2
                if self.model.core.use_alm:
                    h_term = self.model.core.get_dag_penalty(self.model.core.W_dag)
                    alm_loss = out_loss + (alpha * h_term) + (0.5 * rho * h_term * h_term)
                else:
                    alm_loss = out_loss
                
                alm_loss.backward() # Backpropagation / Lan truyền ngược
                
                # Gradient clipping to prevent exploding gradients when Rho is high
                # Cắt gradient để tránh hiện tượng nổ gradient khi Rho lớn
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
                self.optimizer.step() # Update weights / Cập nhật trọng số
                
                # Accumulate logs / Tích lũy nhật ký
                epoch_loss += alm_loss.item()
                epoch_reg += reg_loss.item()
                epoch_hsic += hsic_loss.item()
                epoch_nll += nll_loss.item()
            
            # Store epoch metrics / Lưu chỉ số epoch
            n_batches = len(loader)
            self.history['loss'].append(epoch_loss / n_batches)
            self.history['nll'].append(epoch_nll / n_batches)
            self.history['reg'].append(epoch_reg / n_batches)
            self.history['hsic'].append(epoch_hsic / n_batches)
            self.history['h_val'].append(curr_h_val)
            
            # Display status periodically / Hiển thị trạng thái định kỳ
            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | ALM Loss: {epoch_loss/n_batches:.4f} | "
                      f"Reg: {epoch_reg/n_batches:.4f} | NLL: {epoch_nll/n_batches:.4f} | "
                      f"h(W): {curr_h_val:.6f} | Rho: {rho:.1e} | Alpha: {alpha:.2f}")
        
        return self.history # Return full training history / Trả về lịch sử huấn luyện đầy đủ
