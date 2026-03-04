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
from src.models.deepanm import DeepANM

class DeepANMTrainer:
    """
    Training orchestrator for DeepANM.
    Uses AdamW optimizer with optional Augmented Lagrangian DAG penalty.

    Bộ điều phối huấn luyện cho DeepANM.
    Dùng optimizer AdamW với phạt DAG Lagrangian Tăng cường tùy chọn.
    """
    def __init__(self, model, lr=2e-3, weight_decay=1e-2):
        """Initialize optimizer. / Khởi tạo optimizer."""
        self.model = model
        # AdamW with L2 regularization via weight_decay
        # AdamW với điều chuẩn L2 qua weight_decay
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.history = {"loss": [], "nll": [], "hsic": [], "reg": []}
        
    def train(self, X, epochs=200, batch_size=64, verbose=True):
        """
        Standard mini-batch training loop with ALM DAG enforcement.

        ALM updates:
          - Every 10 epochs: if h(W) not improving → double rho (tighten constraint)
          - Otherwise: update alpha (Lagrange multiplier)

        Vòng lặp huấn luyện mini-batch chuẩn với ép DAG bằng ALM.

        Cập nhật ALM:
          - Mỗi 10 epochs: nếu h(W) không cải thiện → nhân đôi rho (siết ràng buộc)
          - Ngược lại: cập nhật alpha (nhân tử Lagrange)
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if verbose:
            print(f">> Training DeepANM on {self.model.device}")
            print(f"   Variables: {X.shape[1]}, Samples: {X.shape[0]}")
            print(f"   Mode: Augmented Lagrangian DAG (NOTEARS) + SplineFlow + VAE")
            print("-" * 60)
        
        # ALM hyperparameters / Siêu tham số ALM
        rho, alpha, h_val = 1.0, 0.0, float('inf')
        rho_max = 1e8
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = epoch_reg = epoch_hsic = epoch_nll = 0.0
            
            # Gumbel temperature annealing: 1.0 → 0.1 over training
            # Giảm nhiệt Gumbel: 1.0 → 0.1 trong suốt quá trình huấn luyện
            temperature = max(0.1, 1.0 - epoch / epochs)
            
            # ALM penalty update (only if ALM mode enabled)
            # Cập nhật phạt ALM (chỉ khi bật chế độ ALM)
            if self.model.core.use_alm:
                curr_h_val = self.model.core.get_dag_penalty(self.model.core.W_dag).item()
                if epoch > 0 and epoch % 10 == 0:
                    if curr_h_val > 0.25 * h_val:
                        rho = min(rho * 2.0, rho_max)  # Tighten / Siết chặt
                    else:
                        alpha += rho * curr_h_val       # Update multiplier / Cập nhật nhân tử
                    h_val = curr_h_val
            else:
                curr_h_val = 0.0
            
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.model.device)
                self.optimizer.zero_grad()
                
                out_loss, reg_loss, hsic_loss, nll_loss = self.model(batch_x, temperature=temperature)
                
                # Augmented Lagrangian: L + α·h(W) + 0.5·ρ·h(W)²
                if self.model.core.use_alm:
                    h_term = self.model.core.get_dag_penalty(self.model.core.W_dag)
                    alm_loss = out_loss + (alpha * h_term) + (0.5 * rho * h_term * h_term)
                else:
                    alm_loss = out_loss
                
                alm_loss.backward()
                
                # Gradient clipping to prevent explosion when rho is large
                # Cắt gradient để tránh nổ gradient khi rho lớn
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
                self.optimizer.step()
                
                epoch_loss += alm_loss.item()
                epoch_reg += reg_loss.item()
                epoch_hsic += hsic_loss.item()
                epoch_nll += nll_loss.item()
            
            n_batches = len(loader)
            self.history['loss'].append(epoch_loss / n_batches)
            self.history['nll'].append(epoch_nll / n_batches)
            self.history['reg'].append(epoch_reg / n_batches)
            self.history['hsic'].append(epoch_hsic / n_batches)
            
            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | ALM Loss: {epoch_loss/n_batches:.4f} | "
                      f"Reg: {epoch_reg/n_batches:.4f} | NLL: {epoch_nll/n_batches:.4f} | "
                      f"h(W): {curr_h_val:.6f} | Rho: {rho:.1e} | Alpha: {alpha:.2f}")
        
        return self.history
