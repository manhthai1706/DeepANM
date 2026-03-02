"""
Training Script for DeepANM / Script huấn luyện cho DeepANM
Handles model training and optimization / Xử lý huấn luyện và tối ưu hóa mô hình
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from deepanm.models.deepanm import DeepANM

class DeepANMTrainer:
    """Trainer class for DeepANM / Lớp điều phối huấn luyện cho DeepANM"""
    def __init__(self, model, lr=2e-3, weight_decay=1e-2):
        """Initialize optimizer / Khởi tạo trình tối ưu hóa"""
        self.model = model # Assigned model / Mô hình được chỉ định
        # AdamW with weight decay / AdamW kèm suy giảm trọng số (regularization)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.history = { # Metrics tracker / Theo dõi các chỉ số
            "loss": [],
            "nll": [],
            "hsic": [],
            "reg": []
        }
        
    def train(self, X, epochs=200, batch_size=64, verbose=True):
        """Standard training loop with Augmented Lagrangian Method / Vòng lặp huấn luyện chuẩn kèm phương pháp ALM"""
        if isinstance(X, np.ndarray): 
            X = torch.from_numpy(X).float()
        
        dataset = TensorDataset(X) 
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if verbose: 
            print(f">> Training DeepANM on {self.model.device}")
            print(f"   Variables: {X.shape[1]}, Samples: {X.shape[0]}")
            print(f"   Mode: Augmented Lagrangian DAG (NOTEARS) + SplineFlow + VAE")
            print("-" * 60)
        
        # --- Khởi tạo siêu tham số Augmented Lagrangian (ALM) ---
        rho, alpha, h_val = 1.0, 0.0, float('inf')
        rho_max = 1e8
        
        self.model.train() 
        for epoch in range(epochs): 
            epoch_loss = 0.0
            epoch_reg = 0.0
            epoch_hsic = 0.0
            epoch_nll = 0.0
            
            # Gumbel Temperature Decay: anneal from 1.0 → 0.1 for hard binary edge decisions
            temperature = max(0.1, 1.0 - epoch / epochs)
            
            # DAGMA DAG penalty h(W): computed once per epoch for ALM update
            curr_h_val = self.model.core.get_dag_penalty(self.model.core.W_dag).item()
            
            # Cập nhật Rho (Multiplier cho L2) và Alpha (Multiplier cho Lagrangian) mỗi 10 epoch 
            # để ép nghiệm bài toán dần đi về h(W) = 0
            if epoch > 0 and epoch % 10 == 0:
                if curr_h_val > 0.25 * h_val: 
                    rho = min(rho * 2.0, rho_max)
                else: 
                    alpha += rho * curr_h_val
                h_val = curr_h_val
            
            for (batch_x,) in loader: 
                batch_x = batch_x.to(self.model.device) 
                
                self.optimizer.zero_grad() 
                
                # Gọi Core Model để lấy bộ Loss cơ bản
                out_loss, reg_loss, hsic_loss, nll_loss = self.model(
                    batch_x, temperature=temperature
                )
                
                # Lấy Penalty h(W) sinh động trên Computation Graph
                h_term = self.model.core.get_dag_penalty(self.model.core.W_dag)
                
                # Hàm mục tiêu mới với Augmented Lagrangian
                # Obj = F(W) + alpha * h(W) + 0.5 * rho * h(W)^2
                alm_loss = out_loss + (alpha * h_term) + (0.5 * rho * h_term * h_term)
                
                alm_loss.backward() 
                
                # Phụ gia Gradient Clipping chống Nổ Gradient khi rho quá lớn
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
                      f"Reg: {epoch_reg/n_batches:.4f} | NLL: {epoch_nll/n_batches:.4f} | h(W): {curr_h_val:.6f} | "
                      f"Rho: {rho:.1e} | Alpha: {alpha:.2f}")
        
        return self.history 

