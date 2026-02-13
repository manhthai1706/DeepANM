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
            "likelihood": [],
            "hsic": []
        }
        
    def train(self, X, epochs=200, batch_size=64, verbose=True):
        """Standard training loop / Vòng lặp huấn luyện chuẩn"""
        if isinstance(X, np.ndarray): # Array to tensor / Chuyển mảng sang tensor
            X = torch.from_numpy(X).float()
        
        dataset = TensorDataset(X) # Wrapper / Gói dữ liệu
        # Parallel data loader / Trình tải dữ liệu song song
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if verbose: # Print info / In thông tin
            print(f">> Training DeepANM on {self.model.device}")
            print(f"   Variables: {X.shape[1]}, Samples: {X.shape[0]}")
            print(f"   Mode: DAG (NOTEARS) + SplineFlow + VAE Latents")
            print("-" * 60)
        
        self.model.train() # Set status to Train / Đặt trạng thái Huấn luyện
        for epoch in range(epochs): # Epoch loop / Vòng lặp Epoch
            epoch_loss = 0.0
            epoch_reg = 0.0
            epoch_hsic = 0.0
            
            # Annealing temperature / Giảm dần nhiệt độ cho Gumbel-Softmax
            temperature = max(0.5, 1.0 - epoch / epochs)
            
            for (batch_x,) in loader: # Mini-batch loop / Vòng lặp mini-batch
                batch_x = batch_x.to(self.model.device) # Move to GPU/CPU / Chuyển vào GPU/CPU
                
                self.optimizer.zero_grad() # Clear gradients / Xóa gradient
                # Core forward pass / Lan truyền tiến cốt lõi
                total_loss, reg_loss, hsic_loss = self.model(
                    batch_x, temperature=temperature
                )
                
                total_loss.backward() # Path back / Lan truyền ngược
                self.optimizer.step() # Update weights / Cập nhật trọng số
                
                epoch_loss += total_loss.item() # Accumulate loss / Tích lũy mất mát
                epoch_reg += reg_loss.item() # Accumulate MSE / Tích lũy MSE
                epoch_hsic += hsic_loss.item() # Accumulate HSIC / Tích lũy HSIC
            
            n_batches = len(loader) # Batch count / Số lượng batch
            if verbose and (epoch + 1) % 50 == 0: # Periodic logging / Ghi nhật ký định kỳ
                print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {epoch_loss/n_batches:.4f} | "
                      f"Reg: {epoch_reg/n_batches:.4f} | HSIC: {epoch_hsic/n_batches:.6f}")
        
        return self.history # Training summary / Tổng kết huấn luyện

def train_deepanm(X, Y, x_dim, y_dim, n_clusters=2, hidden_dim=64, 
                   lda=1.0, epochs=200, batch_size=64, lr=2e-3, 
                   device=None, verbose=True):
    """Convenience helper for training / Hàm hỗ trợ huấn luyện nhanh"""
    model = DeepANM(
        x_dim=x_dim, y_dim=y_dim, n_clusters=n_clusters,
        hidden_dim=hidden_dim, lda=lda, device=device
    ) # Instantiate / Khởi tạo thực thể
    
    trainer = DeepANMTrainer(model, lr=lr) # Setup trainer / Cài đặt bộ huấn luyện
    # Start process / Bắt đầu quy trình
    history = trainer.train(X, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    return model, trainer, history # Return results / Trả về kết quả


