"""
Unified PyTorch Kernels Library / Thư viện Kernel thống nhất
Tối ưu chuyên sâu cho kiểm định tính độc lập thống kê (HSIC) trong học Nhân quả (Causal Discovery)

NOTE: RBFKernel được giữ lại cho các use-case tuỳ chỉnh bên ngoài,
nhưng DeepANM nội bộ dùng RFFGPLayer (trong gppom_hsic.py) vì nhanh hơn O(n²) → O(n*D).
LinearKernel dành cho kiểm định tuyến tính nhanh khi cần baseline.
"""

import torch
import torch.nn as nn

class RBFKernel(nn.Module):
    """
    Gaussian (RBF) Kernel / Hàm nhân Gaussian
    Bổ sung tính năng tự động ước lượng dải thông (Bandwidth) qua Median Heuristic.
    Đây là kỹ thuật bắt buộc để đo đạc độ phụ thuộc phi tuyến (HSIC) chính xác trong bài toán ANM/PNL, 
    tránh hiện tượng sụp đổ gradient (Vanishing gradients) ở Kernel.
    """
    def __init__(self, input_dim, ARD=False, use_median_heuristic=True):
        super().__init__()
        self.use_median_heuristic = use_median_heuristic
        self.log_alpha = nn.Parameter(torch.zeros(1)) # Biên độ Kernel
        self.log_gamma = nn.Parameter(torch.zeros(input_dim if ARD else 1)) # Băng thông tự học thêm
        self.ARD = ARD # Tùy chọn trọng số cho từng chiều độc lập (Automatic Relevance Determination)

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1
            
        gamma_learned = torch.exp(self.log_gamma)
        
        x1_scaled = x1 * torch.sqrt(gamma_learned)
        x2_scaled = x2 * torch.sqrt(gamma_learned)

        dist_sq = torch.cdist(x1_scaled, x2_scaled, p=2)**2
        
        gamma_baseline = 1.0
        if self.use_median_heuristic and x1 is x2:
            with torch.no_grad():
                n = dist_sq.shape[0]
                if n > 1:
                    triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=dist_sq.device), diagonal=1)
                    median_dist_sq = torch.median(dist_sq[triu_mask])
                    if median_dist_sq.item() > 0:
                        gamma_baseline = 1.0 / median_dist_sq
        
        alpha = torch.exp(self.log_alpha)
        return alpha * torch.exp(-0.5 * gamma_baseline * dist_sq)
