"""
Unified PyTorch Kernels Library / Thư viện Kernel thống nhất
Tối ưu chuyên sâu cho kiểm định tính độc lập thống kê (HSIC) trong học Nhân quả (Causal Discovery)
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
        
        # Áp dụng trọng số chiều nếu sử dụng đồ thị tính toán ARD
        x1_scaled = x1 * torch.sqrt(gamma_learned)
        x2_scaled = x2 * torch.sqrt(gamma_learned)

        # Trích xuất ma trận khoảng cách bình phương (L2 Squared)
        dist_sq = torch.cdist(x1_scaled, x2_scaled, p=2)**2
        
        # ----------------------------------------------------
        # MEDIAN HEURISTIC: ĐỘC QUYỀN TRONG KIỂM ĐỊNH NHÂN QUẢ 
        # Căn chỉnh ma trận tự động để HSIC luôn hội tụ.
        # ----------------------------------------------------
        gamma_baseline = 1.0
        if self.use_median_heuristic and x1 is x2:
            with torch.no_grad(): # Không truyền đạo hàm qua tác vụ tìm ngưỡng Heuristic
                n = dist_sq.shape[0]
                if n > 1:
                    # Lấy phân nửa ma trận tam giác trên để tìm giá trị trung vị
                    triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=dist_sq.device), diagonal=1)
                    median_dist_sq = torch.median(dist_sq[triu_mask])
                    if median_dist_sq.item() > 0:
                        gamma_baseline = 1.0 / median_dist_sq
        
        alpha = torch.exp(self.log_alpha)
        
        # Trả về ma trận Gram (Kernel Matrix) theo chuẩn học máy nhân quả
        return alpha * torch.exp(-0.5 * gamma_baseline * dist_sq)

class LinearKernel(nn.Module):
    """
    Linear Kernel: K(x, y) = scale * x^T y + bias / Hàm nhân tuyến tính
    Được tối ưu để trích xuất tương quan tuyến tính vững và nhẹ nhàng.
    """
    def __init__(self, input_dim=None):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1)) # Tỉ lệ (Scale)
        self.log_bias = nn.Parameter(torch.zeros(1))  # Chệch (Bias)

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        scale = torch.exp(self.log_scale)
        bias = torch.exp(self.log_bias)
        
        # Phương pháp nhân vô hướng siêu nhanh trên PyTorch
        return scale * (x1 @ x2.T) + bias
