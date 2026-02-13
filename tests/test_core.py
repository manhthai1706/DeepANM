import torch
import numpy as np
import pytest
from deepanm.core.mlp import MLP
from deepanm.core.hsic import hsic_gam
from deepanm.core.gppom_hsic import GPPOMC_lnhsic_Core, FastHSIC
from deepanm.models.deepanm import DeepANM

def test_mlp_shapes():
    """Verify MLP output shapes / Kiểm tra kích thước đầu ra của MLP"""
    batch_size = 10 # Sample size / Số lượng mẫu
    input_dim = 5 # Num variables / Số lượng biến
    n_clusters = 3 # Num mechanisms / Số lượng cơ chế
    # Initialize MLP / Khởi tạo mạng MLP
    mlp = MLP(input_dim=input_dim, hidden_dim=32, output_dim=input_dim, n_clusters=n_clusters)
    
    x = torch.randn(batch_size, input_dim) # Random input / Đầu vào ngẫu nhiên
    out = mlp(x) # Forward pass / Chạy qua mạng
    
    # Check cluster shape / Kiểm tra hình dạng cụm
    assert out['z_soft'].shape == (batch_size, n_clusters)
    # Check prediction shape / Kiểm tra hình dạng dự báo
    assert out['mu'].shape == (batch_size, input_dim)
    # Probabilities should sum to 1 / Tổng xác suất phải bằng 1
    assert torch.allclose(out['z_soft'].sum(dim=1), torch.ones(batch_size))

def test_hsic_consistency():
    """Verify HSIC identifies independence / Kiểm tra tính độc lập của HSIC"""
    n = 200 # Num samples / Số lượng mẫu
    x = np.random.randn(n, 1) # Independent X / Biến X độc lập
    y = np.random.randn(n, 1) # Independent Y / Biến Y độc lập (không phụ thuộc X)
    
    stat, _, p = hsic_gam(x, y) # Run HSIC test / Chạy kiểm định HSIC
    # p-value > 0.05 means we accept H0 (Independence) / p > 0.05 nghĩa là chấp nhận giả thuyết độc lập
    assert p > 0.05 

def test_dag_penalty():
    """Verify NOTEARS DAG penalty / Kiểm tra hàm phạt đồ thị DAG"""
    # Core model setup / Cài đặt mô hình cốt lõi
    core = GPPOMC_lnhsic_Core(x_dim=2, y_dim=0, n_clusters=2, hidden_dim=16, lda=1.0, device='cpu')
    
    # Zero matrix is a valid DAG / Ma trận không là một DAG hợp lệ
    core.W_dag.data = torch.zeros(2, 2)
    assert core.get_dag_penalty() < 1e-6 # Penalty should be ~0 / Điểm phạt phải xấp xỉ 0
    
    # Matrix with cycle (0->1->0) / Ma trận có chu trình (vòng lặp)
    core.W_dag.data = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert core.get_dag_penalty() > 0.5 # Penalty should be high / Điểm phạt phải cao

def test_model_integration():
    """Verify DeepANM full integration / Kiểm tra tích hợp toàn diện DeepANM"""
    data = np.random.randn(50, 2) # Sample data / Dữ liệu mẫu
    model = DeepANM(data=data, epochs=2) # Run 2 epochs / Chạy thử 2 vòng lặp
    
    W, W_bin = model.get_dag_matrix() # Get adjacency / Lấy ma trận kề
    assert W.shape == (2, 2) # Correct weight shape / Hình dạng trọng số đúng
    assert W_bin.shape == (2, 2) # Correct binary shape / Hình dạng nhị phân đúng
    
    res = model.get_residuals(data) # Compute residuals / Tính toán phần dư
    assert res.shape == (50, 2) # Correct residual shape / Hình dạng phần dư đúng
