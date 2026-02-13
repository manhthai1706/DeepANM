import torch
import numpy as np
import pytest
from deepanm.core.mlp import MLP
from deepanm.core.hsic import hsic_gam
from deepanm.core.gppom_hsic import GPPOMC_lnhsic_Core, FastHSIC
from deepanm.models.deepanm import DeepANM

def test_mlp_shapes():
    """Kiểm tra output của MLP backbone"""
    batch_size = 10
    input_dim = 5
    n_clusters = 3
    mlp = MLP(input_dim=input_dim, hidden_dim=32, output_dim=input_dim, n_clusters=n_clusters)
    
    x = torch.randn(batch_size, input_dim)
    out = mlp(x)
    
    assert out['z_soft'].shape == (batch_size, n_clusters)
    assert out['mu'].shape == (batch_size, input_dim)
    assert torch.allclose(out['z_soft'].sum(dim=1), torch.ones(batch_size))

def test_hsic_consistency():
    """Kiểm tra tính đúng đắn của HSIC (hai biến độc lập hsic xấp xỉ 0)"""
    n = 200
    x = np.random.randn(n, 1)
    y = np.random.randn(n, 1) # Độc lập với x
    
    stat, _, p = hsic_gam(x, y)
    # Với dữ liệu độc lập, p-value thường > 0.05
    assert p > 0.05 

def test_dag_penalty():
    """Kiểm tra hàm phạt DAG (h(W) = 0 nếu W là ma trận tam giác trên/dưới)"""
    core = GPPOMC_lnhsic_Core(x_dim=2, y_dim=0, n_clusters=2, hidden_dim=16, lda=1.0, device='cpu')
    
    # Ma trận zero (là DAG)
    core.W_dag.data = torch.zeros(2, 2)
    assert core.get_dag_penalty() < 1e-6
    
    # Ma trận có vòng lặp (1->0->1)
    core.W_dag.data = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert core.get_dag_penalty() > 0.5

def test_model_integration():
    """Test luồng tích hợp của DeepANM class"""
    data = np.random.randn(50, 2)
    model = DeepANM(data=data, epochs=2) # Chạy vài epoch kiểm tra crash
    
    W, W_bin = model.get_dag_matrix()
    assert W.shape == (2, 2)
    assert W_bin.shape == (2, 2)
    
    res = model.get_residuals(data)
    assert res.shape == (50, 2)
