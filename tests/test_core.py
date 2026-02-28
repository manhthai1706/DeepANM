import torch
import numpy as np
import pytest
from deepanm.core.mlp import MLP, HeterogeneousNoiseModel
from deepanm.core.gppom_hsic import GPPOMC_lnhsic_Core, FastHSIC
from deepanm.models.deepanm import DeepANM

def test_mlp_shapes():
    """Verify MLP output shapes / Kiểm tra kích thước đầu ra của MLP"""
    batch_size = 10 
    input_dim = 5 
    n_clusters = 3 
    
    mlp = MLP(input_dim=input_dim, hidden_dim=32, output_dim=input_dim, n_clusters=n_clusters)
    x = torch.randn(batch_size, input_dim) 
    
    # Forward without y
    out = mlp(x) 
    assert out['z_soft'].shape == (batch_size, n_clusters)
    assert out['mu'].shape == (batch_size, input_dim)
    assert torch.allclose(out['z_soft'].sum(dim=1), torch.ones(batch_size))
    
    # log_prob_noise is now always computed (self-supervised noise proxy)
    assert 'log_prob_noise' in out
    assert out['log_prob_noise'].shape == (batch_size,)

    # Forward with y uses exact noise instead of proxy
    y = torch.randn(batch_size, input_dim)
    out_with_y = mlp(x, y=y)
    assert 'log_prob_noise' in out_with_y
    assert out_with_y['log_prob_noise'].shape == (batch_size,)

def test_heterogeneous_noise_model():
    """Verify DECI-inspired GMM Flow / Kiểm tra mô hình nhiễu lai DECI"""
    batch_size = 20
    dim = 4
    n_components = 5
    
    noise_model = HeterogeneousNoiseModel(dim=dim, n_components=n_components)
    fake_noise = torch.randn(batch_size, dim)
    log_prob = noise_model.compute_log_prob(fake_noise)
    
    # Should output scalar per sample
    assert log_prob.shape == (batch_size,)
    
    # Probabilities should be valid numbers
    assert not torch.isnan(log_prob).any()
    assert not torch.isinf(log_prob).any()

def test_fast_hsic():
    """Verify FastHSIC functionality (O(dN) HSIC approximation)"""
    x_dim = 3
    z_dim = 2
    n_samples = 50
    
    fast_hsic = FastHSIC(x_dim, z_dim, n_features=64)
    X = torch.randn(n_samples, x_dim)
    Z = torch.randn(n_samples, z_dim)
    
    hsic_val = fast_hsic(X, Z)
    
    # HSIC is a squared norm approximation, must be non-negative
    assert hsic_val.item() >= 0.0
    assert not torch.isnan(hsic_val)

def test_dag_penalty():
    """Verify NOTEARS DAG penalty formulation / Kiểm tra hàm phạt chu trình h(W)"""
    core = GPPOMC_lnhsic_Core(x_dim=2, y_dim=0, n_clusters=2, hidden_dim=16, lda=1.0, device='cpu')
    
    # A zero matrix is acyclic (DAG)
    core.W_val.data = torch.ones(2, 2)
    core.W_logits.data = torch.zeros(2, 2) # Equivalent to W_dag = 0.5, let's just construct artificial mask for test
    assert core.get_dag_penalty(W_mask=torch.zeros(2, 2)).item() < 1e-6 
    
    # An acyclic chain (0 -> 1)
    assert core.get_dag_penalty(W_mask=torch.tensor([[0.0, 1.0], [0.0, 0.0]])).item() < 1e-6
    
    # A cyclic matrix (0 -> 1 -> 0)
    assert core.get_dag_penalty(W_mask=torch.tensor([[0.0, 1.0], [1.0, 0.0]])).item() > 0.5 
    
    # Diagonal element (self-loop) is physically masked out, penalty should be 0 because constraint_mask
    assert core.get_dag_penalty(W_mask=torch.tensor([[1.0, 0.0], [0.0, 1.0]])).item() < 1e-6

def test_gppomc_core_forward():
    """Verify forward logic of Core Engine (GPPOMC + FastHSIC + DECI NLL)"""
    batch_size = 10
    d = 4
    core = GPPOMC_lnhsic_Core(x_dim=d, y_dim=0, n_clusters=2, hidden_dim=16, lda=1.0, device='cpu')
    
    batch_data = torch.randn(batch_size, d)
    base_loss, loss_reg, loss_hsic_clu, loss_nll = core(batch_data)
    
    # Verify outputs are valid scalars
    for loss in [base_loss, loss_reg, loss_hsic_clu, loss_nll]:
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0 # scalar
        assert not torch.isnan(loss)

def test_global_ate_matrix():
    """Verify computation of Neural ATE Jacobian"""
    batch_size = 20
    d = 3
    mlp = MLP(input_dim=d, hidden_dim=16, output_dim=d, n_clusters=2)
    x = torch.randn(batch_size, d)
    
    ate_matrix = mlp.get_global_ate_matrix(x)
    
    # Output should be d x d
    assert ate_matrix.shape == (d, d)
    
    # Diagnonal should be explicitly 0.0 due to no self-intervention impact rule
    diagonal = torch.diag(ate_matrix)
    assert torch.all(diagonal == 0.0)

def test_deepanm_integration():
    """Verify full end-to-end wrapper functionality / Kiểm tra vòng đời Wrapper DeepANM"""
    d = 3
    n_samples = 50
    data = np.random.randn(n_samples, d)
    
    # Small epochs for testing
    model = DeepANM()
    model.fit(data, epochs=2, batch_size=25, verbose=False)
    
    # Verify adjacency matrix extractors (raw W, no X)
    W, W_bin = model.get_dag_matrix()
    assert W.shape == (d, d)
    assert W_bin.shape == (d, d)

    # Verify Adaptive LASSO-assisted matrix extractor
    ATE_np, W_bin_ate = model.get_dag_matrix(X=data)
    assert ATE_np.shape == (d, d)
    assert W_bin_ate.shape == (d, d)
    
    # Predict clusters
    clusters = model.predict_clusters(data)
    assert clusters.shape == (n_samples,)

