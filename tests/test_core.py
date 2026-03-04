import torch
import numpy as np
import pytest

from src.core.mlp import MLP, HeterogeneousNoiseModel, Encoder, ANM_SEM, Decoder
from src.core.gppom_hsic import GPPOMC_lnhsic_Core, FastHSIC
from src.core.toposort import hsic_greedy_order
from src.utils.adaptive_lasso import adaptive_lasso_dag
from src.models.deepanm import DeepANM


def test_mlp_components():
    """Verify input/output shapes of internal MLP components (Encoder, SEM, Decoder, MNN)"""
    batch_size = 10 
    input_dim = 5 
    n_clusters = 3 
    
    # Check Encoder & Gumbel Selection
    encoder = Encoder(input_dim, hidden_dim=16, n_mechanisms=n_clusters)
    x = torch.randn(batch_size, input_dim)
    feat, z_soft, kl_loss = encoder(x)
    assert z_soft.shape == (batch_size, n_clusters)
    assert kl_loss.dim() == 0  # Scalar loss
    
    # Check SEM
    sem = ANM_SEM(input_dim, hidden_dim=16, output_dim=input_dim)
    mu = sem(x)
    assert mu.shape == (batch_size, input_dim)
    
    # Check Decoder (Monotonic)
    decoder = Decoder(input_dim)
    y_trans = decoder(x)
    assert y_trans.shape == (batch_size, input_dim)
    x_inv = decoder.inverse(y_trans)
    # Inverse should ideally get back to exactly similar space, check shapes
    assert x_inv.shape == (batch_size, input_dim)

    # Full MLP integration
    mlp = MLP(input_dim=input_dim, hidden_dim=32, output_dim=input_dim, n_clusters=n_clusters)
    out = mlp(x) 
    assert out['z_soft'].shape == (batch_size, n_clusters)
    assert out['mu'].shape == (batch_size, input_dim)
    assert torch.allclose(out['z_soft'].sum(dim=1), torch.ones(batch_size))
    
    # NLL is always computed natively using noise proxy
    assert 'log_prob_noise' in out
    assert out['log_prob_noise'].shape == (batch_size,)


def test_heterogeneous_noise():
    """Verify DECI-inspired GMM output"""
    batch_size = 20
    dim = 4
    n_components = 5
    
    noise_model = HeterogeneousNoiseModel(dim=dim, n_components=n_components)
    fake_noise = torch.randn(batch_size, dim)
    log_prob = noise_model.compute_log_prob(fake_noise)
    
    assert log_prob.shape == (batch_size,)
    assert not torch.isnan(log_prob).any()
    assert not torch.isinf(log_prob).any()


def test_fast_hsic():
    """Verify RFF-based HSIC approximation"""
    x_dim = 3
    z_dim = 2
    n_samples = 50
    
    fast_hsic = FastHSIC(x_dim, z_dim, n_features=64)
    X = torch.randn(n_samples, x_dim)
    Z = torch.randn(n_samples, z_dim)
    
    hsic_val = fast_hsic(X, Z)
    assert hsic_val.item() >= 0.0
    assert not torch.isnan(hsic_val)


def test_dag_penalty():
    """Verify DAGMA Log-determinant DAG acyclicity penalty h(W)"""
    core = GPPOMC_lnhsic_Core(x_dim=2, y_dim=0, n_clusters=2, hidden_dim=16, lda=1.0, device='cpu')
    
    # Graph 1: Empty graph is acyclic
    core.W_val.data = torch.ones(2, 2)
    core.W_logits.data = torch.zeros(2, 2)
    assert core.get_dag_penalty(W_mask=torch.zeros(2, 2)).item() < 1e-6 
    
    # Graph 2: Chain 0 -> 1 is acyclic
    assert core.get_dag_penalty(W_mask=torch.tensor([[0.0, 1.0], [0.0, 0.0]])).item() < 1e-6
    
    # Graph 3: Cycle 0 -> 1 -> 0
    assert core.get_dag_penalty(W_mask=torch.tensor([[0.0, 1.0], [1.0, 0.0]])).item() > 0.5 
    
    # Graph 4: Self loop (should be naturally filtered by zero-diagonal in GPPOM constraint mask)
    assert core.get_dag_penalty(W_mask=torch.tensor([[1.0, 0.0], [0.0, 1.0]])).item() < 1e-6


def test_score_asymmetry():
    """Verify TopoSort logic using random noise (checking logic/shape bounds)"""
    n_vars = 4
    n_samples = 50
    X = np.random.randn(n_samples, n_vars)
    order = hsic_greedy_order(X, n_rff=64, verbose=False)
    
    # Must contain 0 to n_vars-1
    assert len(order) == n_vars
    assert set(order) == set(range(n_vars))


def test_gppomc_core_forward():
    """Verify the Core Engine correctly routes variables and computes all loss components"""
    batch_size = 10
    d = 4
    core = GPPOMC_lnhsic_Core(x_dim=d, y_dim=0, n_clusters=2, hidden_dim=16, lda=1.0, device='cpu')
    
    batch_data = torch.randn(batch_size, d)
    base_loss, loss_reg, loss_hsic_clu, loss_nll = core(batch_data)
    
    for loss in [base_loss, loss_reg, loss_hsic_clu, loss_nll]:
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar
        assert not torch.isnan(loss)


def test_global_ate_matrix():
    """Verify computation of Neural ATE Jacobian with Diagonals strictly zeroed"""
    batch_size = 20
    d = 3
    mlp = MLP(input_dim=d, hidden_dim=16, output_dim=d, n_clusters=2)
    x = torch.randn(batch_size, d)
    
    ate_matrix = mlp.get_global_ate_matrix(x)
    assert ate_matrix.shape == (d, d)
    
    diagonal = torch.diag(ate_matrix)
    assert torch.all(diagonal == 0.0)


def test_adaptive_lasso():
    """Verify Edge Selection Module works logically with given structural rules"""
    np.random.seed(42)
    n_samples = 50
    n_vars = 3
    # Chain: 0 -> 1 -> 2
    X0 = np.random.randn(n_samples)
    X1 = X0 + np.random.randn(n_samples) * 0.1
    X2 = X1 + np.random.randn(n_samples) * 0.1
    X = np.column_stack([X0, X1, X2])
    
    order = [0, 1, 2]
    # For speed in test, use OLS instead of Random Forest
    W_bin = adaptive_lasso_dag(X, order, use_rf=False, use_ci_pruning=False)
    
    # Checks dimensions and content
    assert W_bin.shape == (n_vars, n_vars)
    # Check strict topological constraint (lower triangle must be 0)
    assert np.all(np.tril(W_bin) == 0)


def test_deepanm_modes():
    """Verify Full Pipeline fits via different modes without crashing"""
    d = 3
    n_samples = 40
    data = np.random.randn(n_samples, d)
    
    # ALM Mode (pure deep learning)
    model1 = DeepANM()
    model1.fit(data, epochs=2, batch_size=20, verbose=False, discovery_mode="alm")
    W1, W_bin1 = model1.get_dag_matrix()
    assert W1.shape == (d, d)
    
    # Fast Mode (TopoSort + ALASSO + Neural SCM gate checking)
    model2 = DeepANM()
    model2.fit(data, epochs=2, batch_size=20, verbose=False, discovery_mode="fast")
    # Provide data for internal Jacobians computation
    ATE2, W_bin2 = model2.get_dag_matrix(X=data)
    assert ATE2.shape == (d, d)


def test_prior_constraint():
    """Verify the Exogenous prior constraint mechanism works properly to exclude incoming causal arrows"""
    d = 3
    model = DeepANM(x_dim=d)
    
    # By default, constraint mask allows any edge except diagonal
    # Mark node 1 as Exogenous (meaning NO incoming edges)
    model.set_exogenous([1])
    
    W_mask = model.core.constraint_mask.cpu().numpy()
    
    # Column 1 should be entirely 0.0 (no parent -> 1)
    assert np.all(W_mask[:, 1] == 0.0)
