import numpy as np
import pytest
import os
import matplotlib
# Use non-interactive backend for testing
matplotlib.use('Agg')

from src.models.deepanm import DeepANM
from src.models.fast_baseline import FastANM
from src.models.lite_baseline import LiteANM
from src.utils.visualize import plot_dag
from src.utils.adaptive_lasso import adaptive_lasso_dag


def test_fastanm_pipeline():
    """Verify FastANM logic works with and without RF/CI pruning"""
    np.random.seed(42)
    X = np.random.randn(30, 3)
    
    model = FastANM()
    W_fast = model.fit(X, verbose=False, use_rf=False, use_ci_pruning=False)
    assert W_fast.shape == (3, 3)
    
    # Test with Random Forest and CI
    W_fast_rf = model.fit(X, verbose=False, use_rf=True, use_ci_pruning=True)
    assert W_fast_rf.shape == (3, 3)


def test_liteanm_pipeline():
    """Verify LiteANM overrides fit defaults properly"""
    np.random.seed(42)
    X = np.random.randn(30, 3)
    
    model = LiteANM()
    # 2 epochs is enough to test if the engine compiles and runs the loop
    history = model.fit(X, epochs=2, batch_size=10, verbose=False)
    assert 'loss' in history
    assert len(history['loss']) == 2
    
    prob, ate = model.fit_bootstrap(X, n_bootstraps=2, epochs=2, batch_size=10, verbose=False)
    assert prob.shape == (3, 3)
    assert ate.shape == (3, 3)


def test_deepanm_bootstrap_and_ate():
    """Verify DeepANM bootstrap selection, ATE estimation and clustering predict"""
    np.random.seed(42)
    X = np.random.randn(30, 3)
    
    model = DeepANM(n_clusters=2, hidden_dim=16)
    
    # Fit Bootstrap with fast mode internally
    prob, ate_matrix = model.fit_bootstrap(
        X, n_bootstraps=2, epochs=2, batch_size=10, 
        verbose=False, discovery_mode="fast"
    )
    
    assert prob.shape == (3, 3)
    assert ate_matrix.shape == (3, 3)
    
    # Predict clusters
    clusters = model.predict_clusters(X)
    assert clusters.shape == (30,)
    assert set(clusters).issubset({0, 1})
    
    # Estimate specific ATE
    ate = model.estimate_ate(X, from_idx=0, to_idx=1)
    assert isinstance(ate, float)


def test_visualize_plot_dag(tmp_path):
    """Verify plot_dag function generates output without crashing"""
    W = np.array([
        [0.0, 1.5, 0.0],
        [0.0, 0.0, -0.8],
        [0.0, 0.0, 0.0]
    ])
    
    save_file = tmp_path / "test_graph.png"
    
    # Run plot_dag, it should not raise an exception and should create a file
    plot_dag(W_matrix=W, labels=['X', 'Y', 'Z'], threshold=0.1, save_path=str(save_file))
    
    assert os.path.exists(save_file)
    assert os.path.getsize(save_file) > 0


def test_adaptive_lasso_rf_ci():
    """Verify Adaptive LASSO with Random Forest and Partial Correlation Pruning"""
    np.random.seed(42)
    X0 = np.random.randn(50)
    X1 = X0 + np.random.randn(50) * 0.1
    X2 = X1 + np.random.randn(50) * 0.1
    X = np.column_stack([X0, X1, X2])
    
    order = [0, 1, 2]
    # use_rf=True invokes sklearn RF permutation importance
    # use_ci_pruning=True invokes HistGradientBoosting conditional tests
    W_bin = adaptive_lasso_dag(X, order, use_rf=True, use_ci_pruning=True)
    
    assert W_bin.shape == (3, 3)
    assert np.all(np.tril(W_bin) == 0)
