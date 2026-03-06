import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.deepanm import DeepANM
from src.utils.trainer import DeepANMTrainer

def generate_synthetic_data(n_samples=800, n_vars=10):
    """
    Generate complex 10-node synthetic causal data (High Non-linearity).
    Structure: complex DAG with colliders, forks, and long skip connections.
    """
    np.random.seed(42)
    # Root nodes
    X0 = np.random.normal(0, 1, n_samples)
    X1 = np.random.normal(0, 1, n_samples)
    
    # Complex interactions
    # X2 is a collider: f(X0, X1)
    X2 = 0.5 * (X0**2) + np.sin(X1) + np.random.normal(0, 0.1, n_samples)
    
    # X3 is a simple transformation of X0
    X3 = np.tanh(X0) + np.random.normal(0, 0.1, n_samples)
    
    # X4 follows X2 (Non-monotonic)
    X4 = np.exp(-(X2**2)) + np.random.normal(0, 0.1, n_samples)
    
    # X5 is a fork from X1 and X3 (Multiplicative interaction)
    X5 = X1 * X3 + np.random.normal(0, 0.1, n_samples)
    
    # X6 combines X4 and X5
    X6 = np.cos(X4) + (X5**2) + np.random.normal(0, 0.1, n_samples)
    
    # X7 is a long skip connection from X0 and X6
    X7 = np.sqrt(np.abs(X0)) + np.sin(X6) + np.random.normal(0, 0.1, n_samples)
    
    # X8 is a deep node
    X8 = (X7**3) - X7 + np.random.normal(0, 0.1, n_samples)
    
    # X9 is the final sink node combining X8 and X3
    X9 = 1.0 / (1.0 + np.exp(-X8)) + X3 + np.random.normal(0, 0.1, n_samples)
    
    data = np.stack([X0, X1, X2, X3, X4, X5, X6, X7, X8, X9], axis=1).astype(np.float32)
    # Standardize data
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    labels = [f"X{i}" for i in range(10)]
    return data, labels

def test_alm_convergence():
    print("=== Testing ALM Convergence ===")
    
    # 1. Prepare data
    data, labels = generate_synthetic_data()
    n_vars = data.shape[1]
    
    # 2. Initialize model in ALM mode
    # x_dim is the correct argument name
    model = DeepANM(x_dim=n_vars)
    
    # Ensure alm is enabled (it should be by default if no order/graph is passed)
    if model.core is not None:
        model.core.use_alm = True
    
    trainer = DeepANMTrainer(model, lr=0.01)
    
    # 3. Training
    print("Starting training loop...")
    epochs = 400
    history = trainer.train(data, epochs=epochs, batch_size=64, verbose=True)
    
    # 4. Visualization
    os.makedirs("results", exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Loss (Left Axis)
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Total Loss (ALM)', color=color)
    ax1.plot(history['loss'], color=color, label='Total Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot h(W) - Acyclicity Constraint (Right Axis)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Acyclicity Constraint h(W)', color=color)
    # Ensure h_val exists in history (we added it to trainer.py)
    if 'h_val' in history:
        ax2.plot(history['h_val'], color=color, label='h(W) Penalty', linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Title and Legend
    plt.title('Convergence Analysis of DeepANM (Augmented Lagrangian Method)')
    fig.tight_layout()
    
    save_path = "results/convergence_test.png"
    plt.savefig(save_path)
    print(f"\n[Success] Convergence plot saved to: {save_path}")
    
    # Check convergence status
    if 'h_val' in history and len(history['h_val']) > 0:
        final_h = history['h_val'][-1]
        if final_h < 1e-3:
            print(f"RESULT: SUCCESS - Model converged to a DAG (h(W) = {final_h:.8f})")
        else:
            print(f"RESULT: WARNING - Model h(W) is still high ({final_h:.8f}).")
    else:
        print("RESULT: ERROR - h_val not found in history.")

if __name__ == "__main__":
    test_alm_convergence()
