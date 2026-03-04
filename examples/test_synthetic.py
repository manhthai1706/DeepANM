"""
Test DeepANM on 5-node Synthetic Nonlinear Data.
Ground Truth DAG:
  X0 -> X1  (cubic)
  X0 -> X2  (sin)
  X1 -> X3  (exp interaction with X2)
  X2 -> X3
  X3 -> X4  (quadratic)
"""
import os, sys
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.deepanm import DeepANM

def generate_5node_nonlinear(n_samples=2000, seed=42):
    np.random.seed(seed)
    
    # X0: Root (Uniform)
    X0 = np.random.uniform(-2, 2, n_samples)
    
    # X1 = f(X0) + noise  (Cubic)
    X1 = 0.8 * X0**3 + np.random.normal(0, 0.5, n_samples)
    
    # X2 = f(X0) + noise  (Sine)
    X2 = 2.0 * np.sin(1.5 * X0) + np.random.normal(0, 0.3, n_samples)
    
    # X3 = f(X1, X2) + noise  (Exp interaction)
    X3 = np.exp(-np.abs(X1) * 0.3) * X2 + 0.5 * X1 + np.random.normal(0, 0.4, n_samples)
    
    # X4 = f(X3) + noise  (Quadratic)
    X4 = 0.3 * X3**2 - X3 + np.random.normal(0, 0.5, n_samples)
    
    return np.column_stack([X0, X1, X2, X3, X4])

def main():
    print("="*60)
    print(" TEST: 5-Node Nonlinear Synthetic DAG")
    print(" Ground Truth: X0->X1, X0->X2, X1->X3, X2->X3, X3->X4")
    print("="*60)
    
    data = generate_5node_nonlinear(2000)
    labels = ['X0', 'X1', 'X2', 'X3', 'X4']
    
    # Ground Truth
    gt_edges = [(0,1), (0,2), (1,3), (2,3), (3,4)]
    W_true = np.zeros((5, 5))
    for i, j in gt_edges:
        W_true[i, j] = 1
    
    print(f"\nData shape: {data.shape}")
    print(f"Ground Truth edges: {len(gt_edges)}")
    
    # Run DeepANM
    print("\nRunning DeepANM (discovery_mode='fast', hidden=32)...")
    model = DeepANM(n_clusters=1, hidden_dim=32, lda=0.0)
    start = time.time()
    
    prob_matrix, avg_W = model.fit_bootstrap(
        data, n_bootstraps=1, apply_quantile=True,
        discovery_mode='fast', layer_constraint=None, verbose=True
    )
    
    W = (prob_matrix > 0).astype(int)
    elapsed = time.time() - start
    
    print(f"\nTime taken: {elapsed:.2f} seconds")
    print(f"Total edges found: {int(W.sum())}")
    
    # Evaluate
    print("\nDiscovered Edges:")
    tp = 0
    fp = 0
    found_edges = []
    for i in range(5):
        for j in range(5):
            if W[i, j] == 1:
                found_edges.append((i, j))
                is_tp = (i, j) in gt_edges
                if is_tp:
                    print(f"  [TRUE]  {labels[i]} -> {labels[j]}")
                    tp += 1
                else:
                    print(f"  [FALSE] {labels[i]} -> {labels[j]}")
                    fp += 1
    
    fn = len(gt_edges) - tp
    
    # SHD
    reversals = 0
    extra_edges = 0
    missing_edges = 0
    
    for u, v in gt_edges:
        if W[u, v] == 0:
            if W[v, u] == 1:
                reversals += 1
            else:
                missing_edges += 1
    
    for i, j in found_edges:
        if (i, j) not in gt_edges and (j, i) not in gt_edges:
            extra_edges += 1
    
    shd = reversals + extra_edges + missing_edges
    
    # Additional metrics
    n_vars = 5
    total_possible = n_vars * (n_vars - 1)
    tn = total_possible - tp - fp - fn
    
    accuracy = (tp + tn) / total_possible
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    nhd = shd / total_possible
    
    print(f"\nEvaluation Metrics:")
    print(f"  TP: {tp}/{len(gt_edges)}  |  FP: {fp}  |  FN: {fn}  |  TN: {tn}")
    print(f"  --------------------------")
    print(f"  SHD Breakdown:")
    print(f"    - Missing: {missing_edges}  Extra: {extra_edges}  Reversed: {reversals}")
    print(f"  => SHD: {shd}")
    print(f"  --------------------------")
    print(f"  Accuracy:   {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  Precision:  {precision:.4f}  ({precision*100:.1f}%)")
    print(f"  Recall:     {recall:.4f}  ({recall*100:.1f}%)")
    print(f"  F1 Score:   {f1:.4f}  ({f1*100:.1f}%)")
    print(f"  NHD:        {nhd:.4f}  ({nhd*100:.1f}%)")
    
    print(f"\nPredicted adjacency matrix:")
    print(W)
    print(f"\nGround Truth adjacency matrix:")
    print(W_true.astype(int))

if __name__ == '__main__':
    main()
