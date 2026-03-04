import pandas as pd
import numpy as np
import time
from src.models.deepanm import DeepANM

def main():
    print("Download Sachs Data (CDT Version)...")
    url = 'https://raw.githubusercontent.com/FenTechSolutions/CausalDiscoveryToolbox/master/cdt/data/resources/cyto_full_data.csv'
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print("Error downloading Sachs data:", e)
        return

    labels = df.columns.tolist()

    print(f"Data shape: {df.shape}")

    # Chạy DeepANM (Discovery Mode: Fast, NO Constraints)
    print("\nRunning DeepANM (discovery_mode='fast') WITHOUT Biological Layer Constraint...")
    model = DeepANM(n_clusters=1, hidden_dim=32, lda=0.0)
    start = time.time()
    
    # layer_constraint=None ensures that FastANM only uses TopoSort and CI Pruning
    prob_matrix, avg_W = model.fit_bootstrap(df, n_bootstraps=1, apply_quantile=True, 
                                             discovery_mode='fast', layer_constraint=None, verbose=True)
    
    W = (prob_matrix > 0).astype(int)
    
    end = time.time()
    
    # Mạch chuẩn CDT format
    gt_edges_str = [
        ('praf', 'pmek'), ('plcg', 'PIP2'), ('PIP3', 'PIP2'), ('PIP3', 'plcg'), 
        ('pmek', 'p44/42'), ('p44/42', 'pakts473'), 
        ('PKA', 'praf'), ('PKA', 'pmek'), ('PKA', 'p44/42'), ('PKA', 'pakts473'), ('PKA', 'P38'), ('PKA', 'pjnk'),
        ('PKC', 'praf'), ('PKC', 'pmek'), ('PKC', 'P38'), ('PKC', 'pjnk')
    ]
    
    # Map labels to indices for ground truth
    label_to_idx = {l.lower(): i for i, l in enumerate(labels)}
    gt_edges = []
    for u, v in gt_edges_str:
        if u.lower() in label_to_idx and v.lower() in label_to_idx:
            gt_edges.append((label_to_idx[u.lower()], label_to_idx[v.lower()]))

    print(f"\nTime taken: {end - start:.2f} seconds")
    print(f"Total edges found: {int(W.sum())}")
    
    print("\nDiscovered Edges Evaluation:")
    tp = 0
    fp = 0
    found_edges = []
    for i in range(len(labels)):
        for j in range(len(labels)):
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
    
    # Calculate Structural Hamming Distance (SHD)
    reversals = 0
    extra_edges = 0
    missing_edges = 0
    
    # Missing edges and Reversals
    for u, v in gt_edges:
        if W[u, v] == 0:
            if W[v, u] == 1:
                reversals += 1
            else:
                missing_edges += 1
                
    # Extra edges
    for i, j in found_edges:
        if (i, j) not in gt_edges and (j, i) not in gt_edges:
            extra_edges += 1
            
    shd = reversals + extra_edges + missing_edges
    
    print(f"\nEvaluation Metrics:")
    print(f"  True Positives (TP): {tp} / {len(gt_edges)}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  --------------------------")
    print(f"  SHD Breakdown:")
    print(f"    - Missing Edges: {missing_edges}")
    print(f"    - Extra Edges:   {extra_edges}")
    print(f"    - Reversed Edges: {reversals}")
    print(f"  => Structural Hamming Distance (SHD): {shd}")
    
    # Additional Metrics
    n_vars = len(labels)
    total_possible_edges = n_vars * (n_vars - 1)  # All directed edges (excluding self-loops)
    tn = total_possible_edges - tp - fp - fn  # True Negatives
    
    accuracy = (tp + tn) / total_possible_edges if total_possible_edges > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    nhd = shd / total_possible_edges if total_possible_edges > 0 else 0
    
    print(f"\n  Additional Metrics:")
    print(f"    - True Negatives (TN): {tn}")
    print(f"    - Accuracy:   {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"    - Precision:  {precision:.4f}  ({precision*100:.1f}%)")
    print(f"    - Recall:     {recall:.4f}  ({recall*100:.1f}%)")
    print(f"    - F1 Score:   {f1:.4f}  ({f1*100:.1f}%)")
    print(f"    - NHD (Normalized Hamming Distance): {nhd:.4f}  ({nhd*100:.1f}%)")

if __name__ == '__main__':
    main()
