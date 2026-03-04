import pandas as pd
import numpy as np
import time
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.deepanm import DeepANM

def main():
    print("Download Sachs Data...")
    url = 'https://raw.githubusercontent.com/FenTechSolutions/CausalDiscoveryToolbox/master/cdt/data/resources/cyto_full_data.csv'
    df = pd.read_csv(url)
    labels = df.columns.tolist()
    label_to_idx = {l.lower(): i for i, l in enumerate(labels)}

    # Define Complex 4-Layer Biological Constraints
    # Layer 0: Roots
    # Layer 1: Upstream signaling
    # Layer 2: Middle relay
    # Layer 3: Downstream effects
    layer_constraint = {
        label_to_idx['pka']: 0,
        label_to_idx['pkc']: 0,
        label_to_idx['praf']: 1,
        label_to_idx['plcg']: 1,
        label_to_idx['pip3']: 1,
        label_to_idx['pmek']: 2,
        label_to_idx['pip2']: 2,
        label_to_idx['p44/42']: 3,
        label_to_idx['pakts473']: 3,
        label_to_idx['p38']: 3,
        label_to_idx['pjnk']: 3
    }
    
    print(f"Applying Deep 4-Layer Constraints: {len(layer_constraint)} variables mapped.")

    model = DeepANM(n_clusters=1, hidden_dim=32, lda=0.0)
    start = time.time()
    
    prob_matrix, avg_W = model.fit_bootstrap(df, n_bootstraps=1, apply_quantile=True, 
                                             discovery_mode='fast', 
                                             layer_constraint=layer_constraint, 
                                             verbose=True)
    
    W = (prob_matrix > 0).astype(int)
    end = time.time()
    
    # Ground Truth
    gt_edges_str = [
        ('praf', 'pmek'), ('plcg', 'PIP2'), ('PIP3', 'PIP2'), ('PIP3', 'plcg'), 
        ('pmek', 'p44/42'), ('p44/42', 'pakts473'), 
        ('PKA', 'praf'), ('PKA', 'pmek'), ('PKA', 'p44/42'), ('PKA', 'pakts473'), ('PKA', 'P38'), ('PKA', 'pjnk'),
        ('PKC', 'praf'), ('PKC', 'pmek'), ('PKC', 'P38'), ('PKC', 'pjnk')
    ]
    gt_edges = []
    for u, v in gt_edges_str:
        if u.lower() in label_to_idx and v.lower() in label_to_idx:
            gt_edges.append((label_to_idx[u.lower()], label_to_idx[v.lower()]))

    print(f"\nTime taken: {end - start:.2f} seconds")
    
    tp = 0
    fp = 0
    found_edges = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if W[i, j] == 1:
                found_edges.append((i, j))
                if (i, j) in gt_edges:
                    tp += 1
                else:
                    fp += 1
                    
    reversals = 0
    extra_edges = 0
    missing_edges = 0
    for u, v in gt_edges:
        if W[u, v] == 0:
            if W[v, u] == 1: reversals += 1
            else: missing_edges += 1
    for i, j in found_edges:
        if (i, j) not in gt_edges and (j, i) not in gt_edges:
            extra_edges += 1
            
    shd = reversals + extra_edges + missing_edges
    
    print(f"\nEvaluation Metrics (WITH CONSTRAINTS):")
    print(f"  TP: {tp} / {len(gt_edges)}")
    print(f"  FP: {fp}")
    print(f"  SHD: {shd} (Missing: {missing_edges}, Extra: {extra_edges}, Rev: {reversals})")

if __name__ == '__main__':
    main()
