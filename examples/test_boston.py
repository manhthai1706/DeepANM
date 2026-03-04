"""
Test DeepANM on Boston Housing dataset.
Goal: Discover causal relationships among housing features.
No ground truth DAG available — this is an exploratory causal discovery task.
"""
import numpy as np
import pandas as pd
import time
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.deepanm import DeepANM

def main():
    print("="*60)
    print(" CAUSAL DISCOVERY: Boston Housing Dataset")
    print("="*60)
    
    # Load Boston Housing from public URL (sklearn deprecated it)
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    labels = df.columns.tolist()
    print(f"Data shape: {df.shape}")
    print(f"Variables: {labels}")
    print()
    print("Variable descriptions:")
    descriptions = {
        'crim': 'Per capita crime rate',
        'zn': 'Residential land zoned for large lots (%)',
        'indus': 'Non-retail business acres (%)',
        'chas': 'Charles River dummy (0/1)',
        'nox': 'Nitric oxide concentration (pollution)',
        'rm': 'Average rooms per dwelling',
        'age': 'Owner-occupied units built before 1940 (%)',
        'dis': 'Distance to employment centres',
        'rad': 'Highway accessibility index',
        'tax': 'Property tax rate per $10k',
        'ptratio': 'Pupil-teacher ratio',
        'b': 'Black population index',
        'lstat': 'Lower status population (%)',
        'medv': 'Median home value ($1000s)'
    }
    for col in labels:
        desc = descriptions.get(col.lower(), '?')
        print(f"  {col:>8s} : {desc}")
    
    # Run DeepANM (Fast mode, no prior knowledge)
    print("\nRunning DeepANM (discovery_mode='fast', hidden=32)...")
    model = DeepANM(n_clusters=1, hidden_dim=32, lda=0.0)
    start = time.time()
    
    prob_matrix, avg_W = model.fit_bootstrap(
        df.values, n_bootstraps=1, apply_quantile=True,
        discovery_mode='fast', layer_constraint=None, verbose=True
    )
    
    W = (prob_matrix > 0).astype(int)
    elapsed = time.time() - start
    
    n_edges = int(W.sum())
    print(f"\nTime taken: {elapsed:.2f} seconds")
    print(f"Total edges discovered: {n_edges}")
    
    # Print discovered edges sorted by ATE strength
    print("\nDiscovered Causal Edges (sorted by |ATE| strength):")
    print("-" * 55)
    
    edges = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if W[i, j] == 1:
                ate_val = avg_W[i, j]
                edges.append((labels[i], labels[j], ate_val))
    
    # Sort by absolute ATE descending
    edges.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for src, dst, ate in edges:
        src_desc = descriptions.get(src.lower(), '')
        dst_desc = descriptions.get(dst.lower(), '')
        direction = "+" if ate > 0 else "-"
        print(f"  {src:>8s} -> {dst:<8s}  (ATE={ate:+.4f} {direction})  [{src_desc} -> {dst_desc}]")
    
    # Highlight edges pointing to MEDV (house price)
    print("\n" + "="*55)
    print(" Causal factors of MEDV (Home Price):")
    print("="*55)
    medv_idx = [i for i, l in enumerate(labels) if l.lower() == 'medv']
    if medv_idx:
        j = medv_idx[0]
        parents = []
        for i in range(len(labels)):
            if W[i, j] == 1:
                parents.append((labels[i], avg_W[i, j]))
        if parents:
            parents.sort(key=lambda x: abs(x[1]), reverse=True)
            for name, ate in parents:
                desc = descriptions.get(name.lower(), '')
                effect = "INCREASES" if ate > 0 else "DECREASES"
                print(f"  {name:>8s} {effect} home price  (ATE={ate:+.4f})  [{desc}]")
        else:
            print("  No direct causal parents found for MEDV.")
    
    # Highlight edges FROM NOX (pollution)
    print("\n" + "="*55)
    print(" What does NOX (Pollution) cause?")
    print("="*55)
    nox_idx = [i for i, l in enumerate(labels) if l.lower() == 'nox']
    if nox_idx:
        i = nox_idx[0]
        children = []
        for j in range(len(labels)):
            if W[i, j] == 1:
                children.append((labels[j], avg_W[i, j]))
        if children:
            children.sort(key=lambda x: abs(x[1]), reverse=True)
            for name, ate in children:
                desc = descriptions.get(name.lower(), '')
                print(f"  NOX -> {name:<8s}  (ATE={ate:+.4f})  [{desc}]")
        else:
            print("  No downstream effects found from NOX.")

if __name__ == '__main__':
    main()
