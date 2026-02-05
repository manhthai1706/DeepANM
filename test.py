# -*- coding: utf-8 -*-
"""
Ultimate ANM-MM Benchmark Suite
Automatically tests Causal Direction and Clustering on synthetic data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from gen_syn import gen_D
from ANMMM import ANMMM_cd, ANMMM_clu
import time

def benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting Benchmark on {device}")
    
    # 1. Test Causal Direction: X -> Y
    print("\n[PART 1: Causal Direction Inference]")
    # Direction: X -> Y (X=rand, f1=exp, noise=normal, f2=sin, n=500)
    # config: [dist_x, f1, noise, f2, n_pts]
    data, labels = gen_D([[1, 1, 2, 3, 500]], device=device)
    
    start_time = time.time()
    result = ANMMM_cd(data, lda=5.0) # Increase lda for stronger independence enforcement
    elapsed = time.time() - start_time
    
    # 2. Test Mechanism Clustering (Mixture of 2 mechanisms)
    print("\n[PART 2: Mechanism Clustering]")
    # Mech 1: Y = X^2 + N
    # Mech 2: Y = sin(X) + N
    config = [
        [2, 0, 2, 2, 250], # Normal X, identity f1, normal noise, t^2
        [2, 0, 2, 3, 250]  # Normal X, identity f1, normal noise, sin
    ]
    data_clu, true_labels = gen_D(config, device=device)
    
    start_time = time.time()
    predicted_labels = ANMMM_clu(data_clu, true_labels, ilda=1.0)
    elapsed = time.time() - start_time
    
    print(f"\nClustering Time: {elapsed:.2f}s")
    print("Benchmark Completed.")

if __name__ == "__main__":
    benchmark()