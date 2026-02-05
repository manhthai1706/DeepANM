# -*- coding: utf-8 -*-
"""
ANM-MM (ANM Mixture Model) - Modern PyTorch Integration
Updated for compatibility with Deep Learning MLP and GPPOM
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

from GPPOM_HSIC import CausalFlow
from HSIC import hsic_gam

def draw_clu(data, label, name):
    ## draw the whole data set
    ## pts in diff color belong to diff clusters
    colours = ['#F13C20','#E27D60','#BC986A','#4056A1','#D79922','#379683','#379683']
    markers = ['o', 'p', 's', 'v', '^', '<', '>']
    label_list = np.unique(label)
    plt.figure(figsize=(8, 6))
    
    is_1d = data.shape[1] == 1
    
    for i, ilabel in enumerate(label_list):
        cu_indices = [idx for idx, x in enumerate(label) if x == ilabel]
        cu_data = data[cu_indices, :]
        
        if is_1d:
            # For 1D data, we plot it on the X-axis and use a fixed Y=0 or jitter
            y_vals = np.zeros_like(cu_data[:, 0])
            plt.scatter(cu_data[:, 0], y_vals, c=colours[i % len(colours)], 
                        marker=markers[i % len(markers)], label=f'Cluster {ilabel}', alpha=0.6)
        else:
            plt.scatter(cu_data[:, 0], cu_data[:, 1], c=colours[i % len(colours)], 
                        marker=markers[i % len(markers)], label=f'Cluster {ilabel}', alpha=0.6)

    if is_1d:
        plt.yticks([]) # Hide Y axis for 1D
        plt.xlabel('Latent Variable Z')
    
    plt.title(name)
    plt.legend()
    plt.grid(True, alpha=0.2)

def ANMMM_cd(data, lda):
    """Causal Direction Inference using Ultimate GPPOM"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    X = data[:,0].reshape(-1 ,1)
    Y = data[:,1].reshape(-1 ,1)

    # 1. Infer X --> Y
    print("\n--- Ultimate Testing X --> Y (RFF + Attention + Gumbel) ---")
    cf1 = CausalFlow(x_dim=1, y_dim=1, n_clusters=2, lda=lda, device=device)
    cf1.fit(X, Y, epochs=150)
    
    with torch.no_grad():
        cf1.model.eval()
        out1 = cf1.model.MLP(cf1.model.XY)
        z_soft1 = out1['z_soft'].cpu().numpy()
        
    # Test independence of learned mechanism indicators and cause
    stat1, thresh1, p1 = hsic_gam(z_soft1, X, 0.05)
    r1 = stat1 / thresh1
    print(f"X->Y: p-value = {p1:.6f}")

    # 2. Infer Y --> X
    print("\n--- Ultimate Testing Y --> X (RFF + Attention + Gumbel) ---")
    cf2 = CausalFlow(x_dim=1, y_dim=1, n_clusters=2, lda=lda, device=device)
    cf2.fit(Y, X, epochs=150)
    
    with torch.no_grad():
        cf2.model.eval()
        out2 = cf2.model.MLP(cf2.model.XY)
        z_soft2 = out2['z_soft'].cpu().numpy()
        
    stat2, thresh2, p2 = hsic_gam(z_soft2, Y, 0.05)
    r2 = stat2 / thresh2
    print(f"Y->X: p-value = {p2:.6f}")

    print(f'\nResults: r1(X->Y) = {r1:.4f}, r2(Y->X) = {r2:.4f}')

    if r1 < r2:
        print('Inferred Cause: X (X --> Y)')
        return 1
    elif r1 > r2:
        print('Inferred Cause: Y (Y --> X)')
        return -1
    else:
        print('Inferred Cause: Unknown')
        return 0

def ANMMM_clu(data, label_true, ilda):
    """Mechanism Clustering using Ultimate GPPOM (End-to-End)"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    X = data[:,0].reshape(-1 ,1)
    Y = data[:,1].reshape(-1 ,1)
    nclu = len(np.unique(label_true))

    # Apply Ultimate GPPOM
    print(f"\n--- End-to-End Clustering (lda={ilda}, clusters={nclu}) ---")
    cf = CausalFlow(x_dim=1, y_dim=1, n_clusters=nclu, lda=ilda, device=device)
    cf.fit(X, Y, epochs=200)

    # Extract learned categorical labels directly
    clu_label = cf.predict_clusters(X, Y)

    ari = metrics.adjusted_rand_score(label_true, clu_label)
    print(f'\nARI of Ultimate ANM-MM: {ari:.4f}')

    # Visualization
    with torch.no_grad():
        model.eval()
        # Use logits or attention features for 2D visualization if needed
        z_viz = model.MLP(model.XY)['logits'].cpu().numpy()

    draw_clu(z_viz, label_true, 'Learned Categorical Logits (Mechanism Space)')
    draw_clu(data, clu_label, 'Final Clustering Results')
    plt.show()

    return clu_label
