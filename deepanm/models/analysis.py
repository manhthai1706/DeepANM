"""
ANM-MM (ANM Mixture Model) - Modern PyTorch Integration
Updated for compatibility with Deep Learning MLP and GPPOM
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

from deepanm.models.deepanm import DeepANM
from deepanm.core.hsic import hsic_gam

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
    """
    Advanced Causal Direction Inference - Fixed Structure Mode
    Forces X->Y and Y->X structures to compare pure independence.
    """
    X = data[:,0].reshape(-1 ,1)
    Y = data[:,1].reshape(-1 ,1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test X -> Y
    print("\n[Testing Hypothesis: X --> Y]")
    cf1 = DeepANM(x_dim=1, y_dim=1, n_clusters=2, lda=lda, device=device)
    # FORCE Structure: X causes Y (W[0, 1] = 1, others = 0)
    with torch.no_grad():
        mask1 = torch.zeros(2, 2).to(device)
        mask1[0, 1] = 1.0
        cf1.core.W_dag.data = mask1
        cf1.core.W_dag.requires_grad = False # Lock structure
        
    cf1.fit(X, Y, epochs=200, verbose=False)
    combined1 = np.hstack([X, Y])
    stab1, _ = cf1.check_stability(combined1)
    res1 = cf1.get_residuals(combined1)
    stat1, _, p1 = hsic_gam(res1[:, 1:2], X) 

    # Test Y -> X
    print("[Testing Hypothesis: Y --> X]")
    cf2 = DeepANM(x_dim=1, y_dim=1, n_clusters=2, lda=lda, device=device)
    # FORCE Structure: Y causes X (W[1, 0] = 1, others = 0)
    with torch.no_grad():
        mask2 = torch.zeros(2, 2).to(device)
        mask2[1, 0] = 1.0
        cf2.core.W_dag.data = mask2
        cf2.core.W_dag.requires_grad = False # Lock structure
        
    cf2.fit(Y, X, epochs=200, verbose=False)
    combined2 = np.hstack([Y, X])
    stab2, _ = cf2.check_stability(combined2)
    res2 = cf2.get_residuals(combined2)
    stat2, _, p2 = hsic_gam(res2[:, 1:2], Y)

    # Final Decision logic
    print(f"X->Y HSIC: {stat1:.6f} | Y->X HSIC: {stat2:.6f}")
    
    score1 = stat1 * (1.0 + stab1 * 0.5)
    score2 = stat2 * (1.0 + stab2 * 0.5)
    
    if score1 < score2:
        return 1, cf1
    else:
        return -1, cf2

def ANMMM_clu(data, label_true, ilda):
    """Mechanism Clustering using DeepANM (End-to-End)"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    X = data[:,0].reshape(-1 ,1)
    Y = data[:,1].reshape(-1 ,1)
    nclu = len(np.unique(label_true))

    # Apply DeepANM
    print(f"\n--- End-to-End Clustering (lda={ilda}, clusters={nclu}) ---")
    cf = DeepANM(x_dim=1, y_dim=1, n_clusters=nclu, lda=ilda, device=device)
    cf.fit(X, Y, epochs=200)

    # Extract learned categorical labels directly
    combined = np.hstack([X, Y])
    clu_label = cf.predict_clusters(combined)

    ari = metrics.adjusted_rand_score(label_true, clu_label)
    print(f'\nARI of DeepANM: {ari:.4f}')

    # Visualization
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(device)
        Y_tensor = torch.from_numpy(Y).float().to(device)
        xy = torch.cat([X_tensor, Y_tensor], dim=1)
        z_viz = cf.core.MLP(xy)['logits'].cpu().numpy()

    draw_clu(z_viz, label_true, 'Learned Categorical Logits (Mechanism Space)')
    draw_clu(data, clu_label, 'Final Clustering Results')
    plt.show()

    return clu_label
