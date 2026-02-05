# -*- coding: utf-8 -*-
"""
Main DL Training Pipeline - standard Deep Learning workflow.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from gen_syn import gen_D
from GPPOM_HSIC import CausalFlow
from sklearn import metrics

def main():
    # 1. Configuration & Data Generation
    config = [
        [2, 0, 2, 2, 500], # Mechanism 1: Y = X^2
        [2, 0, 2, 3, 500]  # Mechanism 2: Y = sin(X)
    ]
    data, labels = gen_D(config)
    X, Y = data[:, 0:1], data[:, 1:2]
    
    # 2. STANDARD PREPROCESSING (The DL way)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_norm = scaler_x.fit_transform(X)
    Y_norm = scaler_y.fit_transform(Y)
    
    # 3. Model Initialization
    model = CausalFlow(x_dim=1, y_dim=1, n_clusters=2, lda=5.0)
    
    # 4. TRAINING LOOP
    model.fit(X_norm, Y_norm, epochs=150, batch_size=64)
    
    # 5. EVALUATION
    pred_labels = model.predict_clusters(X_norm, Y_norm)
    ari = metrics.adjusted_rand_score(labels, pred_labels)
    print(f"\nFinal ARI Score: {ari:.4f}")
    
    # 6. SAVING ARTIFACTS
    model.save("causal_flow_final.safetensors")
    
    # 7. PLOTTING (Visual feedback)
    plt.figure(figsize=(12, 5))
    
    # Plot Loss History
    plt.subplot(1, 2, 1)
    plt.plot(model.history["loss"], label='Total Loss')
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Plot Clustering Result
    plt.subplot(1, 2, 2)
    colors = ['#F13C20', '#4056A1']
    for i in range(2):
        mask = pred_labels == i
        plt.scatter(X[mask], Y[mask], c=colors[i], label=f'Mech {i}', alpha=0.5)
    plt.title(f"Clustering Result (ARI: {ari:.4f})")
    plt.传说 = plt.legend()
    
    plt.tight_layout()
    plt.savefig("results.png")
    print("Visual results saved to results.png")
    plt.show()

if __name__ == "__main__":
    main()
