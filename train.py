# -*- coding: utf-8 -*-
"""
Training Script for CausalFlow
Handles model training, validation, and logging
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from CausalFlow import CausalFlow


class CausalFlowTrainer:
    """
    Trainer class for CausalFlow model
    Handles training loop, optimization, and history tracking
    """
    def __init__(self, model, lr=2e-3, weight_decay=1e-2):
        """
        Initialize trainer
        
        Args:
            model (CausalFlow): The CausalFlow model to train
            lr (float): Learning rate
            weight_decay (float): Weight decay for regularization
        """
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.history = {
            "loss": [],
            "likelihood": [],
            "hsic": []
        }
        
    def train(self, X, epochs=200, batch_size=64, verbose=True):
        """
        Train the CausalFlow model in Multivariate mode
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        # Create data loader
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if verbose:
            print(f">> Training SOTA CausalFlow on {self.model.device}")
            print(f"   Variables: {X.shape[1]}, Samples: {X.shape[0]}")
            print(f"   Features: DAG (NOTEARS) + SplineFlow + VAE Latents")
            print("-" * 60)
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_reg = 0.0
            epoch_hsic = 0.0
            
            temperature = max(0.5, 1.0 - epoch / epochs)
            
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.model.device)
                
                self.optimizer.zero_grad()
                total_loss, reg_loss, hsic_loss = self.model(
                    batch_x, temperature=temperature
                )
                
                total_loss.backward()
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_reg += reg_loss.item()
                epoch_hsic += hsic_loss.item()
            
            n_batches = len(loader)
            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {epoch_loss/n_batches:.4f} | "
                      f"Reg: {epoch_reg/n_batches:.4f} | HSIC: {epoch_hsic/n_batches:.6f}")
        
        return self.history


def train_causalflow(X, Y, x_dim, y_dim, n_clusters=2, hidden_dim=64, 
                     lda=1.0, epochs=200, batch_size=64, lr=2e-3, 
                     device=None, verbose=True):
    """
    Convenience function to train a CausalFlow model
    """
    model = CausalFlow(
        x_dim=x_dim,
        y_dim=y_dim,
        n_clusters=n_clusters,
        hidden_dim=hidden_dim,
        lda=lda,
        device=device
    )
    
    trainer = CausalFlowTrainer(model, lr=lr)
    history = trainer.train(X, Y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    return model, trainer, history


if __name__ == "__main__":
    print("CausalFlow Training Script")
    print("=" * 60)
    
    np.random.seed(42)
    N = 1000
    x_dim, y_dim = 1, 1
    n_clusters = 2
    
    X = np.random.randn(N, x_dim)
    Y = np.random.randn(N, y_dim)
    
    model, trainer, history = train_causalflow(X, Y, x_dim, y_dim, epochs=50)
