#!/usr/bin/env python
# coding: utf-8

# # DeepANM Training on Sachs Dataset
# 
# This notebook demonstrates training DeepANM for causal discovery on the Sachs dataset from CMU-PHIL example causal datasets.

# In[2]:


# get_ipython().system('pip install git+https://github.com/manhthai1706/DeepANM.git')


# In[3]:


import numpy as np
import pandas as pd
import urllib.request
import sys
sys.path.insert(0, '/root/DeepANM')

from deepanm import DeepANM, plot_dag
import matplotlib.pyplot as plt


# ## Download and Load Sachs Dataset
# 
# Load the Sachs dataset from the CMU-PHIL repository:

# In[4]:


# Try loading from local file first, if not available download from GitHub
import os

local_path = 'datasets/sachs/sachs.2005.continuous.txt'

if os.path.exists(local_path):
    print(f"Loading local file from {local_path}")
    X = pd.read_csv(local_path, sep='\t')
else:
    # Download from GitHub
    url = 'https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/sachs/data/sachs.2005.continuous.txt'
    print(f"Downloading from {url}")
    X = pd.read_csv(url, sep='\t')

print(f"Data shape: {X.shape}")
print(f"\nFirst few rows:")
print(X.head())
print(f"\nData info:")
print(X.info())
print(f"\nBasic statistics:")
print(X.describe())


# ## Data Preprocessing
# 
# Check and prepare data for training:

# In[5]:


# Check for missing values
print(f"Missing values: {X.isnull().sum().sum()}")

# Convert to numpy array
X_array = X.values.astype(np.float32)

# Standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_array)

print(f"\nScaled data shape: {X_scaled.shape}")
print(f"Scaled data mean: {X_scaled.mean(axis=0)[:5]}")
print(f"Scaled data std: {X_scaled.std(axis=0)[:5]}")

# Get variable names
var_names = X.columns.tolist()
print(f"\nVariable names: {var_names}")


# ## Train DeepANM Model
# 
# Initialize and train the model using bootstrap stability selection:

# In[6]:


# Initialize DeepANM model
model = DeepANM(
    n_clusters=2,      # Number of mixture components for heterogeneous noise
    hidden_dim=64,     # Hidden dimension of the neural network
    lda=1.0            # Lagrange multiplier for acyclicity constraint
)

print("Model initialized")

# Train with bootstrap stability selection
# This runs topological sort once, then trains multiple rounds with different bootstrap samples
prob_matrix, avg_ATE = model.fit_bootstrap(
    X_scaled,
    n_bootstraps=5,    # Number of bootstrap rounds for stability estimation
    epochs=200,        # Epochs per round
    lr=5e-3,           # Learning rate
    verbose=True,
    apply_isolation=True,
    apply_quantile=True
)

print(f"\nTraining completed!")
print(f"Probability matrix shape: {prob_matrix.shape}")
print(f"ATE matrix shape: {avg_ATE.shape}")


# ## Extract and Visualize Causal Graph
# 
# Get the binary DAG matrix and visualize the discovered causal structure:

# In[7]:


# Extract binary DAG using stability threshold
stability_threshold = 0.6
W_binary = (prob_matrix >= stability_threshold).astype(int)

print(f"Binary DAG matrix (threshold={stability_threshold}):")
print(W_binary)
print(f"\nNumber of edges: {W_binary.sum()}")

# Create adjacency matrix weighted by ATE
W_weighted = W_binary * avg_ATE
print(f"\nWeighted adjacency matrix (W * ATE):")
print(W_weighted)


# ## Analysis of Results
# 
# Examine the discovered causal relationships:

# In[8]:


# Print discovered edges with their stability and ATE
print("Discovered causal edges (with stability and ATE):\n")
print(f"{'From':<15} {'To':<15} {'Stability':<15} {'ATE':<15}")
print("-" * 60)

edges = []
for i in range(len(var_names)):
    for j in range(len(var_names)):
        if W_binary[i, j] == 1:
            edges.append((var_names[i], var_names[j], prob_matrix[i, j], avg_ATE[i, j]))
            print(f"{var_names[i]:<15} {var_names[j]:<15} {prob_matrix[i, j]:<15.3f} {avg_ATE[i, j]:<15.4f}")

print(f"\nTotal edges discovered: {len(edges)}")

# Estimate pairwise Average Treatment Effects (ATE)
if len(edges) > 0:
    print("\n\nPairwise ATE for discovered edges:")
    for from_var, to_var, stab, ate in edges[:10]:  # Show first 10
        print(f"  ATE({from_var} → {to_var}) = {ate:.6f}")

