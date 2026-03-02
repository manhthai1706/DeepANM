<div align="center">

# DeepANM

**Deep Additive Noise Model for Nonlinear Causal Discovery**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/tests-7%20passed-brightgreen?style=flat-square)](#testing)

</div>

---

**DeepANM** is a Python library for learning causal graphs from observational data using deep learning. It combines topological ordering, neural structural equation modeling, and Adaptive LASSO edge selection into a unified, statistically principled pipeline.

## How It Works

DeepANM operates in three phases:

```
Phase 1: Topological Ordering (HSIC Greedy — Sink-First)
    Raw data X ──► Discover causal order [X₀ → X₂ → X₁ → X₃]

Phase 2: Neural Training (Augmented Lagrangian)
    X + causal order ──► Learn DAG weight matrix W, noise distributions

Phase 3: Edge Selection (Adaptive LASSO + Neural ATE)
    W + X ──► Adaptive LASSO per variable + ATE double-gate ──► Binary DAG
```

**Key design decisions:**
- **No hard threshold** — edge selection via cross-validated Adaptive LASSO, not a manual cutoff
- **TopoSort once** — causal order computed once before bootstrap, reused across all rounds
- **Self-supervised NLL** — `HeterogeneousNoiseModel` (GMM) is always active via noise proxy `g(X) - f(X)`

---

## Architecture

```
DeepANM/
├── deepanm/
│   ├── core/
│   │   ├── gppom_hsic.py      # Core engine: Gumbel-gate DAG, FastHSIC loss, ALM penalty
│   │   ├── mlp.py             # Backbone: Encoder (Gumbel-Softmax) + SEM + GMM noise + PNL Decoder
│   │   └── toposort.py        # Phase 1: Sink-First HSIC greedy order (RFF-approximated, O(n·D))
│   ├── models/
│   │   └── deepanm.py         # Public API: fit, fit_bootstrap, get_dag_matrix, estimate_ate
│   └── utils/
│       ├── trainer.py         # Augmented Lagrangian training loop
│       ├── adaptive_lasso.py  # Phase 3: Adaptive LASSO edge selection
│       └── visualize.py       # plot_dag() via NetworkX
├── examples/
│   └── synthetic_demo.py      # Nonlinear synthetic benchmark
└── tests/
    └── test_core.py           # 7 unit tests (pytest)
```

---

## Installation

```bash
git clone https://github.com/manhthai1706/DeepANM.git
cd DeepANM
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.8 · PyTorch ≥ 2.0 · scikit-learn ≥ 1.0 · numpy · scipy · matplotlib

---

## Quick Start

### Single fit
```python
import numpy as np
from deepanm import DeepANM

# Generate data (n_samples, n_vars)
X = np.random.randn(500, 5)

model = DeepANM(n_clusters=2, hidden_dim=64, lda=1.0)
model.fit(X, epochs=300, lr=2e-3)

# Get adjacency matrix (no threshold needed — Adaptive LASSO handles it)
ATE, W_binary = model.get_dag_matrix(X=X)
print(W_binary)  # binary (n_vars, n_vars), W[i,j]=1 means i → j
```

### Bootstrap Stability Selection (recommended)
```python
from deepanm import DeepANM, plot_dag

X = np.random.randn(500, 5)
labels = ["A", "B", "C", "D", "E"]

model = DeepANM(n_clusters=2, hidden_dim=64, lda=1.0)

# TopoSort runs ONCE on full X, then is reused for all bootstrap rounds
prob_matrix, avg_ATE = model.fit_bootstrap(
    X,
    n_bootstraps=10,   # stability aggregation rounds
    epochs=200,
    lr=5e-3
)

# prob_matrix[i,j] = fraction of bootstrap rounds that confirmed edge i → j
W_stable = (prob_matrix >= 0.6).astype(int)   # 60% stability threshold

plot_dag(
    W_matrix=W_stable * avg_ATE,
    labels=labels,
    title="DeepANM Causal Graph",
    save_path="result.png"
)
```

### Prior Knowledge: Lock exogenous variables
```python
model = DeepANM(n_clusters=2, hidden_dim=64)
model._build_core(X.shape[1], X=X)

# Forbid any variable from having CRIM as a parent
model.set_exogenous([0])   # variable index 0 is exogenous

model.fit_bootstrap(X, n_bootstraps=5, epochs=200)
```

### Pairwise causal effect estimation
```python
# After fit_bootstrap, query ATE between any pair
ate = model.estimate_ate(X, from_idx=0, to_idx=3)
print(f"ATE of X0 → X3: {ate:.4f}")
```

---

## Technical Details

### Phase 1 — Topological Ordering (`toposort.py`)

Implements **Sink-First HSIC Greedy** ordering (inspired by RESIT, Peters et al. 2014):

1. **QuantileTransform** to Gaussian to handle heavily skewed/heavy-tailed variables
2. At each step, for each candidate sink `k`, regress all other `Xᵢ` on `Xk` using **HistGradientBoosting** (for nonlinear relations) or linear regression
3. Compute `sum HSIC(residᵢ, Xk)` for all `i ≠ k`
4. Variable with minimum score is the sink (leaf) — peel it off
5. Repeat until one variable remains (root)

**RFF approximation** (O(n·D) vs exact O(n²) Gram matrix):  
```
phi(X) = sqrt(2/D) * cos(X @ W_rff + b)   # D=128 random features
HSIC ≈ ||phi_x.T @ phi_y||_F² / n²
```

### Phase 2 — Neural Training (`gppom_hsic.py`)

Jointly optimizes:
```
L = MSE(f(X), X)                    # Structural equation regression
  + λ · HSIC(residuals, X)          # Independence constraint (FastHSIC, O(n·D))
  + λ · HSIC(mechanism Z, X)        # Mechanism clustering constraint  
  + 0.1 · NLL_GMM(noise)            # GMM heterogeneous noise model
  + 0.1 · L1(W) + 0.02 · L2(W)      # Sparsity regularization
  + 0.1 · KL(q_z || uniform)        # VAE mechanism prior

+ ALM: α·h(W) + 0.5·ρ·h(W)²        # DAGMA acyclicity penalty (trainer)
```

**Gumbel-Softmax STE** on the edge gate `W_logits` enforces hard binary decisions.  
**TopoMask** (strict triangular from Phase 1) forbids reverse-direction edges.

### Phase 3 — Adaptive LASSO Edge Selection (`adaptive_lasso.py`)

For each variable `j` in causal order:
1. OLS fit `Xⱼ ~ X_parents` → get `|β_OLS|`
2. Adaptive weights `wᵢ = 1 / (|β_OLS[i]| + ε)`
3. LASSO on re-weighted design matrix with cross-validated `α`
4. Gate with **Direct Causal Effect (ATE) > 0.01**. (Computed via do-calculus on the DAG-masked neural inputs to avoid leaking indirect effects).

This replaces hard thresholding with statistically principled, scale-adaptive sparsity.

---

## Testing

```bash
pytest tests/ -v
```

```
tests/test_core.py::test_mlp_shapes              PASSED
tests/test_core.py::test_heterogeneous_noise_model PASSED
tests/test_core.py::test_fast_hsic               PASSED
tests/test_core.py::test_dag_penalty             PASSED
tests/test_core.py::test_gppomc_core_forward     PASSED
tests/test_core.py::test_global_ate_matrix       PASSED
tests/test_core.py::test_deepanm_integration     PASSED

7 passed in ~4s
```

---

## References & Acknowledgements

DeepANM builds on ideas from the following works:

| Reference | Contribution to DeepANM |
|---|---|
| amber0309 — [ANM-MM](https://github.com/amber0309/ANM-MM) | Original ANM framework & VAE clustering |
| Zheng et al. (2018) — NOTEARS | Continuous DAG optimization, L1/L2 sparsity |
| Bello et al. (2022) — DAGMA | Log-determinant acyclicity penalty |
| Brouillard et al. (2020) — DECI/Causica | Heterogeneous GMM noise model |
| Peters et al. (2014) — RESIT | Sink-first HSIC topological ordering |
| Shimizu et al. (2011) — LiNGAM | Adaptive LASSO edge selection |
| Rahimi & Recht (2007) — RFF | Random Fourier Features for O(n·D) HSIC |
| Zhang & Hyvarinen (2009) — PNL | Post-Nonlinear causal model |

---

## License

MIT License — see [LICENSE](LICENSE). Please acknowledge this repository and the upstream works listed above when building upon this project.
