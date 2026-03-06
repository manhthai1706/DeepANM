<div align="center">

# DeepANM: End-to-End Nonlinear Causal Discovery

**A Differentiable Framework for Unifying Structure Learning and Causal Inference**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-ee4c2c.svg)](https://pytorch.org/)

---

**DeepANM** is a research-grade framework designed for **end-to-end causal discovery and inference** from purely observational data. By leveraging differentiable acyclicity constraints and deep additive noise models, DeepANM simultaneously learns the causal structure (DAG) and the underlying functional mechanisms of a system.

</div>

## 🧬 Overview

Traditional causal discovery often decouples structure search from functional estimation. **DeepANM** unifies these processes into a cohesive, gradient-based optimization pipeline. It is particularly designed for complex, nonlinear systems where Gaussian noise assumptions do not hold.

### Core Capabilities
*   **Differentiable DAG Learning:** Optimized via an Augmented Lagrangian Method (ALM) with log-determinant acyclicity characterization.
*   **Nonlinear Functional Mechanisms:** Captures deep nonlinear parent-child relationships using high-capacity neural networks.
*   **Heterogeneous Noise Modeling:** Robust against non-Gaussian noise through the Additive Noise Model (ANM) framework.
*   **Integrated Treatment Effects:** Automatically estimates Average Treatment Effects (ATE) for every discovered causal link.

---

## �️ Getting Started

### Installation
```bash
git clone https://github.com/manhthai1706/DeepANM.git
cd DeepANM
pip install -r requirements.txt
```

### Simple End-to-End API
DeepANM provides a high-level API for rapid discovery. The `fit` method handles the entire three-phase pipeline internally.

```python
import numpy as np
from src.models.deepanm import DeepANM

# Initialize the End-to-End Model
model = DeepANM()

# Execute Discovery and Inference
# W_weights: Matrix of causal strengths (ATE)
# W_adj: Binary adjacency matrix (DAG)
W_weights, W_adj = model.fit(data)

print(f"Discovered {int(W_adj.sum())} causal edges.")
```

---

## 🏗️ The Three-Phase Synergetic Pipeline

DeepANM utilizes a multi-phase approach to navigate the massive space of potential DAGs efficiently:

1.  **Topological Discovery:** Identifies a valid information flow using non-parametric HSIC (Hilbert-Schmidt Independence Criterion).
2.  **Differentiable Structural Learning:** Fits an Neural SCM under acyclicity constraints provided by the **DAGMA** logic.
3.  **Stability-Driven Refinement:** Prunes the graph using a "Double-Gate" selection process (Nonlinear Adaptive LASSO + ATE Strength Filtering).

---

## � Benchmarks & Gallery

### 1. ALM Convergence Analysis
Monitoring the optimization of the total loss alongside the strict $h(W)$ acyclicity penalty ensures a valid DAG output.
![Convergence](results/convergence_test.png)

### 2. SCM mechanism Discovery (Sachs)
Validated on the gold-standard biological signaling network. DeepANM recovers hidden pathways in high-noise environments.
![Sachs](results/sachs_comparison.png)

### 3. Systematic Ablation Study
Incremental performance gains achieved by moving from OLS baselines to the full Three-Phase DeepANM pipeline.
![Ablation](results/ablation_study_comparison.png)

### 4. Real-World Discovery (Boston Housing)
Exploring the causal drivers behind housing prices and environmental factors.
![Boston](results/boston_dag.png)

---

## 📁 Project Structure

*   `src/core/`: Mathematical engines for HSIC and TopoSort.
*   `src/models/`: Main `DeepANM` class and differentiable components.
*   `src/utils/`: ALM trainer, refinement tools (Lasso), and visualization.
*   `examples/`: Evaluation scripts for Sachs, Boston Housing, and synthetic data.
*   `tests/`: Unit tests and convergence validation.

---

## 🎓 References & Acknowledgements

DeepANM is built upon state-of-the-art research in causal discovery:
- **DAGMA:** M-matrices and log-determinant acyclicity (Bello et al., 2022).
- **Additive Noise Models:** Identifiability in nonlinear cases (Hoyer et al., 2009).
- **Non-Parametric Dependencies:** HSIC for large scale (Gretton et al., 2005).

---

## 📄 License
MIT License.

<div align="center">
  <b>Developed by Manh Thai | 2026</b>
</div>
