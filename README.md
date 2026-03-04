<div align="center">

# DeepANM: Nonlinear Causal Structure Discovery

**Nonlinear Causal Structure Discovery via Deep Additive Noise Model**

---

DeepANM is a comprehensive framework for discovering causal relationships from purely observational data. By combining non-parametric statistical tests with deep neural networks, it effectively identifies directed acyclic graphs (DAGs) in complex, nonlinear environments with heterogeneous noise.

</div>

## Key Features

- **Three-Phase Synergetic Pipeline:** Decouples causal ordering, structure learning, and edge refinement.
- **Efficient Nonlinear Testing:** Uses HSIC with Random Fourier Features (RFF) for $O(N \cdot D)$ complexity.
- **Heterogeneous Noise Modeling:** Adaptable to non-Gaussian and mechanism-switching noise distributions.
- **Prior Knowledge Integration:** Easily incorporate biological or domain-specific layer constraints.
- **Interpretable Effects:** Direct estimation of Average Treatment Effects (ATE) for all discovered edges.

---

## Architecture Overview

DeepANM operates through a systematic three-phase process:

1.  **Phase 1: Causal Ordering (HSIC TopoSort):** Identifies the topological flow of information using the asymmetry of Additive Noise Models (ANM).
2.  **Phase 2: Neural SCM Fitting:** A deep neural network learns the functional mechanisms $f_j(X_{pa(j)})$ under the topological constraints identified in Phase 1.
3.  **Phase 3: Adaptive Refinement (Double-Gate):** Combines Adaptive LASSO and neural ATE estimation to prune spurious edges and ensure high structural precision.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/manhthai1706/DeepANM.git
cd DeepANM

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 2.0, scikit-learn, numpy, scipy, matplotlib.

---

## Usage

### Quick Discovery
```python
import numpy as np
from src.models.deepanm import DeepANM

# Prepare your data (n_samples, n_variables)
X = np.random.randn(1000, 5)

# Initialize and fit model
model = DeepANM(n_clusters=1, hidden_dim=32)
prob_matrix, _ = model.fit_bootstrap(X, n_bootstraps=5, discovery_mode='fast')

# Extract the discovered DAG
W = (prob_matrix > 0).astype(int)
print(f"Discovered {W.sum()} causal edges.")
```

### Running Benchmarks
We provide several scripts as entry points for evaluating performance:
```bash
python examples/test_sachs.py      # Protein signaling network (Sachs)
python examples/test_boston.py     # Economic causal discovery (Boston Housing)
python examples/ablation_study.py  # Component-wise performance analysis
```

---

## Experimental Results

### Sachs Dataset (Biological Benchmark)
| Method | SHD | F1-Score | Note |
|:---|:---:|:---:|:---|
| PC Algorithm | 23 | ~35% | Constraint-based linear |
| GES | 21 | ~38% | Score-based linear |
| GraN-DAG | 13 | ~48% | Neural SOTA |
| **DeepANM** | **12** | **~59%** | **With Layer Constraints** |

*SHD = Structural Hamming Distance (Lower is better).*

### Component Contribution (Ablation)
Incremental performance gain on the Sachs dataset:
- **Baseline (OLS):** SHD = 42
- **+ Non-linear (Random Forest):** SHD = 23 ($-45\%$)
- **+ CI Pruning:** SHD = 18 ($-22\%$)
- **+ Full Pipeline (Double-Gate):** **SHD = 17** (Best blind discovery)

---

## Project Structure

```text
DeepANM/
├── src/
│   ├── core/         # Core logic (HSIC, MLP, TopoSort)
│   ├── models/       # Model APIs (DeepANM, FastANM)
│   └── utils/        # Training loops and refinement tools
├── examples/         # Evaluation and demo scripts
├── tests/            # Unit tests using pytest
└── docx/             # Detailed technical documentation
```

---

## Testing

Ensuring reliability through comprehensive unit tests:
```bash
pytest tests/ -v
```

---

## References

[1] Hoyer, P. O., Janzing, D., Mooij, J. M., Peters, J., & Schölkopf, B. (2009). Nonlinear causal discovery with additive noise models. *Advances in neural information processing systems*, 21.

[2] Zhang, K., & Hyvärinen, A. (2009). On the identifiability of the post-nonlinear causal model. In *Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence* (pp. 647-655).

[3] Gretton, A., Bousquet, O., Smola, A., & Schölkopf, B. (2005). Measuring statistical dependence with Hilbert-Schmidt norms. In *Algorithmic Learning Theory* (pp. 63-77). Springer Berlin Heidelberg.

[4] Rahimi, A., & Recht, B. (2007). Random features for large-scale kernel machines. *Advances in neural information processing systems*, 20.

[5] Peters, J., Mooij, J. M., Janzing, D., & Schölkopf, B. (2014). Causal discovery with continuous additive noise models. *The Journal of Machine Learning Research*, 15(1), 2009-2053.

[6] Zheng, X., Aragam, B., Ravikumar, P. K., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous optimization for structure learning. *Advances in neural information processing systems*, 31.

[7] Bello, K., Aragam, B., & Ravikumar, P. (2022). DAGMA: Learning DAGs via M-matrices and a log-determinant acyclicity characterization. *Advances in Neural Information Processing Systems*, 35, 8226-8239.

[8] Brouillard, P., Lachapelle, S., Lacoste, A., Lacoste-Julien, S., & Drouin, A. (2020). Differentiable causal discovery from interventional data. *Advances in Neural Information Processing Systems*, 33, 21865-21877.

[9] Zou, H. (2006). The adaptive lasso and its oracle properties. *Journal of the American statistical association*, 101(476), 1418-1429.

[10] Breiman, L. (2001). Random forests. *Machine learning*, 45(1), 5-32.

[11] Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan, G. P. (2005). Causal protein-signaling networks derived from multiparameter single-cell data. *Science*, 308(5721), 523-529.

[12] Harrison Jr, D., & Rubinfeld, D. L. (1978). Hedonic housing prices and the demand for clean air. *Journal of environmental economics and management*, 5(1), 81-102.

[13] amber0309. (2023). *GPPOM: Gaussian Process partially observable model for causal discovery*. GitHub repository, Retrieved from https://github.com/amber0309/GPPOM

[14] Jang, E., Gu, S., & Poole, B. (2016). Categorical reparameterization with gumbel-softmax. *arXiv preprint arXiv:1611.01144*.

[15] Pearl, J. (2009). *Causality*. Cambridge university press.

[16] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). Lightgbm: A highly efficient gradient boosting decision tree. *Advances in neural information processing systems*, 30.

[17] Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. In *2008 eighth ieee international conference on data mining* (pp. 413-422). IEEE.

[18] Bertsekas, D. P. (1982). *Constrained optimization and Lagrange multiplier methods*. Academic press.

[19] Bolstad, B. M., Irizarry, R. A., Astrand, M., & Speed, T. P. (2003). A comparison of normalization methods for high density oligonucleotide array data based on variance and bias. *Bioinformatics*, 19(2), 185-193.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---
<div align="center">
Developed by Manh Thai | 2025
</div>
