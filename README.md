<div align="center">

# DeepANM

### Đồ án Tốt nghiệp — Đại học [Tên Trường]

**Khám phá Cấu trúc Nhân quả Phi tuyến bằng Mô hình Nhiễu Cộng Sâu**  
*Nonlinear Causal Structure Discovery via Deep Additive Noise Model*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square)](https://pytorch.org)

> **Sinh viên thực hiện:** Manh Thai  
> **Năm học:** 2024 – 2025

</div>

---

## Giới thiệu

**DeepANM** là một hệ thống khám phá quan hệ nhân quả từ dữ liệu quan sát thuần túy *(observational data)*, không cần thực nghiệm can thiệp *(interventional experiments)*. Dự án được xây dựng nhằm giải quyết bài toán: *"Từ dữ liệu đã có, hãy tìm ra biến nào là nguyên nhân, biến nào là kết quả."*

Điểm cốt lõi của DeepANM là pipeline **ba pha kết hợp học thống kê và học sâu**:

```
Pha 1 — Thứ tự Nhân quả (HSIC TopoSort)
    Dữ liệu X ──► Tìm thứ tự [X₀ → X₂ → X₁ → X₃]

Pha 2 — Khớp SCM Neural (Neural SCM Fitting)
    X + thứ tự ──► Học ma trận trọng số W bằng mạng neural sâu

Pha 3 — Lọc Cạnh Thích nghi (Adaptive ATE Gate)
    W + X ──► Random Forest LASSO + Cổng ATE Neural ──► DAG nhị phân
```

---

## Vấn đề nghiên cứu

Các thuật toán khám phá nhân quả truyền thống (PC, GES) gặp khó khăn với:
- **Quan hệ phi tuyến** phức tạp trong dữ liệu sinh học và kinh tế
- **Nhiễu không đồng nhất** (heteroscedastic noise, multimodal distributions)
- **False Positive cao** khi không có tri thức miền (domain knowledge)

DeepANM giải quyết những hạn chế này bằng cách:
- Dùng **HSIC Asymmetry** để xác định chiều nhân quả phi tuyến
- Dùng **Random Forest** thay vì hồi quy tuyến tính để chọn cạnh
- Dùng **Đồ thị CHÍnh xác từ Pha 1** làm ràng buộc cứng cho mạng Neural ở Pha 2
- Dùng **Cổng ATE Thích nghi** để loại bỏ cạnh giả thay vì ngưỡng cố định

---

## Kết quả thực nghiệm

### Tập dữ liệu Sachs (Benchmark Sinh học — 11 biến, 7.466 mẫu)

| Cấu hình | TP | FP | **SHD** ↓ | F1 |
|:---|:---:|:---:|:---:|:---:|
| Không có tri thức trước (Zero Prior) | 11 | 11 | 15 | ~52% |
| Có Layer Constraint sinh học | 10 | 7 | **11** | ~59% |

> SHD = Structural Hamming Distance — thấp hơn là tốt hơn.

### Ablation Study — Đóng góp từng thành phần

| Cấu hình | TP | FP | **SHD** ↓ | Ghi chú |
|:---|:---:|:---:|:---:|:---|
| TopoSort + OLS (Baseline) | 13 | 42 | 42 | Tuyến tính, nhiều FP |
| + Random Forest | 12 | 21 | 23 | **−45% SHD** |
| + CI Pruning | 10 | 14 | 18 | **−22% SHD** |
| + Neural SCM Filter | 9–11 | 11–13 | **15–19** | Tốt nhất tổng thể |

### Khám phá thăm dò — Boston Housing (14 biến)

Một số quan hệ nhân quả tự động phát hiện:
- `NOX (ô nhiễm)` → `MEDV (giá nhà)` *(âm — ô nhiễm làm giảm giá nhà)*
- `RM (số phòng)` → `MEDV` *(dương — nhiều phòng tăng giá nhà)*
- `AGE (tuổi nhà)` → `NOX` *(dương — khu vực cũ, gần nhà máy, ô nhiễm hơn)*

---

## Cấu trúc dự án

```
DeepANM/
├── src/
│   ├── core/
│   │   ├── gppom_hsic.py      # Lõi mô hình: Gumbel-gate DAG, FastHSIC, phạt ALM
│   │   ├── mlp.py             # Mạng neural: Encoder, SEM, GMM Noise, PNL Decoder
│   │   └── toposort.py        # Pha 1: Sắp xếp topo HSIC Sink-First, O(N·D)
│   ├── models/
│   │   ├── deepanm.py         # API chính: fit, fit_bootstrap, get_dag_matrix
│   │   ├── fast_baseline.py   # FastANM: TopoSort + RF/CI, không dùng neural
│   │   └── lite_baseline.py   # LiteANM: neural nhẹ hơn, mặc định 50 epochs
│   └── utils/
│       ├── trainer.py         # Vòng lặp huấn luyện Augmented Lagrangian
│       ├── adaptive_lasso.py  # Pha 3: RF LASSO + CI Pruning + ATE Gate
│       └── visualize.py       # Vẽ DAG qua NetworkX + Matplotlib
├── examples/
│   ├── test_sachs.py          # Benchmark Sachs (có tri thức trước)
│   ├── ablation_study.py      # So sánh 4 cấu hình thành phần
│   ├── test_boston.py         # Khám phá nhân quả Boston Housing
│   └── test_synthetic.py      # Kiểm tra tổng hợp phi tuyến 5 node
└── tests/
    └── test_core.py           # 7 unit tests (pytest)
```

---

## Hướng dẫn chạy

### Cài đặt

```bash
git clone https://github.com/manhthai1706/DeepANM.git
cd DeepANM
pip install -r requirements.txt
```

**Yêu cầu:** Python ≥ 3.8 · PyTorch ≥ 2.0 · scikit-learn ≥ 1.0 · numpy · scipy · matplotlib

### Chạy thử nghiệm nhanh

```bash
# Benchmark Sachs (có tri thức sinh học)
python examples/test_sachs.py

# Ablation Study (so sánh 4 cấu hình)
python examples/ablation_study.py

# Khám phá Boston Housing
python examples/test_boston.py

# Tổng hợp phi tuyến 5 node
python examples/test_synthetic.py
```

### Dùng trong code

```python
import numpy as np
from src.models.deepanm import DeepANM

# Dữ liệu quan sát (n_samples, n_vars)
X = np.random.randn(2000, 5)

model = DeepANM(n_clusters=1, hidden_dim=32, lda=0.0)

# Chạy bootstrap (khuyến nghị cho kết quả ổn định)
prob_matrix, avg_ATE = model.fit_bootstrap(
    X,
    n_bootstraps=5,
    discovery_mode='fast',   # Dùng FastANM (TopoSort + RF) làm Pha 1+2
    apply_quantile=True
)

# Ma trận kề nhị phân
W = (prob_matrix > 0).astype(int)
print("Discovered edges:", W.sum(), "edges")
```

---

## Kiến trúc kỹ thuật

### Pha 1 — HSIC TopoSort

Xác định thứ tự nhân quả dựa trên **Tính bất đối xứng ANM**: nếu `X → Y` thì `HSIC(Y - f(X), X) < HSIC(X - g(Y), Y)`. Dùng **Random Fourier Features** để xấp xỉ HSIC với độ phức tạp `O(N·D)` thay vì `O(N²)`.

### Pha 2 — Neural SCM Fitter

Mạng neural học hàm nhân quả `fⱼ(X_parents)` cho từng biến j. Hàm mất mát:

```
L = MSE + λ·HSIC(nhiễu, nguyên nhân) + λ·HSIC(cơ chế Z, X)
  + 0.1·NLL_GMM + 0.1·L1(W) + 0.02·L2(W) + 0.1·KL
  + ALM: α·h(W) + 0.5·ρ·h(W)²
```

Cấu trúc đồ thị từ Pha 1 được dùng làm **mặt nạ topological cứng** — mạng neural không thể tạo cạnh ngoài thứ tự nhân quả.

### Pha 3 — Adaptive ATE Gate

Sau khi Neural SCM học xong, tính **ma trận ATE Jacobian** (tác động nhân quả trực tiếp). Loại bỏ 15% cạnh yếu nhất theo cường độ ATE — thay thế ngưỡng cố định bằng ngưỡng thích nghi theo phân phối thực tế của từng bộ dữ liệu.

---

## Chạy Unit Tests

```bash
pytest tests/ -v
```

```
tests/test_core.py::test_mlp_shapes               PASSED
tests/test_core.py::test_heterogeneous_noise_model PASSED
tests/test_core.py::test_fast_hsic                PASSED
tests/test_core.py::test_dag_penalty              PASSED
tests/test_core.py::test_gppomc_core_forward      PASSED
tests/test_core.py::test_global_ate_matrix        PASSED
tests/test_core.py::test_deepanm_integration      PASSED

7 passed in ~4s
```

---

## Tài liệu tham khảo

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

[13] amber0309. (2023). *ANM-MM: Additive Noise Model with Mixture Models*. GitHub repository, Retrieved from https://github.com/amber0309/ANM-MM

---

## Giấy phép

MIT License — xem [LICENSE](LICENSE).

---

<div align="center">
<sub>Đồ án Tốt nghiệp | Manh Thai | 2025</sub>
</div>
