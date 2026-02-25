# DeepANM: A Deep Learning Approach to Additive Noise Models for Causal Discovery

[![Architecture](https://img.shields.io/badge/Architecture-Details-blueviolet?style=flat-square)](ARCH.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square)](https://pytorch.org)

**DeepANM** (Deep Additive Noise Model) là một thư viện Python mã nguồn mở phục vụ cho bài toán Khám phá Nhân quả (Causal Discovery) từ dữ liệu quan sát. 

Dự án này được khởi tạo và truyền cảm hứng rất lớn từ framework **ANM-MM** (Additive Noise Model - Mixture Model) của tác giả [amber0309](https://github.com/amber0309/ANM-MM). Trong quá trình học hỏi và mở rộng, chúng tôi đã tích hợp thêm một số ý tưởng toán học và kiến trúc học sâu từ các bài báo khoa học nổi bật như **DECI (Causica)**, **NOTEARS**, và **DAGMA** nhằm nỗ lực cải thiện việc xử lý nhiễu phi tuyến tính và tối ưu hóa đồ thị DAG.

Mục tiêu của DeepANM là tạo ra một công cụ thực nghiệm, cung cấp các hàm API dễ sử dụng cho những ai đang tìm hiểu và nghiên cứu về AI Nhân quả.

---

## 🌟 Ý Tưởng và Nguồn Cảm Hứng (Inspirations)

DeepANM không phải là một thuật toán hoàn toàn mới, mà là sự tổng hợp và tinh chỉnh từ những công trình xuất sắc đi trước:

### 1. Mô hình Nhiễu Dị Thể (Lấy cảm hứng từ DECI / Causica)
Thay vì sử dụng các giả định nhiễu Gaussian cơ bản, mô hình tích hợp `HeterogeneousNoiseModel`. Ý tưởng này học hỏi từ kiến trúc **DECI** (Deep End-to-end Causal Inference) của Microsoft Research, sử dụng cấu trúc Gaussian Mixture Model (GMM) và thủ thuật Log-Sum-Exp. Mục đích là để mô hình có thể xấp xỉ tốt hơn các loại nhiễu phức tạp, lệch đuôi thường gặp trong dữ liệu thực tế.

### 2. Ràng Buộc Đồ Thị Hướng Không Chu Trình (NOTEARS & DAGMA)
Để đảm bảo ma trận trọng số (Adjacency Matrix) học được tạo thành một đồ thị DAG hợp lệ (không có chu trình), DeepANM áp dụng kỹ thuật tối ưu hóa liên tục (Continuous Optimization):
- **NOTEARS** (Zheng et al., 2018): Đặt nền móng cho việc chuyển bài toán tìm kiếm tổ hợp (combinatorial) khó khăn thành biểu thức toán học khả vi qua hàm vết (Trace exponential constraint). Chúng tôi cũng sử dụng L1/L2 Regularization từ bài báo này để làm đồ thị thưa thớt (sparse) hơn.
- **DAGMA** (Bello et al., 2022): Để giảm thiểu rủi ro biến mất gradient (Vanishing Gradients) khi đồ thị lớn có các chu trình dài, DeepANM tham khảo cách xấp xỉ Log-Determinant bằng đa thức (Polynomial series) của DAGMA, giúp việc tính toán Gradient ổn định hơn.

### 3. Đánh Giá Tính Độc Lập bằng FastHSIC (RFF)
Hàm mục tiêu cốt lõi của Additive Noise Models là đảm bảo phần dư (noise/residuals) độc lập thống kê với nguyên nhân (cause). Hệ thống sử dụng Hilbert-Schmidt Independence Criterion (**HSIC**). Để tăng tốc HSIC từ độ phức tạp O(N²) xuống O(N), chúng tôi đã tham khảo và áp dụng kỹ thuật **Random Fourier Features (RFF)** (Rahimi & Recht, 2007).

### 4. Neural ATE Jacobian
Để xác nhận lại độ mạnh của các liên kết nhân quả, DeepANM thử nghiệm áp dụng Đạo hàm Jacobian (Jacobian Matrix) lên mô hình học sâu để xấp xỉ **Average Treatment Effect (ATE)**. Khi kết hợp ma trận cấu trúc (topological mask) và ATE, mô hình hy vọng sẽ lọc bớt được các cạnh giả (false positives).

---

## 📦 Cài Đặt

```bash
git clone https://github.com/manhthai1706/DeepANM.git
cd DeepANM
pip install -r requirements.txt
```
> **Yêu cầu hệ thống:** Python ≥ 3.8 | PyTorch ≥ 2.0 | Numpy, Matplotlib, NetworkX

---

## 💡 Hướng Dẫn Sử Dụng (Quick Start)

Dưới đây là một ví dụ cơ bản về cách sử dụng class `DeepANM` để học và trực quan hóa ma trận liên kết.

```python
import numpy as np
from deepanm import DeepANM, plot_dag

# 1. Dữ liệu giả lập (Ví dụ: [2000 mẫu, 5 biến])
data = np.random.randn(2000, 5) 
labels = ["Gen_A", "Gen_B", "C", "D", "E"]

# 2. Khởi tạo DeepANM
model = DeepANM(
    n_clusters=2,     # Số cụm phân phối GMM
    hidden_dim=32,    # Kích thước lớp ẩn MLP
    lda=0.2          # Trọng số ép tính độc lập HSIC
)

# 3. Chạy quá trình học sử dụng Bootstrap (Stability Selection)
prob_matrix, avg_W_ATE = model.fit_bootstrap(
    X=data, 
    n_bootstraps=5, 
    threshold=0.01, 
    epochs=150, 
    lr=1e-2
)

# 4. Lọc các cạnh có độ tin cậy >= 30% (xuất hiện trong 30% số vòng bootstrap)
W_pred = (prob_matrix > 0.3).astype(int)

# 5. Sử dụng hàm plot_dag để trực quan hóa bằng NetworkX
plot_dag(
    W_matrix=W_pred * avg_W_ATE, 
    labels=labels,
    title="Causal Graph Discovery",
    threshold=0.01,
    save_path="my_discovery.png", # Có thể thay bằng hiển thị trực tiếp (bỏ save_path)
    node_size=2000
)
```

---

## 📊 Kết Quả Thực Nghiệm Tham Khảo (Sachs 2005)

Hệ thống cung cấp sẵn một ví dụ thực nghiệm trong thư mục `examples/sachs_eval.py` để chạy thử nghiệm trên bộ dữ liệu dòng chảy tín hiệu protein tế bào Sachs (11 biến, 2000 quan sát).

Dưới đây là một vài số liệu thực nghiệm đơn giản của thư viện trên bộ dữ liệu thuần quan sát (Continuous Observational), so với một số phương pháp truyền thống:

| Phương pháp | Loại thuật toán | SHD (Structural Hamming Distance) |
| :--- | :--- | :--- |
| PC Algorithm | Constraint-based | ~17 |
| GES | Score-based | ~15 |
| NOTEARS | Continuous Opt. | > 12 |
| DAG-GNN | VAE Deep Learning | ~19 |
| **DeepANM** | Học sâu + GMM Nhiễu + ALM | **Khoảng 22 - 26** |

*(Ghi chú: Kết quả SHD có thể dao động tùy thuộc vào phương pháp tiền xử lý (như Normalize hay Scaler) và tham số học. Mã nguồn DeepANM hiện đang nghiêng về việc loại trừ rủi ro tạo ra cạnh sai thay vì cố tìm mọi cạnh nhỏ.)*

---

## 🧠 Cấu Trúc Mã Nguồn

```text
DeepANM/
├── deepanm/
│   ├── core/
│   │   ├── mlp.py                  # Backbone Deep Causal (Encoder, GMM, PNL Decoder)
│   │   ├── gppom_hsic.py           # Logic hàm mất mát (ALM, NOTEARS/DAGMA Penalty)
│   │   └── kernels.py              # Thư viện RFF cho FastHSIC
│   ├── models/
│   │   ├── deepanm.py              # API chính xử lý huấn luyện và dự đoán
│   │   └── trainer.py              # Xử lý vòng lặp Augmented Lagrangian
│   └── utils/
│       └── visualize.py            # Hàm vẽ đồ thị plot_dag() dựa trên NetworkX
├── examples/
│   ├── sachs_eval.py               # Chạy thử nghiệm mạng tế bào sinh học Sachs
│   └── boston_global_discovery.py  # Chạy thử nghiệm đánh giá Boston Housing
└── tests/
    └── test_core.py                # Bộ Unit Test sử dụng Pytest
```

---

## 📜 Tài Liệu Học Thuật Xin Chân Thành Cảm Ơn

DeepANM trân trọng ghi nhận và xin gửi lời cảm ơn tới kiến thức quý giá từ các tác giả của những công trình sau:

1. **amber0309** và kho mã nguồn [ANM-MM](https://github.com/amber0309/ANM-MM) đã cung cấp khung sườn ban đầu của bài toán VAE Clustering.
2. **Bello, K. et al. (2022).** *"DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization."*
3. **Ge, T. et al. (2023).** *"Causica (DECI) - Deep End-to-end Causal Inference."* (Microsoft Research).
4. **Zheng, X. et al. (2018).** *"DAGs with NO TEARS: Continuous Optimization for Structure Learning."* NeurIPS.
5. **Rahimi, A. & Recht, B. (2007).** *"Random Features for Large-Scale Kernel Machines."* NeurIPS.
6. **Zhang, K. & Hyvarinen, A. (2009).** *"On the Identifiability of the Post-Nonlinear Causal Model."* UAI.

---

## License
Dự án được phân phối dưới giấy phép [MIT](LICENSE). Vui lòng trích dẫn hoặc giữ nguồn khi bạn phát triển lại từ dự án này cũng như các công trình tiền nhiệm.
