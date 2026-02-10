# CausalFlow: Advanced Causal Discovery Framework

[![Architecture](https://img.shields.io/badge/Architecture-Detailed_Diagrams-blueviolet?style=flat-square)](ARCH.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

CausalFlow là một framework khám phá nhân quả (Causal Discovery) tiên tiến, kết hợp giữa học sâu (Deep Learning) và các lý thuyết thống kê hiện đại. Hệ thống được thiết kế để tự động nhận diện cấu trúc đồ thị nhân quả (DAG) từ dữ liệu quan sát phi tuyến và đa biến.

Dựa trên nền tảng của phương pháp GPPOM-HSIC, CausalFlow tích hợp các công nghệ SOTA để tối ưu hóa khả năng mô hình hóa nhiễu và tìm kiếm cấu trúc đồ thị liên tục.

## Cải tiến so với GPPOM-HSIC (base)

Mô hình CausalFlow mang đến những nâng cấp kỹ thuật quan trọng so với phiên bản GPPOM-HSIC nguyên bản của `amber0309`:

| Tính năng | GPPOM-HSIC (Base) | **CausalFlow (Enhanced)** |
| :--- | :--- | :--- |
| **Mô hình hóa Nhiễu** | Phân phối đơn giản / Gaussian | **Neural Spline Flows (NSF)**: Mô hình hóa nhiễu phi tuyến phức tạp bằng Spline Flows. |
| **Học cấu trúc DAG** | Hạn chế ở bài toán song biến | **NOTEARS Integration**: Tối ưu hóa ma trận kề DAG liên tục cho hệ thống đa biến. |
| **Phân tích Hướng** | Tối ưu hóa tự do (dễ lệch) | **Fixed-Structure Bivariate**: Khóa cứng hướng giả định để tối đa hóa độ chính xác HSIC. |
| **Tiền xử lý** | Cơ bản | **Advanced Pipeline**: Tích hợp Quantile Transformation và Isolation Forest để làm sạch dữ liệu. |
| **Kiến trúc MLP** | Standard MLP | **SOTA Backbone**: Tích hợp Self-Attention, Gated Residual Networks (GRN) và VAE. |

## Đặc điểm Kỹ thuật

- **Neural Spline Flows (NSF):** Khả năng mô hình hóa các hàm chuyển đổi nhiễu phi tuyến bậc cao, giúp trích xuất phần dư (residuals) sạch hơn cho các phép thử độc lập.
- **Differentiable DAG Discovery:** Sử dụng thuật toán NOTEARS để ép ma trận trọng số tuân thủ tính chất đồ thị không vòng (Acyclicity), cho phép tìm kiếm DAG đa biến trực tiếp bằng Gradient Descent.
- **Hybrid Objective Function:** Tối ưu hóa đồng thời sai số dự báo (MSE), tính không vòng (DAG Penalty) và tính độc lập nhân quả (HSIC Penalty).
- **Latent Mechanism Discovery:** Sử dụng đầu VAE kết hợp Gumbel-Softmax để tự động nhận diện các cơ chế nhân quả tiềm ẩn hoặc biến ẩn trong dữ liệu.

## Kết quả Thực nghiệm (Benchmarks)

Hiệu suất của hệ thống được kiểm chứng trên bộ dữ liệu sinh học thực tế **Sachs (Flow Cytometry)**, đạt kết quả vượt trội so với các phương pháp truyền thống:

- **Độ chính xác (Accuracy): 70.6%** (Xác định đúng hướng cho 12/17 cạnh nhân quả đã biết).
- **SHD (Structural Hamming Distance): 5** (Tổng số cạnh bị xác định sai hướng).
- Hệ thống thể hiện khả năng chống nhiễu mạnh mẽ và độ ổn định cao trên dữ liệu quan sát thực tế.

## Cài đặt

Cài đặt trực tiếp từ kho lưu trữ GitHub:

```bash
pip install git+https://github.com/manhthai1706/CausalFlow.git
```

## Hướng dẫn Sử dụng

### Phân tích hướng nhân quả song biến (Bivariate)
```python
from causalflow import ANMMM_cd_advanced
import numpy as np

# pair_data: mảng numpy shape [n_samples, 2]
direction, analyzer = ANMMM_cd_advanced(pair_data, lda=12.0)
# direction = 1 (X->Y) hoặc -1 (Y->X)
```

### Học cấu trúc DAG đa biến (Multivariate)
```python
from causalflow import CausalFlow
import numpy as np

model = CausalFlow(x_dim=11, n_clusters=3)
model.fit(data_matrix, epochs=200)
W_raw, W_binary = model.get_dag_matrix(threshold=0.1)
```

## Tham khảo

- **GPPOM-HSIC (amber0309).** [GitHub Repository](https://github.com/amber0309). (Cơ sở thuật toán ban đầu).
- **Zheng, X., et al. (2018).** "DAGs with NO TEARS." *NeurIPS*.
- **Durkan, C., et al. (2019).** "Neural Spline Flows." *NeurIPS*.
- **Zhang, K., & Hyvarinen, A. (2009).** "Identifiability of Post-Nonlinear Causal Model." *UAI*.
- **Lim, B., et al. (2021).** "Temporal Fusion Transformers (GRN)." *IJF*.
- **Paszke, A., et al. (2019).** "PyTorch: High-Performance Deep Learning Library." *NeurIPS*.

## License
Dự án được phát hành dưới giấy phép MIT License.
