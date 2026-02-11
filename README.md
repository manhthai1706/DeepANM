# CausalFlow: Advanced Causal Discovery Framework

[![Architecture](https://img.shields.io/badge/Architecture-Detailed_Diagrams-blueviolet?style=flat-square)](ARCH.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

CausalFlow là một engine khám phá nhân quả (Causal Discovery) mạnh mẽ, cung cấp nền tảng tính toán cho hệ thống phân tích ANM-MM (Post-Nonlinear Model with Mixture Mechanisms). Dự án tập trung vào việc mô hình hóa các cơ chế nhân quả phức tạp thông qua sự kết hợp giữa Deep Learning và các phép thử thống kê độc lập.

Về bản chất, CausalFlow đóng vai trò là "động cơ" (backbone) xử lý dữ liệu phi tuyến và đa biến, phục vụ cho quy trình suy diễn nhân quả chính xác của thuật toán ANM-MM.

## Cải tiến so với GPPOM-HSIC (base)

CausalFlow nâng cấp khả năng của GPPOM-HSIC bằng cách tích hợp các công nghệ hiện đại, biến nó thành một engine chuyên sâu cho việc khám phá hướng nhân quả:

| Tính năng | GPPOM-HSIC (Base) | **CausalFlow Architecture (Enhanced)** |
| :--- | :--- | :--- |
| **Mô hình hóa Nhiễu** | Phân phối đơn giản / Gaussian | **Neural Spline Flows (NSF)**: Trích xuất phần dư (residuals) sạch thông qua mô hình hóa nhiễu phi tuyến bậc cao. |
| **Học cấu trúc DAG** | Hạn chế ở bài toán song biến | **NOTEARS Integration**: Hỗ trợ tìm kiếm cấu trúc đa biến liên tục để dẫn dắt cơ chế nhân quả. |
| **Phân tích Hướng** | Tối ưu hóa tự do | **Fixed-Structure Bivariate**: Quy trình khóa hướng chặt chẽ, tối ưu hóa độ nhạy cho phép thử HSIC. |
| **Tiền xử lý** | Cơ bản | **Standard Sklearn Pipeline**: Kết hợp Quantile Transform và Isolation Forest để triệt tiêu nhiễu sinh học. |
| **Kiến trúc Backbone** | Standard MLP | **Advanced MLP**: Tích hợp Self-Attention, Gated Residual Networks (GRN) để tự động hóa việc chọn lọc đặc trưng. |

## Đặc điểm Kỹ thuật của Engine

- **Neural Spline Flows (NSF):** Đảm bảo tính khả nghịch trong việc chuyển đổi nhiễu, cho phép framework mô tả chính xác các cơ chế Post-Nonlinear (PNL).
- **Differentiable Structural Optimization:** Sử dụng các thành phần từ thuật toán NOTEARS để tối ưu hóa ma trận kề, định hướng cho việc tìm kiếm quan hệ phục thuộc giữa các biến.
- **Adaptive Kernel Selection:** Tự động tối ưu hóa tham số hàm nhân cho các phép thử độc lập HSIC, giúp hệ thống nhạy bén hơn với các tín hiệu nhân quả yếu.
- **Latent Confounder Identification:** Tích hợp đầu VAE để nhận diện các cơ chế hỗn hợp (mixture mechanisms) hoặc sự hiện diện của các biến ẩn trong hệ thống.

## Kết quả Thực nghiệm (Benchmarks)

Hiệu suất của quy trình phân tích sử dụng động cơ CausalFlow được kiểm chứng trên bộ dữ liệu sinh học thực tế **Sachs**, đạt kết quả vượt trội so với các phương pháp truyền thống:

- **Độ chính xác hướng (Accuracy): 70.6%** (Xác định đúng hướng cho 12/17 cạnh nhân quả).
- **SHD (Structural Hamming Distance): 5**.
- Kết quả chứng minh sức mạnh của engine trong việc trích xuất sạch phần dư để đưa ra kết luận về tính độc lập nhân quả.

## Cài đặt

Cài đặt trực tiếp từ kho lưu trữ GitHub:

```bash
pip install git+https://github.com/manhthai1706/CausalFlow.git
```

## Hướng dẫn Sử dụng (Entry Points)

Quy trình sử dụng tập trung vào các hàm phân tích hướng nhân quả mạnh mẽ (ANM-MM):

### Phân tích hướng nhân quả song biến
Đây là cách sử dụng chính để đạt được hiệu suất tối ưu:
```python
from causalflow import ANMMM_cd_advanced
import numpy as np

# pair_data: mảng numpy [n_samples, 2]
# Sử dụng động cơ CausalFlow bên dưới để thực hiện Fixed-Structure Test
direction, analyzer = ANMMM_cd_advanced(pair_data, lda=12.0)
# direction = 1 (X->Y) hoặc -1 (Y->X)
```

### Học cấu trúc sơ bộ cho hệ đa biến
Sử dụng engine để ước lượng ma trận DAG:
```python
from causalflow import CausalFlow
import numpy as np

# Khởi tạo engine CausalFlow
engine = CausalFlow(x_dim=11, n_clusters=3)
engine.fit(data_matrix, epochs=200)

# Trích xuất ma trận cấu trúc học được
W_raw, W_binary = engine.get_dag_matrix(threshold=0.1)
```

## Tham khảo

- **ANM-MM (amber0309).** [GitHub Repository](https://github.com/amber0309/ANM-MM). (Cơ sở thuật toán ban đầu).
- **Zheng, X., et al. (2018).** "DAGs with NO TEARS: Continuous Optimization for Structure Learning." *NeurIPS*.
- **Durkan, C., et al. (2019).** "Neural Spline Flows." *NeurIPS*.
- **Zhang, K., & Hyvarinen, A. (2009).** "On the Identifiability of the Post-Nonlinear Causal Model." *UAI*.
- **Rahimi, A., & Recht, B. (2007).** "Random Features for Large-Scale Kernel Machines." *NeurIPS*. (Tối ưu hóa tốc độ HSIC thông qua RFF).
- **Gretton, A., et al. (2007).** "A Kernel Statistical Test of Independence." *NeurIPS*. (Nền tảng của các phép thử độc lập HSIC).
- **Vaswani, A., et al. (2017).** "Attention Is All You Need." *NeurIPS*. (Cơ chế Self-Attention trong lớp MLP để trọng số hóa đặc trưng).
- **Jang, E., et al. (2016).** "Categorical Reparameterization with Gumbel-Softmax." *ICLR*. (Cơ chế phân cụm cơ chế nhân quả có thể đạo hàm).
- **Kingma, D. P., & Welling, M. (2013).** "Auto-Encoding Variational Bayes." *ICLR*. (Kiến trúc VAE để phát hiện cơ cấu tiềm ẩn).
- **He, K., et al. (2016).** "Deep Residual Learning for Image Recognition." *CVPR*. (Cơ chế Residual Connections trong khối ResBlock).
- **Ba, J. L., et al. (2016).** "Layer Normalization." *arXiv*. (Kỹ thuật chuẩn hóa lớp để ổn định quá trình huấn luyện).
- **Hendrycks, D., & Gimpel, K. (2016).** "Gaussian Error Linear Units (GELUs)." *arXiv*. (Hàm kích hoạt GELU trong mô hình MLP).
- **Lim, B., et al. (2021).** "Temporal Fusion Transformers." *International Journal of Forecasting*. (Cấu trúc Gated Residual Network - GRN cho việc chọn lọc đặc trưng).
- **Loshchilov, I., & Hutter, F. (2017).** "Decoupled Weight Decay Regularization." *ICLR*. (Thuật toán tối ưu AdamW sử dụng trong Trainer).
- **Liu, F. T., et al. (2008).** "Isolation Forest." *ICDM*. (Sử dụng lọc Outliers trong tiền xử lý).
- **Pedregosa, F., et al. (2011).** "Scikit-learn: Machine Learning in Python." *JMLR*. (Cung cấp QuantileTransformer).
- **Paszke, A., et al. (2019).** "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS*.

## License
Dự án được phát hành dưới giấy phép MIT License.
