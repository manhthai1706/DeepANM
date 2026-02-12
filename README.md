# CausalFlow: Deep Neural Causal Discovery Architecture

[![Architecture](https://img.shields.io/badge/Architecture-Detailed_Diagrams-blueviolet?style=flat-square)](ARCH.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

CausalFlow là một **kiến trúc mạng nơ-ron sâu (Deep Neural Architecture)** hợp nhất, được thiết kế chuyên biệt cho bài toán khám phá nhân quả. Hệ thống tích hợp trực tiếp các cơ chế mô hình hóa phi tuyến và quy trình suy diễn nhân quả vào trong một mô hình duy nhất, giúp tối ưu hóa khả năng nhận diện cấu trúc từ dữ liệu phức tạp.

Dựa trên nền tảng của phương pháp ANM-MM, CausalFlow đóng vai trò là một engine tính toán mạnh mẽ, kết hợp giữa học sâu và các lý thuyết nhân quả hiện đại.

## Cải tiến so với GPPOM-HSIC (base)

CausalFlow chuyển đổi nền tảng GPPOM-HSIC từ một bộ công cụ rời rạc thành một **Kiến trúc Mô hình Hợp nhất**:

| Tính năng | GPPOM-HSIC (Base) | **CausalFlow Architecture (Unified)** |
| :--- | :--- | :--- |
| **Kiến trúc** | Module rời rạc | **Unified Model**: Tích hợp sẵn `fit`, `predict_direction`, `counterfactual`. |
| **Sử dụng** | Gọi hàm thủ công | **Smart Init**: Tự động train và suy diễn chiều dữ liệu khi khởi tạo. |
| **Mô hình hóa** | Phân phối đơn giản | **Neural Spline Flows (NSF)**: Học sâu phân phối nhiễu phi tuyến. |
| **Backbone** | Standard MLP | **Deep ResNet + Self-Attention**: Tự động chọn lọc đặc trưng quan trọng. |
| **Suy diễn** | Tìm kiếm tổ hợp | **Differentiable DAG Discovery**: Tối ưu hóa đồ thị liên tục (NOTEARS). |

## Đặc điểm Kỹ thuật

- **Deep Neural Backbone:** Hệ thống sử dụng các khối ResNet và Gated Residual Networks (GRN) để xử lý dữ liệu, đảm bảo tính ổn định và khả năng hội tụ cao.
- **Neural Spline Flows (NSF):** Mô hình hóa nhiễu thông qua các hàm Spline đơn điệu, giúp trích xuất phần dư sạch hơn cho các phép thử độc lập.
- **Unified Inference API:** Cung cấp các phương thức cấp cao để thực hiện phân tích độ ổn định (stability) và dự báo giả tưởng (counterfactual) trực tiếp từ mô hình.
- **Hybrid Loss Function:** Tối ưu hóa đồng thời độ chính xác dự báo (MSE), tính không vòng của đồ thị (DAG penalty) và tính độc lập nhân quả (HSIC).

## Kết quả Thực nghiệm

Hiệu suất được kiểm chứng trên bộ dữ liệu sinh học thực tế **Sachs**, đạt kết quả vượt trội:

- **Độ chính xác hướng (Accuracy): 70.6%** (12/17 cạnh đúng).
- **SHD (Structural Hamming Distance): 5**.
- Khả năng xử lý phi tuyến mạnh mẽ, lọc nhiễu hiệu quả bằng Isolation Forest và Quantile Transformation.

## Cài đặt

```bash
pip install git+https://github.com/manhthai1706/CausalFlow.git
```

## Hướng dẫn Sử dụng (Unified API)

### 1. Khám phá hướng nhân quả song biến (SOTA Pattern)
Sử dụng quy trình Hypotheses Testing tích hợp để đạt độ chính xác cao nhất:
```python
from causalflow import CausalFlow

# Khởi tạo và dự đoán hướng ngay lập tức (X->Y: 1, Y->X: -1)
model = CausalFlow(lda=12.0)
direction = model.predict_direction(pair_data)
```

### 2. Huấn luyện và Suy diễn Đa biến
Mô hình tự động nhận diện chiều dữ liệu và huấn luyện:
```python
# Cách 1: Train ngay khi khởi tạo
model = CausalFlow(data=data_matrix, epochs=200)

# Cách 2: Gọi model như một hàm để train
model = CausalFlow()
model(data_matrix, epochs=200)

# Trích xuất ma trận DAG
W_raw, W_binary = model.get_dag_matrix()
```

### 3. Phân tích Giả tưởng (Counterfactual)
```python
# Dự đoán Y sẽ thế nào nếu thay đổi giá trị của X
y_cf = model.predict_counterfactual(x_orig, y_orig, x_new)
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
