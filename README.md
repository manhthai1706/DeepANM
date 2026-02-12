# CausalFlow: Unified Deep Neural Engine for Causal Discovery

[![Architecture](https://img.shields.io/badge/Architecture-Detailed_Diagrams-blueviolet?style=flat-square)](ARCH.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

**CausalFlow** là một hệ thống khám phá nhân quả (Causal Discovery) tiên tiến, được xây dựng như một công cụ học sâu hợp nhất (Unified Deep Learning Engine). Dự án tập hợp các công nghệ SOTA trong mô hình hóa phi tuyến và tối ưu hóa đồ thị để giải quyết bài toán suy diễn cấu trúc nhân quả từ dữ liệu quan sát phức tạp.

Khác với các công cụ truyền thống, CausalFlow đóng gói toàn bộ quy trình từ tiền xử lý, huấn luyện cơ chế đến phân tích giả tưởng vào trong một kiến trúc mạng nơ-ron sâu duy nhất, mang lại hiệu suất vượt trội và sự tiện dụng tối đa.

---

## 🛠 Công nghệ & Kiến trúc Cốt lõi

CausalFlow sở hữu một "Backbone" kỹ thuật mạnh mẽ, kết hợp giữa học sâu hiện đại và lý thuyết nhân quả:

- **Deep Neural Backbone (ResNet + GRN + Attention):** Sử dụng các khối ResNet kết hợp với Gated Residual Networks (GRN) và cơ chế Self-Attention để tự động sàng lọc đặc trưng, giúp mô hình nhạy bén với các tín hiệu nhân quả thực sự và loại bỏ biến nhiễu.
- **Neural Spline Flows (NSF):** Tích hợp công nghệ Normalizing Flows thông qua các hàm Spline đơn điệu để mô hình hóa phân phối nhiễu phi tuyến bậc cao, đảm bảo việc trích xuất phần dư (residuals) đạt độ tinh khiết tối ưu.
- **Differentiable DAG Discovery (NOTEARS):** Sử dụng phương pháp tối ưu hóa liên tục để tìm kiếm đồ thị nhân quả đa biến, đảm bảo tính không vòng (Acyclicity) thông qua các ràng buộc đại số đạo hàm được.
- **Hybrid Independence Testing (HSIC):** Kết hợp Hilbert-Schmidt Independence Criterion làm hàm phạt (penalty) để cưỡng bức tính độc lập nhân quả giữa các biến và phần dư.

---

## 🚀 Sự tiến hóa từ Base Project (amber0309)

CausalFlow không chỉ kế thừa mà còn tái cấu trúc toàn diện dự án ANM-MM/GPPOM-HSIC ban đầu:

| Khía cạnh | Base Project (amber0309) | **CausalFlow (Ours)** | Giá trị hệ thống |
| :--- | :--- | :--- | :--- |
| **Triết lý thiết kế** | Tập hợp các Script nghiên cứu | **Unified Machine Learning Engine** | Chuyển đổi từ công cụ đơn lẻ thành một Framework hoàn chỉnh. |
| **Kiến trúc mã nguồn** | Phẳng & Phân mảnh | **Phân lớp Chuyên nghiệp (Core/Models/Utils)** | Dễ dàng bảo trì, mở rộng và tích hợp vào các hệ thống khác. |
| **Mô hình hóa Nhiễu** | Giả định đơn giản | **Neural Spline Flows (NSF)** | Khả năng học các phân phối nhiễu phi tuyến phức tạp nhất. |
| **Cấu trúc Đồ thị** | Giới hạn ở song biến (Bivariate) | **NOTEARS (Multivariate DAG)** | Khám phá cấu trúc của hàng chục biến cùng lúc một cách đồng bộ. |
| **Giao diện lập trình** | Hàm rời rạc, gọi thủ công | **Sklearn-compatible OO API** | Thân thiện với lập trình viên: `model.fit()`, `model.predict()`. |
| **Xử lý dữ liệu** | Tiền xử lý tối giản | **SOTA Hybrid Pipeline (IsoForest + QT)** | Loại bỏ nhiễu sinh học, tăng tính hội tụ cho mô hình sâu. |
| **Phân tích nâng cao** | Không hỗ trợ | **Counterfactual & Stability Suite** | Cho phép mô phỏng kịch bản giả tưởng "What-if" và thẩm định kết quả. |
| **Độ tin cậy** | Chỉ test trên dữ liệu mô phỏng | **Real-world Sachs Benchmark (70.6%)** | Được kiểm chứng trên bộ dữ liệu protein thực tế khắt khe nhất. |
| **Tài liệu & Đặc tả** | README ngắn gọn | **Hệ thống ARCH.md & Đặc tả chi tiết** | Minh bạch về thuật toán và cấu trúc sơ đồ hoạt động. |

---

## 📦 Cài đặt

```bash
pip install git+https://github.com/manhthai1706/CausalFlow.git
```

## 💡 Hướng dẫn Sử dụng (Unified API)

### 1. Khám phá hướng nhân quả song biến (SOTA Pattern)
Tự động chạy quy trình Hypotheses Testing tích hợp:
```python
from causalflow import CausalFlow

model = CausalFlow(lda=12.0)
direction = model.predict_direction(pair_data) # Trả về 1 (X->Y) hoặc -1 (Y->X)
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

---

## 📊 Kết quả Thực nghiệm

Hiệu suất được kiểm chứng trên bộ dữ liệu sinh học thực tế **Sachs** (Protein Signaling Network), đạt kết quả vượt trội:

- **Độ chính xác xác định hướng (Accuracy): 70.6%** (12/17 cạnh được xác định đúng).
- **SHD (Structural Hamming Distance): 5**.
- Khả năng xử lý phi tuyến mạnh mẽ, lọc nhiễu hiệu quả bằng Isolation Forest và Quantile Transformation.

### Chỉ số hiệu năng so sánh

| Thuật toán | Xử lý Phi tuyến | Độ chính xác (Sachs) | SHD | Tính ổn định |
| :--- | :--- | :--- | :--- | :--- |
| **PC Algorithm** | Kém | ~50-55% | Cao | Thấp |
| **NOTEARS (Original)** | Trung bình | ~60% | > 8 | Trung bình |
| **CausalFlow (Ours)** | **Rất tốt (NSF)** | **70.6%** | **5** | **Cao** |

---

## 📚 Tham khảo

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

## ⚖️ License
Dự án được phát hành dưới giấy phép MIT License.
