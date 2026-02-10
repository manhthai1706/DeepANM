# CausalFlow: My Advanced Causal Discovery Framework

CausalFlow là dự án cá nhân của tôi tập trung vào việc nghiên cứu và triển khai các thuật toán khám phá nhân quả (Causal Discovery) tiên tiến. Mục tiêu của dự án là xây dựng một framework mạnh mẽ, tích hợp học sâu để giải quyết bài toán tìm kiếm cấu trúc nhân quả trong các hệ thống đa biến phức tạp.

Dựa trên nền tảng nghiên cứu GPPOM-HSIC, tôi đã nâng cấp và tối ưu hóa hệ thống này để đạt được hiệu suất cao hơn trong việc xử lý dữ liệu phi tuyến và nhiễu thực tế.

## Những cải tiến tôi đã triển khai (Personal Contributions)

Trong phiên bản này, tôi đã tập trung thực hiện các nâng cấp kỹ thuật quan trọng:

- **Neural Spline Flows (NSF):** Tôi tích hợp NSF để thay thế các mô hình nhiễu đơn giản, giúp framework mô hình hóa được các phân phối nhiễu phức tạp bằng các hàm Spline đơn điệu bậc ba.
- **NOTEARS Differentiable DAG:** Áp dụng phương pháp tối ưu hóa đồ thị liên tục để tìm kiếm cấu trúc DAG đa biến, cho phép mô hình học trực tiếp bằng Gradient Descent.
- **Fixed-Structure Bivariate Testing:** Để cải thiện độ chính xác hướng nhân quả, tôi đã triển khai cơ chế khóa hướng (Fixed-structure fit) kết hợp với phép thử HSIC, giúp triệt tiêu nhiễu từ các biến ẩn.
- **Advanced Preprocessing Pipeline:** Tôi xây dựng luồng xử lý dữ liệu chuyên sâu sử dụng `QuantileTransformer` để chuẩn hóa phân phối và `IsolationForest` để lọc nhiễu sinh học.

## Kết quả đạt được (My Benchmarks)

Tôi đã thực hiện kiểm chứng mô hình trên bộ dữ liệu sinh học thực tế **Sachs (Flow Cytometry)** với các kết quả cụ thể:

- **Độ chính xác hướng nhân quả (Accuracy): 70.6%** (Xác định đúng hướng cho 12/17 cạnh quan trọng).
- **Chỉ số SHD (Structural Hamming Distance): 5**.
- Kết quả này vượt trội đáng kể so với các thuật quy trình truyền thống như PC hay GES trên cùng một tập dữ liệu quan sát.

## Cài đặt và Sử dụng

Bạn có thể cài đặt thư viện này trực tiếp từ GitHub của tôi:

```bash
pip install git+https://github.com/manhthai1706/CausalFlow.git
```

### Cách tôi sử dụng mô hình để phân tích:

```python
from causalflow import ANMMM_cd_advanced, CausalFlow
import numpy as np

# 1. Phân tích hướng nhân quả giữa 2 protein
direction, analyzer = ANMMM_cd_advanced(pair_data, lda=12.0)

# 2. Học cấu trúc DAG cho toàn bộ 11 biến trong tập Sachs
model = CausalFlow(x_dim=11, n_clusters=3)
model.fit(data)
W_raw, W_binary = model.get_dag_matrix(threshold=0.1)
```

## Tham khảo

- **Zheng, X., et al. (2018).** "DAGs with NO TEARS: Continuous Optimization for Structure Learning." *NeurIPS*. (Thuật toán học đồ thị DAG liên tục).
- **Durkan, C., et al. (2019).** "Neural Spline Flows." *NeurIPS*. (Mô hình hóa nhiễu phi tuyến phức tạp bằng Spline Flows).
- **Zhang, K., & Hyvarinen, A. (2009).** "On the Identifiability of the Post-Nonlinear Causal Model." *UAI*. (Cơ sở cho mô hình PNL mà tôi đã tích hợp).
- **Rahimi, A., & Recht, B. (2007).** "Random Features for Large-Scale Kernel Machines." *NeurIPS*. (Tối ưu hóa tốc độ HSIC thông qua RFF).
- **Gretton, A., et al. (2007).** "A Kernel Statistical Test of Independence." *NeurIPS*. (Nền tảng của các phép thử độc lập HSIC).
- **Vaswani, A., et al. (2017).** "Attention Is All You Need." *NeurIPS*. (Cơ chế Self-Attention trong lớp MLP để trọng số hóa đặc trưng).
- **Jang, E., et al. (2016).** "Categorical Reparameterization with Gumbel-Softmax." *ICLR*. (Cơ chế phân cụm cơ chế nhân quả có thể đạo hàm).
- **Kingma, D. P., & Welling, M. (2013).** "Auto-Encoding Variational Bayes." *ICLR*. (Kiến trúc VAE để phát hiện cơ cấu tiềm ẩn).
- **He, K., et al. (2016).** "Deep Residual Learning for Image Recognition." *CVPR*. (Cơ chế Residual Connections trong khối ResBlock).
- **Ba, J. L., et al. (2016).** "Layer Normalization." *arXiv*. (Kỹ thuật chuẩn hóa lớp để ổn định quá trình huấn luyện).
- **Hendrycks, D., & Gimpel, K. (2016).** "Gaussian Error Linear Units (GELUs)." *arXiv*. (Hàm kích hoạt GELU trong mô hình MLP).
- **Lim, B., et al. (2021).** "Temporal Fusion Transformers." *International Journal of Forecasting*. (Cấu trúc Gated Residual Network - GRN cho việc chọn lọc đặc trưng).
- **Loshchilov, I., & Hutter, F. (2017).** "Decoupled Weight Decay Regularization." *ICLR*. (Thuật toán tối ưu AdamW tôi sử dụng trong Trainer).
- **Liu, F. T., et al. (2008).** "Isolation Forest." *ICDM*. (Sử dụng để loại bỏ Outliers trong tiền xử lý).
- **Pedregosa, F., et al. (2011).** "Scikit-learn: Machine Learning in Python." *JMLR*. (Cung cấp QuantileTransformer).
- **Paszke, A., et al. (2019).** "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS*.

## License
Dự án được phát hành dưới giấy phép MIT License.
