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

## Tri ân & Tham khảo

Dự án này được tôi phát triển dựa trên cảm hứng và mã nguồn gốc từ nghiên cứu GPPOM-HSIC của [amber0309](https://github.com/amber0309). Tôi xin gửi lời cảm ơn chân thành đến tác giả vì những đóng góp nền tảng cho cộng đồng nghiên cứu nhân quả.

- **Zheng, X., et al. (2018).** "DAGs with NO TEARS: Continuous Optimization for Structure Learning." *NeurIPS*. (Cơ sở cho thuật toán học đồ thị liên tục).
- **Durkan, C., et al. (2019).** "Neural Spline Flows." *NeurIPS*. (Cơ sở cho việc mô hình hóa nhiễu phi tuyến bằng Spline).
- **Gretton, A., et al. (2005).** "Measuring Statistical Dependence with Hilbert-Schmidt Norms." *ALT*. (Nền tảng của phép thử độc lập HSIC).
- **Gretton, A., et al. (2007).** "A Kernel Statistical Test of Independence." *NeurIPS*. (Phép thử HSIC với xấp xỉ Gamma và Permutation test).
- **Kingma, D. P., & Welling, M. (2013).** "Auto-Encoding Variational Bayes." *ICLR*. (Cơ sở cho cơ chế VAE phát hiện cơ chế tiềm ẩn trong MLP).
- **Liu, F. T., et al. (2008).** "Isolation Forest." *ICDM*. (Công nghệ lọc Outlier để làm sạch dữ liệu protein).
- **Pedregosa, F., et al. (2011).** "Scikit-learn: Machine Learning in Python." *JMLR*. (Cung cấp QuantileTransformer và công cụ thực nghiệm mạnh mẽ).
- **Paszke, A., et al. (2019).** "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS*.

## License
Dự án được phát hành dưới giấy phép MIT License.
