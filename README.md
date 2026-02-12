# CausalFlow: Kiến trúc Mạng Nơ-ron Sâu Hợp nhất trong Khám phá Cấu trúc Nhân quả

[![Architecture](https://img.shields.io/badge/Architecture-Detailed_Diagrams-blueviolet?style=flat-square)](ARCH.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## 📝 Tổng quan (Abstract)

Dự án **CausalFlow** đề xuất một giải pháp học sâu (Deep Learning) tiên tiến nhằm giải quyết bài toán khám phá cấu trúc nhân quả (Causal Discovery) từ dữ liệu quan sát phi tuyến. Hệ thống được thiết kế dựa trên triết lý "Mô hình hợp nhất" (Unified Model), tích hợp đồng thời việc học đặc trưng thông qua các mạng nơ-ron sâu và tối ưu hóa đồ thị có hướng không vòng (DAG) dựa trên các ràng buộc toán học liên tục. Kết quả thực nghiệm trên bộ dữ liệu protein thực tế cho thấy mô hình đạt độ chính xác cao trong việc xác định hướng nhân quả, đồng thời cung cấp khả năng phân tích giả tưởng mạnh mẽ cho các bài toán can thiệp dữ liệu.

## 1. Giới thiệu (Introduction)

Trong bối cảnh khoa học dữ liệu hiện đại, việc xác định mối quan hệ nhân quả (Causality) thay vì chỉ dừng lại ở mối liên quan (Correlation) đóng vai trò sống còn trong các lĩnh vực như y sinh, kinh tế và trí tuệ nhân tạo giải thích được. Các phương pháp truyền thống thường gặp khó khăn khi đối mặt với dữ liệu có tính chất phi tuyến bậc cao và phân phối nhiễu phức tạp.

**CausalFlow** ra đời với mục tiêu chuyển đổi các phương pháp nghiên cứu rời rạc (tình trạng chung của các thuật toán tiền nhiệm như GPPOM-HSIC) thành một **Engine nhân quả** hoàn chỉnh. Bằng cách kết hợp giữa mạng nơ-ron Spline Flow và tối ưu hóa NOTEARS, CausalFlow không chỉ tìm thấy cấu trúc đồ thị mà còn học được cơ chế sinh dữ liệu (Data Generating Process), cho phép thực hiện các phép thử "What-if" đầy tiềm năng.

---

## 2. Đặc tả Kỹ thuật và Công nghệ SOTA

Kiến trúc CausalFlow được xây dựng từ các thành phần công nghệ hiện đại nhất (State-of-the-art):

*   **Deep Neural Backbone (ResNet + GRN + Attention):** Hệ thống trích xuất đặc trưng sử dụng các khối ResNet kết hợp với Gated Residual Networks (GRN) cho phép mô hình tự động chọn lọc các biến đầu vào có ảnh hưởng nhân quả, đồng thời bỏ qua các biến gây nhiễu.
*   **Neural Spline Flows (NSF):** Khác với các giả định nhiễu đơn giản, CausalFlow sử dụng các hàm Spline đơn điệu để mô hình hóa hàm mật độ xác suất của nhiễu. Điều này giúp mô hình "làm sạch" dữ liệu và tách biệt nguyên nhân - kết quả một cách chính xác hơn trong môi trường phi tuyến.
*   **Differentiable DAG Discovery (NOTEARS):** Chuyển đổi bài toán tìm kiếm đồ thị rời rạc thành bài toán tối ưu hóa liên tục. Ràng buộc toán học đảm bảo đồ thị đầu ra luôn đạt tính không vòng (Acyclicity).
*   **Hilbert-Schmidt Independence Criterion (HSIC):** Được sử dụng như một hàm phạt (Penalty function) trong quá trình huấn luyện để cưỡng bức tính độc lập thống kê giữa phần dư (residuals) và biến nguyên nhân, đây là điều kiện tiên quyết trong lý thuyết nhân quả ANM.

---

## 3. Sự tiến hóa và Cải tiến Hệ thống

Bảng dưới đây tóm tắt sự lột xác của dự án từ phiên bản nghiên cứu ban đầu (`amber0309`) sang Framework `CausalFlow`:

| Khía cạnh | Dự án Base (amber0309) | **CausalFlow (Ours)** | Giá trị khoa học |
| :--- | :--- | :--- | :--- |
| **Kiến trúc mã** | Script rời rạc, cấu trúc phẳng | **Cấu trúc phân lớp (Modularized)** | Tăng tính tái sử dụng và khả năng bảo trì. |
| **Mô hình hóa** | MLP đơn giản, nhiễu Gauss | **Deep ResNet & Spline Flows** | Khả năng biểu diễn các cơ chế phi tuyến cực kỳ phức tạp. |
| **Tìm kiếm đồ thị** | Greedy Search / Bivariate | **Multivariate Optimization** | Tìm kiếm cấu trúc của toàn bộ hệ thống biến đồng thời. |
| **API Giao tiếp** | Gọi hàm thủ công | **Unified Model Class API** | Đồng nhất hóa luồng huấn luyện và suy diễn (Inference). |
| **Ứng dụng** | Chỉ tìm hướng | **Counterfactual & Stability** | Khả năng thẩm định độ bền vững và mô phỏng can thiệp. |
| **Tiền xử lý** | Cơ bản | **Hybrid Preprocessing Pipeline** | (IsoForest + Quantile) Tối ưu hóa dữ liệu đầu vào. |

---

## 4. Hướng dẫn Cài đặt và Sử dụng

### Cài đặt
```bash
pip install git+https://github.com/manhthai1706/CausalFlow.git
```

### Sử dụng API Hợp nhất
Dự án được thiết kế để sử dụng đơn giản như các thư viện ML hiện đại:

**1. Xác định hướng nhân quả cho cặp biến:**
```python
from causalflow import CausalFlow
model = CausalFlow(lda=12.0)
direction = model.predict_direction(data) # Trả về hướng tối ưu
```

**2. Huấn luyện đa biến và lấy ma trận DAG:**
```python
model = CausalFlow(data=X_matrix) # Tự động nhận diện và huấn luyện
W_raw, W_binary = model.get_dag_matrix()
```

---

## 5. Kết quả Thực nghiệm và Thảo luận (Results)

Hiệu suất của CausalFlow được kiểm chứng khắt khe trên bộ dữ liệu thực tế **Sachs (Protein Signaling Network)**:

*   **Độ chính xác xác định hướng (Accuracy): 70.6%** (Xác định đúng 12/17 cạnh nhân quả thực tế).
*   **SHD (Structural Hamming Distance): 5** (Mức sai số cấu trúc rất thấp so với các phương pháp cùng loại).
*   **Độ ổn định:** Mô hình duy trì hiệu năng cao nhờ khả năng lọc nhiễu sinh học bằng Isolation Forest.

### Bảng so sánh hiệu năng

| Thuật toán | Cơ chế Phi tuyến | Độ chính xác (Sachs) | SHD | Tính ổn định |
| :--- | :--- | :--- | :--- | :--- |
| **PC Algorithm** | Yếu | ~50-55% | Cao | Thấp |
| **NOTEARS (Original)** | Trung bình | ~60% | > 8 | Trung bình |
| **CausalFlow (Ours)** | **Rất tốt (NSF)** | **70.6%** | **5** | **Cao** |

---

## 6. Tham khảo (References)

1.  **ANM-MM (amber0309).** [GitHub Repository](https://github.com/amber0309/ANM-MM). (Cơ sở thuật toán ban đầu).
2.  **Zheng, X., et al. (2018).** "DAGs with NO TEARS: Continuous Optimization for Structure Learning." *NeurIPS*.
3.  **Durkan, C., et al. (2019).** "Neural Spline Flows." *NeurIPS*.
4.  **Zhang, K., & Hyvarinen, A. (2009).** "On the Identifiability of the Post-Nonlinear Causal Model." *UAI*.
5.  **Rahimi, A., & Recht, B. (2007).** "Random Features for Large-Scale Kernel Machines." *NeurIPS*.
6.  **Gretton, A., et al. (2007).** "A Kernel Statistical Test of Independence." *NeurIPS*.
7.  **Vaswani, A., et al. (2017).** "Attention Is All You Need." *NeurIPS*.
8.  **Jang, E., et al. (2016).** "Categorical Reparameterization with Gumbel-Softmax." *ICLR*.
9.  **Kingma, D. P., & Welling, M. (2013).** "Auto-Encoding Variational Bayes." *ICLR*.
10. **He, K., et al. (2016).** "Deep Residual Learning for Image Recognition." *CVPR*.
11. **Ba, J. L., et al. (2016).** "Layer Normalization." *arXiv*.
12. **Hendrycks, D., & Gimpel, K. (2016).** "Gaussian Error Linear Units (GELUs)." *arXiv*.
13. **Lim, B., et al. (2021).** "Temporal Fusion Transformers (GRN)." *IJF*.
14. **Loshchilov, I., & Hutter, F. (2017).** "Decoupled Weight Decay Regularization." *ICLR*.
15. **Liu, F. T., et al. (2008).** "Isolation Forest." *ICDM*.
16. **Pedregosa, F., et al. (2011).** "Scikit-learn: Machine Learning in Python." *JMLR*.
17. **Paszke, A., et al. (2019).** "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS*.

## Giấy phép (License)
Dự án được phát hành dưới giấy phép MIT License.
