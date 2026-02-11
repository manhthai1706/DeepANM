# Đặc tả Dự án CausalFlow: Kiến trúc Mạng Nơ-ron Sâu trong Khám phá Nhân quả

Tài liệu này đặc tả cơ sở lý thuyết, thiết kế kiến trúc và kết quả thực nghiệm của dự án CausalFlow, bám sát các mục tiêu phát triển hệ thống khám phá nhân quả hiện đại.

---

## 1. Cơ sở lý thuyết và Thiết kế Kiến trúc Mạng Nơ-ron mới

Hệ thống xây dựng một **Kiến trúc Mạng Nơ-ron Nhân quả Sâu (Deep Neural Causal Discovery Architecture)**, giải quyết các hạn chế của các phương pháp truyền thống thông qua việc tích hợp các cơ chế xử lý phi tuyến tính và ràng buộc đại số.

### 1.1. Khối Backbone Nơ-ron hiện đại
Kiến trúc mạng không chỉ dừng lại ở các lớp kết nối đầy đủ (Fully Connected) đơn giản mà sử dụng:
*   **Gated Residual Networks (GRN):** Cơ chế cổng giúp mạng tự động chọn lọc các đặc trưng đầu vào quan trọng, loại bỏ nhiễu từ các biến không liên quan.
*   **Self-Attention Mechanism:** Trọng số hóa các mối tương quan biến số để tập trung vào các tín hiệu nhân quả tiềm năng.

### 1.2. Tích hợp Cơ chế Phi tuyến tính (Neural Spline Flows)
Để xử lý các mối quan hệ phức tạp và phân phối nhiễu không chuẩn (non-Gaussian), kiến trúc tích hợp lớp **Neural Spline Flows (NSF)**:
*   Sử dụng các hàm Spline đơn điệu bậc ba có thể học được để mô hình hóa mọi dạng nhiễu phi tuyến.
*   Đảm bảo tính khả nghịch (invertibility) để trích xuất chính xác phần dư (residuals), phục vụ cho việc kiểm tra tính độc lập nhân quả.

### 1.3. Ràng buộc Đại số tối ưu hóa Đồ thị (NOTEARS)
Hệ thống tích hợp trực tiếp ràng buộc đại số vào hàm mất mát của mạng nơ-ron:
*   **Acyclicity Constraint:** Sử dụng hàm $h(W) = Tr(e^{W \circ W}) - d$ để ép ma trận trọng số luôn tuân thủ cấu trúc đồ thị không vòng (DAG).
*   **Differentiable Optimization:** Cho phép tìm kiếm cấu trúc nhân quả trực tiếp thông qua Gradient Descent thay vì tìm kiếm tổ hợp (combinatorial search) tốn kém.

---

## 2. Mô hình Thuật toán và Khả năng Ứng dụng Thực tế

Mô hình được hoàn thiện dưới dạng một **Engine** có khả năng tự động hóa quy trình phân tích và trích xuất tri thức từ dữ liệu thô.

### 2.1. Quy trình Phân tích Tự động
Hệ thống cung cấp giao diện lập trình mạnh mẽ:
*   **ANM-MM (Adaptive Non-linear Model):** Tự động thực hiện các phép thử hướng cho từng cặp biến hoặc toàn bộ đồ thị.
*   **Mechanism Discovery:** Cấu trúc mạng đa đầu ra (Multi-head) giúp nhận diện đồng thời cả hướng nhân quả và các cơ chế tiềm ẩn (latent mechanisms).

### 2.2. Tính Hội tụ và Độ ổn định
*   **AdamW Optimizer + Weight Decay:** Kiểm soát quá trình huấn luyện mạng nơ-ron sâu, đảm bảo hội tụ ổn định và tránh Overfitting.
*   **Adaptive Kernel Bandwidth:** Tự động điều chỉnh tham số hàm nhân trong các phép thử HSIC để duy trì độ nhạy với các cường độ nhân quả khác nhau trong dữ liệu thực tế.

---

## 3. Báo cáo Thực nghiệm và So sánh Hiệu suất

Kết quả kiểm thử trên các tập dữ liệu chuẩn và dữ liệu mô phỏng chứng minh tính hiệu quả vượt trội của kiến trúc CausalFlow.

### 3.1. Thử nghiệm trên dữ liệu chuẩn thực tế (Sachs Dataset)
Bộ dữ liệu Sachs (biểu hiện protein) là một tiêu chuẩn khắt khe cho bài toán nhân quả với các mối quan hệ phi tuyến và nhiễu nặng.
*   **Độ chính xác hướng (Accuracy):** 70.6% (Xác định đúng hướng 12/17 cạnh nhân quả).
*   **SHD (Structural Hamming Distance):** 5 (Chỉ số lỗi cấu trúc thấp, chủ yếu do các cạnh nhạy cảm với hướng).

### 3.2. Thử nghiệm trên dữ liệu mô phỏng (Synthetic Data)
Sử dụng bộ sinh dữ liệu phức tạp (Exponential, Laplace, Sin/Cos mappings):
*   Hệ thống xử lý xuất sắc các mối quan hệ Post-Nonlinear (PNL), nơi các thuật toán truyền thống thường thất bại.
*   **Tính ổn định:** Đạt được kết quả nhất quán trên nhiều cấu hình dữ liệu khác nhau.

### 3.3. So sánh với các thuật toán tham chiếu

| Chỉ số | Thuật toán PC | NOTEARS (Base) | **CausalFlow (Ours)** |
| :--- | :--- | :--- | :--- |
| **Xử lý Phi tuyến** | Kém (Giả định tuyến tính) | Trung bình | **Rất tốt (Sử dụng NSF)** |
| **TPR (True Positive Rate)** | ~55-60% | ~60-65% | **> 70%** |
| **FDR (False Discovery Rate)** | Cao do nhiễu | Trung bình | **Thấp (Nhờ lọc Outliers)** |
| **SHD** | Tùy thuộc tập dữ liệu | Thường > 8 | **~5 (Trên tập Sachs)** |
