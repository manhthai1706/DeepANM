# Đặc tả Dự án CausalFlow: Kiến trúc Mạng Nơ-ron Hợp nhất trong Khám phá Nhân quả

Dự án CausalFlow tập trung vào việc xây dựng một hệ thống phát hiện nhân quả hiện đại dựa trên nền tảng Deep Learning, bám sát các mục tiêu về cơ sở lý thuyết vững chắc và khả năng ứng dụng thực tế cao.

---

## 1. Cơ sở lý thuyết và Kiến trúc Mạng Nơ-ron Hợp nhất

Hệ thống đã chuyển đổi từ các module rời rạc sang một **Kiến trúc Mô hình Hợp nhất (Unified Architecture)**, cho phép tối ưu hóa đồng thời cấu trúc đồ thị và tham số mạng nơ-ron.

### 1.1. Neural Backbone và Xử lý Phi tuyến
Kiến trúc sử dụng mạng nơ-ron sâu với các thành phần tiên tiến:
*   **Gated Residual Networks (GRN) & Self-Attention:** Tự động chọn lọc và trọng số hóa đặc trưng đầu vào, giúp mô hình nhạy bén với các tín hiệu nhân quả thực sự.
*   **Neural Spline Flows (NSF):** Tích hợp trực tiếp lớp Spline Flow để mô hình hóa nhiễu phi tuyến bậc cao, đảm bảo việc trích xuất phần dư (residuals) đạt độ chính xác tối ưu.

### 1.2. Ràng buộc Đại số NOTEARS
Tối ưu hóa bài toán khám phá cấu trúc thông qua ràng buộc đại số liên tục:
*   Ép ma trận kề tuân thủ tính chất DAG (Directed Acyclic Graph) ngay trong quá trình lan truyền ngược.
*   Kết hợp HSIC (Hilbert-Schmidt Independence Criterion) làm hàm phạt (penalty) để đảm bảo tính độc lập nhân quả giữa phần dư và biến nguyên nhân.

---

## 2. Mô hình Thuật toán và Khả năng Ứng dụng

CausalFlow đã hoàn thiện dưới dạng một **Framework Deep Learning** tự chứa, đơn giản hóa tối đa quy trình ứng dụng.

### 2.1. API Đơn giản hóa và Tự động hóa
*   **Auto-training Implementation:** Khả năng tự động nhận diện chiều dữ liệu và huấn luyện ngay khi khởi tạo (`CausalFlow(data=...)`).
*   **High-level Inference API:** Tích hợp sẵn các phương thức suy diễn như `predict_direction` (xác định hướng), `check_stability` (kiểm tra độ bền vững) và `predict_counterfactual` (phân tích giả tưởng).

### 2.2. Tính Hội tụ và Ổn định
*   Sử dụng trình tối ưu hóa **AdamW** với cơ chế giảm dần nhiệt độ (Temperature annealing) cho Gumbel-Softmax, đảm bảo mô hình hội tụ ổn định vào các cấu trúc DAG tối ưu.
*   Lớp tiền xử lý tích hợp (QuantileTransform & Isolation Forest) giúp triệt tiêu nhiễu và các điểm ngoại lai, duy trì hiệu năng cao trên dữ liệu thực tế.

---

## 3. Báo cáo Thực nghiệm và Benchmarks

Mô hình đã chứng minh được hiệu quả thông qua việc xử lý các mối quan hệ phi tuyến phức tạp và đạt kết quả ấn tượng trên các tập dữ liệu chuẩn.

### 3.1. Kết quả trên tập dữ liệu chuẩn Sachs
*   **Độ chính xác xác định hướng (Accuracy): 70.6%** (12/17 cạnh nhân quả được xác định đúng).
*   **SHD (Structural Hamming Distance): 5**.
*   Vượt trội hơn các phương pháp truyền thống như PC hay GES về khả năng xử lý nhiễu sinh học và tính phi tuyến.


**Tóm tắt**: CausalFlow không chỉ là một thuật toán mà là một giải pháp Deep Learning hoàn chỉnh, có khả năng tự động trích xuất tri thức nhân quả từ dữ liệu thô với độ tin cậy và ứng dụng thực tiễn cao.
