# ĐỀ CƯƠNG BÁO CÁO TỐT NGHIỆP
## Đề tài: Khám phá Cấu trúc Nhân quả Phi tuyến bằng Mô hình Nhiễu Cộng Sâu (DeepANM)

Sinh viên thực hiện: Manh Thai
Năm học: 2024 - 2025

---

## MỤC LỤC LÝ THUYẾT & THỰC HÀNH (DỰ KIẾN)

### CHƯƠNG 1: TỔNG QUAN VỀ KHÁM PHÁ NHÂN QUẢ (INTRODUCTION)
1.1. Bối cảnh và Đặt vấn đề
  - Sự khác biệt giữa Tương quan (Correlation) và Nhân quả (Causation)
  - Tại sao cần học máy để khám phá nhân quả?
1.2. Mục tiêu nghiên cứu
  - Khắc phục giới hạn của PC, GES, LiNGAM.
  - Xử lý dữ liệu phi tuyến và nhiễu phức tạp.
1.3. Đóng góp của đề tài (DeepANM)
1.4. Cấu trúc của báo cáo

### CHƯƠNG 2: CƠ SỞ LÝ THUYẾT (THEORETICAL BACKGROUND)
2.1. Đồ thị có hướng không chu trình (DAG) & Phương trình cấu trúc (SEM)
2.2. Mô hình Nhiễu cộng (Additive Noise Models - ANM)
  - Tính bất đối xứng nhân quả trong ANM.
2.3. Tiêu chuẩn Độc lập Hilbert-Schmidt (HSIC)
  - Ước lượng HSIC bằng Random Fourier Features (RFF).
2.4. Phương pháp DAGMA & Học đồ thị liên tục (Continuous DAG optimization)
2.5. Lựa chọn đặc trưng với Adaptive LASSO & Random Forest

### CHƯƠNG 3: MÔ HÌNH DEEP ADDITIVE NOISE MODEL (DeepANM)
3.1. Kiến trúc tổng thể của hệ thống DeepANM
3.2. Pha 1: Sắp xếp Topological bằng HSIC (TopoSort)
  - Thuật toán Greedy Sink-First.
3.3. Pha 2: Khớp tham số SEM bằng Mạng Neural (Neural SCM Fitter)
  - Cấu trúc mạng Encoder - SEM - Decoder.
  - Phân cụm cơ chế (Mechanism clustering) với Gumbel-Softmax.
  - Mô hình hóa nhiễu lai (Heterogeneous GMM Noise).
  - Hàm mất mát đa mục tiêu (Multi-objective Loss) & Phương pháp Augmented Lagrangian.
3.4. Pha 3: Lọc cạnh qua Adaptive LASSO & ATE Gate
  - Giải quyết bài toán False Positive.
  - Double-gate: Random Forest Importance + Neural ATE Jacobian.

### CHƯƠNG 4: KẾT QUẢ THỰC NGHIỆM (EXPERIMENTS & RESULTS)
4.1. Thiết lập thực nghiệm (Setup)
  - Cấu hình phần cứng và siêu tham số.
  - Các metric đánh giá: SHD, Precision, Recall, F1, Accuracy.
4.2. Đánh giá trên dữ liệu sinh học mạng lưới Protein (Sachs Dataset)
  - Kết quả so sánh khi có/không có tri thức miền.
4.3. Nghiên cứu cắt bỏ thành phần (Ablation Study)
  - Đóng góp của Random Forest, CI Pruning và SCM Filter đến SHD.
4.4. Thử nghiệm thăm dò trên dữ liệu kinh tế (Boston Housing)
  - Trực quan hóa và diễn giải đồ thị phân tích được.

### CHƯƠNG 5: KẾT LUẬN & HƯỚNG PHÁT TRIỂN (CONCLUSION & FUTURE WORK)
5.1. Kết luận những điểm đã đạt được
5.2. Hạn chế của mô hình DeepANM
5.3. Hướng phát triển trong tương lai

---
## TÀI LIỆU THAM KHẢO
(Danh sách 19 papers đã tổng hợp trong README)
