# CHƯƠNG 4: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 4.1 Kết luận những kết quả đạt được

Đồ tài nghiên cứu này đã xây dựng thành công và thực nghiệm mô hình **DeepANM**, một phương pháp tiếp cận mới trong việc khám phá cấu trúc nhân quả phi tuyến dựa trên nền tảng mạng neural sâu và các kiểm định thống kê hiện đại. Trải qua quá trình từ nghiên cứu lý thuyết đến triển khai mã nguồn và đánh giá thực nghiệm, đề tài đã đạt được các kết quả trọng tâm sau:

1.  **Xây dựng Kiến trúc 3 Pha Tương hỗ (3-Phase Synergetic Pipeline):**
    *   Đề xuất thành công quy trình phân tách bài toán nhân quả: từ việc thu hẹp không gian tìm kiếm bằng **TopoSort (Pha 1)**, học hàm chuyển đổi phi tuyến qua **Neural SCM (Pha 2)**, đến bước tinh lọc cạnh bằng cơ chế **Double-Gate (Pha 3)**. 
    *   Việc tách biệt này giúp mô hình vượt qua hạn chế "bùng nổ không gian" của các phương pháp tìm kiếm đồ thị rời rạc truyền thống.

2.  **Khả năng xử lý Phi tuyến và Nhiễu phức tạp:**
    *   Tích hợp thành công **HSIC-RFF** và **Random Forest Permutation Importance**, cho phép mô hình phát hiện các quan hệ nhân quả không chỉ dừng lại ở mức độ tuyến tính mà còn bao hàm các tương tác hàm số phức tạp.
    *   Cơ chế **Heterogeneous Noise Model** và **Gumbel-Softmax** giúp mô hình thích nghi tốt với dữ liệu có nhiễu không đồng nhất và sự thay đổi cơ chế (mechanism switching).

3.  **Hiệu năng thực nghiệm khả quan:**
    *   Trên bộ dữ liệu benchmark **Sachs**, mô hình đạt chỉ số **SHD = 16**. Kết quả này cho thấy mô hình có thể hoạt động ổn định và đem lại hiệu năng tương đối tốt khi đối chiếu với một số phương pháp Baseline truyền thống như OLS hay PC.
    *   Trên dữ liệu thực tế **Boston Housing**, mô hình chứng minh được tính ứng dụng thực tiễn khi đưa ra các giá trị **ATE (Average Treatment Effect)** có tính diễn giải cao, hỗ trợ đắc lực cho việc phân tích tác động chính sách và kinh tế.


## 4.2 Hạn chế của mô hình DeepANM

Mặc dù đạt được nhiều kết quả tích cực, mô hình vẫn tồn tại một số hạn chế cần được xem xét:

1.  **Sự phụ thuộc vào Pha 1 (TopoSort):** Hiện tại, nếu Pha 1 xác định sai thứ tự topological của các biến chủ chốt, các pha sau sẽ không thể tìm thấy các cạnh đi ngược lại thứ tự đó. Điều này đòi hỏi chỉ số HSIC phải đạt độ nhạy cực cao và dữ liệu cần được tiền xử lý kỹ lưỡng.
2.  **Thời gian tính toán:** Việc huấn luyện mô hình (đặc biệt là bước Bootstrapping để lựa chọn ổn định) tiêu tốn nhiều tài nguyên CPU/GPU và thời gian hơn so với các thuật toán tham lam truyền thống.
3.  **Bài toán biến ẩn (Unobserved Confounders):** Giống như hầu hết các mô hình ANM hiện nay, DeepANM hoạt động tốt nhất dưới giả định không có biến ẩn gây nhiễu. Trong thực tế, sự tồn tại của các biến không quan sát được có thể làm sai lệch kết quả ATE.

## 4.3 Hướng phát triển trong tương lai

Dựa trên những kết quả và hạn chế đã nêu, nghiên cứu có thể được mở rộng theo các hướng sau:

1.  **Dữ liệu Chuỗi thời gian (Time-series Causal Discovery):** Nâng cấp kiến trúc DeepANM để xử lý dữ liệu có yếu tố thời gian bằng cách tích hợp các mạng RNN/LSTM hoặc Transformer, cho phép phát hiện trễ nhân quả (causal lags).
2.  **Xử lý Biến ẩn bằng Variational Inference:** Mở rộng khối Latent Z hiện tại để thực hiện ước lượng các biến gây nhiễu không quan sát được, giúp mô hình tiệm cận hơn với các kịch bản dữ liệu thực tế phức tạp.
3.  **Tối ưu hóa Tốc độ huấn luyện:** Áp dụng các kỹ thuật nén mạng neural và song song hóa quy trình Bootstrapping trên hệ thống điện toán đám mây để tăng tốc thời gian xử lý cho các tập dữ liệu có hàng trăm biến.

