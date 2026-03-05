# PHẦN MỞ ĐẦU

## 1. Lý do chọn đề tài

Trong bối cảnh bùng nổ của Khoa học dữ liệu (Data Science) và Học máy (Machine Learning), các mô hình Học sâu (Deep Learning) ngày một thể hiện năng lực cao trong việc giải quyết những bài toán phức tạp từ nhận dạng hình ảnh, xử lý ngôn ngữ tự nhiên cho đến dự báo xu hướng. Hầu hết sự thành công này được xây dựng trên năng lực nắm bắt sự **Tương quan (Correlation)** tốt của mô hình học sâu ẩn dưới các tập dữ liệu khổng lồ. 

Tuy nhiên, trong các bài toán đưa ra quyết định thực tiễn (Ví dụ: Kê đơn thuốc, Lập kế hoạch kinh tế, Tối ưu hóa dây chuyền sản xuất), sự tương quan thống kê không phải lúc nào cũng mang lại giá trị định hướng. Cổ ngữ Toán học có câu "Tương quan không có nghĩa là Nhân quả" (Correlation does not imply Causation). Hai sự kiện có thể xảy ra đồng thời do một sự trùng hợp, hoặc do chúng cùng chịu chi phối của một yếu tố ẩn thứ ba (Confounder), chứ không phải kiện này gây ra kiện kia. Việc chỉ dựa vào phân tích tương quan dẫn đến những quyết sách sai lầm, điều này là cốt lõi của **"Khám phá Nhân quả (Causal Discovery)"**.

Tìm kiếm được cấu trúc nguyên nhân - kết quả (Cause-Effect Topology) là chìa khóa để trả lời các câu hỏi về **Sự can thiệp (Intervention)** (*"Điều gì xảy ra nếu thay đổi A?"*) thay vì chỉ tập trung vào **Khả năng quan sát (Observation)** (*"Thấy A thì B thế nào?"*). Các bài kiểm tra đối chứng ngẫu nhiên (RCT - Randomized Controlled Trials) thường là cách duy nhất để kiểm tra tính nhân quả, nhưng chúng thường rất tốn kém, tốn thời gian, hoặc thậm chí vi phạm đạo đức học (như thử nghiệm chất độc lên con người). Do đó, bài toán học được biểu diễn nhân quả từ bộ dữ liệu quan sát tự nhiên đang là mảnh đất giàu tiềm năng nhưng cũng đầy thách thức. Thấu hiểu nhu cầu này, tôi quyết định chọn đề tài **"Khám phá cấu trúc nhân quả phi tuyến bằng mô hình nhiễu cộng sâu (Deep Additive Noise Model - DeepANM)"** nhằm tận dụng sức mạnh linh hoạt của Mạng Neural để hỗ trợ khai phá chiều nhân quả tự động với độ chính xác và tính giải thích cao. 

## 2. Tổng quan lịch sử nghiên cứu của đề tài

### a) Tại thế giới
Bài toán khám phá nhân quả trên thế giới xuất hiện từ khá sớm với các thuật toán dựa trên ràng buộc độc lập như PC Algorithm (Spirtes et al., 2000) hay tính điểm số đánh giá như GES (Chickering, 2002). Một hướng đi cổ điển khác là LiNGAM (Shimizu, 2006) lợi dụng tính không tuần hoàn của đồ thị tuyến tính để tìm chiều tác động.
  * Trong khoảng thập kỷ trở lại đây, nhóm thuật toán **Causal Continuous Optimization** nhận được sự bùng nổ với công trình **NOTEARS** (Zheng et al., 2018), chuyển đổi bài toán tìm đồ thị DAG từ tổ hợp (NP-Hard) sang bài toán tối ưu hóa liên tục. Năm 2022, Bello et al. giới thiệu **DAGMA** với kỹ thuật hàm phạt Log-determinant Barrier ưu việt hơn.
  * Liên quan đến xử lý dữ liệu Phi tuyến (Nonlinear) hoặc Nhiễu đa phương sai (Heteroscedasticity), thế giới đã đề xuất các mô hình học sâu nhiễu cộng (Additive Noise Models - ANM), kỹ thuật tính toán bất đối xứng HSIC (Gretton, 2005; Peters, 2014) và mô hình Nhiễu lai DECI/Causica (Brouillard, 2020). 

### b) Tại Việt Nam và cơ sở giáo dục
Tại Việt Nam, các hướng nghiên cứu về Trí tuệ Nhân tạo (AI) hiện tại phần lớn vẫn xoáy sâu vào phân tích hình ảnh (Computer Vision), Xử lý ngôn ngữ tự nhiên (Văn bản tiếng Việt) tĩnh và các mạng học dự báo. Mảng Mô hình Nhân quả (Causal AI / Causal Machine Learning) còn khá sơ khai. Những đề tài hiện tại áp dụng phân tích nhân quả thường loanh quanh ở góc độ sử dụng Bayes Networks truyền thống cho Y tế lâm sàng hoặc Thống kê kinh tế vĩ mô. Việc thiết kế và phát triển một hệ thống mạng Neural riêng biệt giải quyết bài toán biểu diễn nhân quả liên tục hầu như chưa có sự đào sâu triệt để tại cấp độ học thuật Đại học.

### c) Tính mới và sự nổi bật của đề tài (DeepANM)
Đề tài này đóng góp một quy trình (Pipeline) mới mẻ, gạt bỏ giới hạn của phân loại tương quan và khắc phục tỷ lệ báo lỗi giả (False Positive) rất cao của các thuật toán nhân quả liên tục toàn cục (Global Optimization) trước đây. **Tính mới bao gồm:**
- Phân rã bài toán thành hệ thống 3 pha riêng biệt: Sắp xếp Topological bằng thuật toán HSIC O(N·D), Khớp nhân quả bằng Mô hình Học Sâu kết hợp biến Gumbel-Softmax có cơ chế trộn (Mixed Mechanisms), và Bộ xử lý Lọc cạnh thích nghi. 
- Thay vì để mạng neural quyết định tùy ý các chiều cạnh dẫn đến chu trình nghịch, tính mới nằm ở việc dùng Thuật toán **Sink-First** khóa chặt định hướng trước khi Neural vào việc.
- Thay vì loại bỏ cạnh giả bằng một ngưỡng cứng (ví dụ `<0.3`), DeepANM lọc cạnh bằng Random Forest LASSO và Cổng thích nghi ATE Percentile.

## 3. Mục tiêu đồ án tốt nghiệp

Đề tài hướng tới việc thực hiện các mục tiêu cốt lõi sau:
1. Số hóa hệ thống lý thuyết các kiến thức nền tảng trong suy luận Cấu trúc Nhân quả (Causal Structure Learning), khái niệm can thiệp Do-Calculus và Mô hình Nhiễu cộng (ANM). 
2. Xây dựng, biên dịch và điều chỉnh thuật toán cho một mô hình Học Máy giải quyết tình trạng **Dữ liệu phân bố phi tuyến (Nonlinear)** và **Nhiễu không đồng nhất (Heteroscedasticity, Multimodal)**.
3. Liên kết được hệ thống Mạng Neural Tự động Gi mã VAE cùng Tối ưu hóa Ràng buộc Không Chu Trình (Acyclicity Constraint) thành một mã nguồn duy nhất. 
4. Cải thiện đáng kể độ đo khoảng cách cấu trúc (Structural Hamming Distance - SHD) so với các giải pháp truyền thống trên các tập dữ liệu thực tế (Boston Housing, Sinh học Tế bào Sachs).

## 4. Đối tượng và phạm vi nghiên cứu

**Đối tượng nghiên cứu:**
- Hệ thống Mạng Neural Nhân tạo (Feed-forward Neural Networks, VAE).
- Thuật toán Máy học phục vụ khám phá thông số cấu trúc (Causal Discovery Structure Learning Models). Cụ thể là trường phái Additive Noise Models (Nhiễu Cộng).

**Phạm vi nghiên cứu:**
- **Về lý thuyết học máy:** Tập trung khai thác mạng Perceptron Nhiều Lớp (MLP), cơ chế suy luận Gumbel-softmax để phân cụm, Hàm phạt tối ưu liên tục ALM, và phương pháp tính ước lượng Độc lập Thống kê bằng Toán tử Kernel.
- **Về dữ liệu áp dụng:** Hệ thống DeepANM trong giới hạn của đề tài chỉ xử lý bộ dữ liệu dạng bảng quan sát đa biến liên tục và chuẩn hóa (Continuous Multivariate Tabular Data). Đề tài sẽ được thử nghiệm trên bộ Benchmark Y - Sinh học có Ground-truth thực tế (Sachs, 2005) và bộ điểm chuẩn Kinh tế Boston Housing.
- Đề tài **không** bao quát dữ liệu xử lý chuỗi thời gian (Time-series / Temporal Causality), cũng chưa xét trường hợp khuyết đặc trưng chứa yếu tố Confounders không quan sát được (Latent Unobserved Confounders). 

## 5. Phương pháp nghiên cứu

Để thực hiện đồ án, em đã vận dụng kết hợp các phương pháp nghiên cứu sau:
- **Nghiên cứu tài liệu (Literature Review):** Đọc dịch và phân tích các bài báo học thuật mới nhất trên Science, IEEE, NeurIPS (2005 - 2023) về chủ đề DAG Liên Tục (NOTEARS, DAGMA), Mô hình Nhiễu lai (DECI), Tiêu chuẩn Tính độc lập HSIC. 
- **Thiết kế Cấu trúc Trạng thái (Architectural Modeling):** Vẽ đồ thị thuật toán vòng làm việc của các module mạng Neural thành một hệ thống mã chuẩn khép kín.
- **Lập trình và Thực nghiệm (Implementation & Empirical Study):** Cài đặt thuật toán bằng ngôn ngữ Python với framework PyTorch (Tối ưu hóa Autograd của Tensor GPU). Đóng gói code thành thư viện theo chuẩn Clean Code (có Script CI test liên tục). 
- **Phương pháp So sánh và Đánh giá (Evaluation Assessment):** Quan sát và ghi nhận các thông số kỹ thuật, chạy phương pháp cắt bỏ phần bổ trợ (Ablation Study) nhằm đối chiếu tác dụng của từng module lên kết quả tổng khi so với biểu đồ Ground-truth có sẵn.  

## 6. Đóng góp mới của đề tài và những vấn đề chưa thực hiện được

### 6.1 Những đóng góp thiết thực
1. **Lọc nhiễu tự động hóa với hiệu năng cao (Adaptive ATE Gate):** Xây dựng Cổng ngưỡng Neural tỷ lệ thuận 15% (Percentile) làm rơi cạnh nhân quả thay vì phải thử-sai thủ công ngưỡng siêu tham số (Hyperparameter Threshold). 
2. Thực thi xấp xỉ bộ đánh giá Loss `FastHSIC` qua phương pháp Đặc trưng Fourier ngẫu nhiên, hạ tầng phức tạp tính toán từ $O(N^2)$ xuống chỉ còn $O(N \cdot D)$ (với N là số lượng mẫu, D là số biến).
3. Biên tập và tích hợp thêm kỹ thuật Cắt tỉa Độc Lập Có điều kiện Phân cực một phần (Partial Correlation CI Pruning) sử dụng Cây quyết định HistGradientBoosting — Giúp nhận diện chính xác các cạnh trực tiếp, loại bỏ các đường truyền gián tiếp (Direct Path Matching).

### 6.2 Những vấn đề chưa thực hiện được 
1. Mất mát khả năng khám phá nguyên nhân khi dữ liệu chịu ảnh hưởng bởi biến ngoại lai ẩn (Latent Variable / Missing data) chưa được giám sát. 
2. Thời gian huấn luyện mạng Deep Neural SCM khá tốn kém nếu dữ liệu lớn với hơn hàng ngàn mốc (Node dimension), do mạng MLP phải rẽ nhánh độc lập cho mỗi biến một kiến trúc phân kỳ riêng biệt. 
3. Chỉ biểu thị được tác động nhân quả qua một ma trận số nguyên thủy dạng vô hướng (ATE Jacobian Float), chưa phản ánh được cơ chế phức tạp dạng hàm truyền hàm (Transfer functions) có chu kỳ không đồng nhất.

## 7. Kết cấu của đề tài

Với nội dung và mục tiêu trên, báo cáo được tổ chức thành 5 chương chính:
- **Phần mở đầu:** Lý do chọn đề tài, mục tiêu, đối tượng, phương pháp và khung tóm tắt đề tài. 
- **Chương 1 – Cơ sở lý thuyết:** Trình bày nền tảng cốt lõi về đồ thị nhân quả DAG, mô hình phương trình cấu trúc (SEM), nguyên lý tính bất đối xứng mô hình Nhiễu ANM, toán tử Hilbert-Schmidt và hàm phạt tối ưu phi chu trình liên tục. 
- **Chương 2 – Mô hình Deep Additive Noise Model (DeepANM):** Lý giải hệ thống Pipeline 3 pha do tác giả tích hợp, mô tả phân rã kỹ thuật từ sắp xếp Topological, quy luật huấn luyện MLP kết hợp Gumbel-Softmax Noise đến phương thức Adaptive LASSO lọc cạnh ngõ ra. 
- **Chương 3 – Thử nghiệm và Đánh giá (Experimental Results):** Giới thiệu các bộ dữ liệu dùng để benchmark, so sánh tham số SHD/Accuracy với cơ sở truyền thống. Trình bày Ablation Study kiểm duyệt sức chịu tải của từng Modun thiết kế. 
- **Chương 4 – Kết luận:** Đúc kết lại sức mạnh của hệ thống DeepANM so với mục tiêu đề ra và mở rộng hướng giải đáp các điểm yếu của thuật toán trong tương lai. 
