# CHƯƠNG 2: KIẾN TRÚC VÀ CƠ CHẾ VẬN HÀNH CỦA MÔ HÌNH DEEPANM

Chương này trình bày một cách hệ thống và chi tiết về cấu trúc kỹ thuật, nền tảng toán học và quy trình thực thi của mô hình **DeepANM (Deep Additive Noise Model)**. Đây là một hệ thống khám phá nhân quả (Causal Discovery) mạnh mẽ, được thiết kế để khắc phục những hạn chế của các phương pháp truyền thống trong việc xử lý dữ liệu phi tuyến, nhiễu không đồng nhất (Heterogeneous noise) và sự phức tạp của không gian trạng thái đồ thị có hướng không chu trình (DAG).

## 2.1 Cấu trúc tổng thể của hệ thống đề xuất

Trong lý thuyết nhân quả, bài toán tìm kiếm đồ thị từ dữ liệu quan sát là một bài khó khăn đặc thù do tính chất **NP-Hard** của không gian tìm kiếm. Khi số lượng biến tăng lên, số lượng đồ thị khả thi tăng trưởng theo hàm siêu mũ. Để giải quyết triệt để sự bùng nổ không gian này, mô hình DeepANM triển khai một lộ trình **3 Pha Tương hỗ (3-Phase Synergetic Pipeline)** tích hợp sâu nhiều thuật toán phân tích thống kê phi tham số, mạng học sâu và tối ưu hóa liên tục.

```mermaid
graph TD
    Data["Dữ liệu quan sát đa biến X"] --> Preprocess("Tiền xử lý đa tầng<br/>(Isolation Forest, Quantile)")
    
    subgraph phase1 ["Pha 1: Định hướng Topological"]
        Preprocess --> FastANM["Mô hình Nhân quả Cơ sở"]
        FastANM --> Topo["Bộ phân tích Thứ tự TopoSort"]
    end
    
    subgraph phase2 ["Pha 2: Học cấu trúc liên tục"]
        Topo -->|"Thứ tự ưu tiên (Prior)"| GPPOM["Mạng Neural Lõi Mô phỏng Lực Cơ chế"]
        GPPOM <--> |"Chấm điểm & Phạt cấu trúc chu trình"| Trainer["Bộ Điều phối Tối ưu (ALM)"]
    end
    
    subgraph phase3 ["Pha 3: Tinh lọc nhân quả"]
        Trainer -->|"Đồ thị thô & Lực tác động"| Lasso["Mạng Rây Adaptive LASSO & Kiểm định Độc lập"]
    end
    
    Lasso --> Final["Đồ thị nhân quả cuối cùng (DAG)"]
```
<p align="center"><b>Hình 2.1: Lộ trình 3 pha tổng thể xây dựng đồ thị nhân quả trong hệ thống DeepANM</b></p>

1.  **Hạn chế không gian (Pha 1):** Sử dụng các mô hình cơ sở tốc độ cao và thuật toán sắp xếp cấu trúc lưới để xác định dòng chảy ưu tiên qua các phép kiểm định thống kê.
2.  **Mô hình hóa sâu (Pha 2):** Sử dụng mạng neural đa tầng kết hợp với bộ tối ưu hóa liên tục để rèn dũa trọng số nhân quả mà không vướng điểm mù tính toán rời rạc.
3.  **Tinh chắt nhân quả (Pha 3):** Xây dựng rào chắn tinh lọc đa cấp độ để vô hiệu hóa những liên kết nhiễu mượn danh quan hệ trực tiếp.

---

## 2.2 Pha 1: Định hướng Cơ sở và Tiền xử lý Dữ liệu

Bước đầu tiên là thanh lọc dữ liệu và phác thảo hướng đi an toàn của luồng thông tin, đảm bảo mạng neural phía sau không lãng phí tài nguyên vào các quỹ đạo mâu thuẫn hay vòng lặp vô tận.

### 2.2.1 Chuẩn hóa Đa tầng và Khử điểm dị biệt

Mô hình sử dụng thuật toán cô lập **Isolation Forest** để loại bỏ các điểm dị biệt (outliers) nghiêm trọng sinh ra bởi sai số thiết bị đo đạc vật lý hoặc lỗi nhập liệu. Tiếp theo đó, kỹ thuật **Quantile Transformer** được áp dụng để chủ động biến đổi phân phối biên của từng biến độc lập về dạng Gaussian chuẩn. Sự nắn chỉnh này hỗ trợ tối đa cho các kiểm định độ nhạy dựa trên Kernel, giúp triệt tiêu hoàn toàn sự sai lệch do đơn vị đo đạc khác nhau.

### 2.2.2 Khởi tạo Cấu trúc Nhanh nguyên sinh

Thay vì bắt mạng neural phải bắt đầu quá trình đào tạo hoàn toàn mơ hồ, hệ thống thực hiện khởi chạy một mô hình rừng ngẫu nhiên (Random Forest) rút gọn để lập tức phác thảo khung đồ thị mồi. Hệ thống này sử dụng phép kiểm định tương quan khoảng cách (Distance Correlation) đánh giá trên phần dư để đề xuất một ma trận trọng số thô khởi đầu, làm tiền đề tăng tốc cho bộ sắp xếp. 

### 2.2.3 Định hướng Cấu trúc (Topological Sort)

DeepANM áp dụng chiến lược sắp xếp chìm (Greedy Sink-First). Nút "Sink" (Nút đáy) được định nghĩa là biến nằm ở tận cùng của nhân quả, không gây ra sự thay đổi cho bất kỳ biến nào khác trong quần thể.

```mermaid
graph TD
    subgraph TopoSort ["Quy trình Sắp xếp Khử dần (TopoSort)"]
        direction TB
        X["Tập biến ứng cử viên còn lại"] --> RFF["Xấp xỉ đặc trưng ngẫu nhiên Fourier"]
        RFF --> Sink_Test["Kiểm định Hilbert-Schmidt (HSIC) \n để đo lường độc lập phần dư"]
        Sink_Test --> |"Đạt chuẩn độc lập tối đa"| Sink_Node["Chọn làm Nút Đáy hiện tại"]
        Sink_Node --> Update["Cập nhật ngược thứ tự nhân quả"]
        Update --> |"Loại bỏ biến đã chốt"| X
    end
```
<p align="center"><b>Hình 2.2: Sơ đồ thuật toán Sắp xếp Topological bằng Đặc trưng Ngẫu nhiên</b></p>

Để phát hiện sự độc lập phi tuyến, hạt nhân Hilbert-Schmidt (HSIC) được sử dụng. Tuy nhiên, tính toán HSIC nguyên thủy làm hao phí thời gian với độ phức tạp song phương $O(N^2)$. Bằng định lý Bochner, bài toán được hệ thống hóa thành xấp xỉ đặc trưng Fourier ngẫu nhiên (RFF) đưa độ cực hạn học máy về thời gian tuyến tính $O(N)$, cho phép đồ thị phản hồi ngay lập tức dẫu lượng mẫu tiến vào hàng nghìn.

---

## 2.3 Pha 2: Mô hình hóa Mô đun Sâu (Deep Neural SCM)

Trong pha này, một tổ hợp mạng neural phức hợp được cấu hình để vi phân hóa một bài toán đồ thị rời rạc về vùng không gian tối ưu toán học trơn tru.

### 2.3.1 Kiến trúc Mạng Neural Nhân quả Lõi (Core Neural SCM)

Phân hệ mạng lõi của mô hình không thiết kế để dự báo một điểm đơn nhất. Nó là một cỗ máy gồm 4 luồng xử lý thực hiện riêng biệt các tác vụ sinh học, cơ chế quy luật, tính toán lượng tử và quản lý bất định: 

1.  **Bộ mã hóa cơ chế (Encoder VAE):** Dùng mạng truyền thẳng kết nối với độ chuẩn hóa lớp dữ liệu nhằm dự đoán xác suất tiềm ẩn. Nó đi qua một hệ thống lấy mẫu hàm nhạt dần năng lượng (Gumbel-Softmax Annealing) nhằm xác định điểm dữ liệu thực sự đang chịu tác động từ luồng quy luật ẩn nào.
2.  **Khối Phương trình Cấu trúc:** Sử dụng các mạng nơ-ron có liên kết thặng dư tắt (Skip Connection) để học hàm sinh nhân quả biểu diễn nội hàm $f(X)$. Hệ thống liên kết thặng dư tránh được sự phai nhạt Gradient của mạng truyền dẫn sâu.
3.  **Bộ Giải mã Hậu Phi tuyến (Post-Nonlinear Decoder):** Biến đổi đầu ra khối hàm bằng hàm kích hoạt đơn điệu Softplus, đảm bảo tính khả nghịch ngặt nghèo. Dãy kiến trúc này cho phép chiết xuất dòng nhiễu nguyên thủy thuần túy theo chuẩn đo lường: `Đại diện nhiễu = Hàm hậu phi tuyến - Hàm sinh nhân quả`.
4.  **Hệ Nhiễu Hỗn hợp Đa phân phối:** Thống kê lượng phần dư chưa lý giải được thông qua một phân phối Gaussian đa đỉnh. Mô hình này giúp DeepANM dễ dàng thích ứng với dữ liệu chứa độ bất định cao, loại hình nhiễu méo lệch hoặc nhiễu có tính chất phân cực (heavy-tailed) của kinh tế, lâm sàng.

Nhờ vào kiến trúc hướng đối tượng bóc tách nghiêm ngặt, dòng chảy truyền tiến không bao giờ chồng chéo nhiệm vụ lặp lại nào:

```mermaid
graph TD
    X["Đầu vào Dữ liệu: x (Đã phủ nhãn che - Masked)"]
    
    subgraph MLP_Forward ["Luồng dữ liệu trong Mạng Lõi"]
        direction TB
        
        X --> Step1_Enc["1. Bộ Mã hóa VAE:<br/>Trích xuất đặc trưng -> Xác suất Cụm -> Điểm phạt Phân kỳ"]
        Step1_Enc --> z_soft["Xác suất Cơ chế Z"]
        Step1_Enc --> kl_loss["Phạt phân kỳ"]
        
        X --> Step2_SEM["2. Bộ Giải phương trình Cấu trúc:<br/>Xử lý qua mạng dư tắt để tìm Lực lượng Trung bình"]
        Step2_SEM --> mu["Lực can thiệp f(X)"]
        
        X --> Step3_Dec["3. Bộ Giải mã Đơn điệu:<br/>Đẩy số liệu qua hàm khả nghịch Softplus"]
        Step3_Dec --> g_x["Kết xuất biên g(X)"]
        
        g_x --> CalcProxy["Tính toán độ biến dạng Nhiễu thuần túy:<br/>Phần dư = g(X) - f(X)"]
        mu --> CalcProxy
        
        CalcProxy --> Step4_Noise["4. Mô hình Nhiễu Hỗn hợp:<br/>Tính toán log hợp lý thống kê dị biệt của phần dư"]
        Step4_Noise --> log_prob_noise["Điểm log-likelihood"]
        
        z_soft --> Output[/"Trả về Khối từ điển Tổng phân tích:<br/>{Nhãn cơ chế, Phạt phân kỳ, Lực can thiệp, Chuẩn nhiễu}"/]
        kl_loss --> Output
        mu --> Output
        log_prob_noise --> Output
    end
```
<p align="center"><b>Hình 2.3: Mô phỏng logic dòng dữ liệu truyền tiến và sự phân giải trách nhiệm chiết tách hàm nhân quả f(X) và hệ nhiễu</b></p>

### 2.3.2 Kiến trúc Điều phối Song song Tổng quát

Mục đích thiết kế trên cùng là nhằm thiết lập song song đầu dự báo để cực đại hóa năng lực chống nhiễu loạn của Gradient.

```mermaid
graph TD
    W_logits["Trọng số Cạnh thô chưa quy chuẩn"] --> Gate["Cổng Gumbel-Sigmoid (Đạo hàm xuyên thấu)"]
    Gate -->|"Tạo vòm ánh xạ cạnh nhị phân"| W_mask["Ma trận Màng lọc Cạnh"]
    X["Dữ liệu quan sát gốc"] --> Masking{"Sàng lọc và nhân ma trận với Màng lọc Cạnh"}
    
    Masking --> MLP["Mạng lõi (Khai thác cơ chế ẩn và Quy luật SCM)"]
    
    subgraph SongSong ["Đầu dự báo phụ trợ theo Tiến trình Gaussian"]
        MLP -.-> |Nhãn cơ chế| GP_Z["Ánh xạ đặc trưng cơ chế học được"]
        Masking --> GP_X["Ánh xạ đặc trưng không gian biến X"]
        GP_Z --> Combine{"Tích chập Hadamard"}
        GP_X --> Combine
        Combine --> Linear["Đầu ra Hàm tuyến tính phụ trợ"]
    end
    
    MLP --> |Gía trị cốt lõi f(X)| Sum{"Tổng hợp Dự đoán Cường độ"}
    Linear --> Sum
    Sum --> Y_pred["Dự đoán Điểm dữ liệu Cuối cùng"]
```
<p align="center"><b>Hình 2.4: Kiến trúc Điều phối Đồng thời tích hợp khai thác quy luật ẩn quy mô rộng và tiến trình nội suy cục bộ.</b></p>

Công nghệ làm thay đổi cục diện thực sự của toàn mạng lưới nằm ở **Cổng Gumbel-Sigmoid sử dụng phương pháp Đạo hàm Xuyên thấu (Straight-Through Estimator)**. Xét bản chất một đồ thị phải tuân thủ dạng nhị phân rạch ròi. Ở hướng truyền tiến quy luật, Cổng Gumbel dùng hàm cắt tầng đột ngột để áp đặt cạnh là số thực 0 hoặc 1. Thế nhưng, tại pha cập nhật sai số đạo hàm ngược, quá trình Gradient sẽ lẩn tránh hàm đột biến và luồn qua cung hàm Sigmoid êm ái, bẻ cong không gian tối ưu để mô hình hoàn thành quá trình đào tạo ma trận không rời rạc hóa.

### 2.3.3 Tối ưu hóa đa mục tiêu với Cơ chế Phạt Lagrangian 

Vì tính phức tạp và cấu hình phân rã rất kỹ của sơ đồ, mô hình nhận về tận 7 mục tiêu đồng tối ưu nghịch lý khốc liệt:
1.  **MSE Gợi lại:** Tái lập khả năng mô phỏng mẫu gốc sát sao nhất.
2.  **Khả năng Hợp lý Âm:** Cực đại hóa xác suất khả dĩ của quần thể nhiễu hỗn hợp phi đồng đều.
3.  **Ràng buộc Tính Độc lập Cụm Sinh cơ chế:** Ép quần thể không gian ẩn phản ánh đúng sự độc lập ngẫu nhiên nguyên thủy khỏi tác nhân can thiệp đầu vào.
4.  **Ràng buộc Phân tích Phần dư:** Đây là trục xoay chuẩn ANM học máy – Ép biến nhiễu sau khai phá độc lập hoàn toàn tuyệt đối với mẫu gốc.
5.  **Dung hòa Cân bằng Phân phối Gumbel:** Khắc phục tính bão hòa năng lượng của vòm mã hóa.
6.  **Sự Thưa thớt Quy chuẩn L1/L2:** Phạt tỷ lệ đồ thị dày đặc vô lý và buộc chúng sinh ra lưới cấu trúc tinh nhuệ thưa thớt.
7.  **Định mức Cấm Vòng Lặp:** Hình thức định thức Logarit đặc biệt chống lại việc cạnh chắp vá quay ngược tạo nên hiệu ứng trứng sinh gà.

Ma trận nghịch lý này đòi hỏi một máy điều nhịp điều hành, gọi là **Bộ điều phối Tối ưu Augmented Lagrangian Method (ALM)**.

```mermaid
sequenceDiagram
    participant T as Bộ Điều phối ALM
    participant O as Tối ưu hóa Bước AdamW
    participant M as Mạng Nhân quả Neural
    
    T->>M: Cung cấp Batch Dữ liệu mẫu, Hệ số kích hoạt mềm
    M->>T: Tính tổn thất (Tổn thất 7 hàm số gốc + Hình phạt cấu trúc DAG)
    T->>O: Tính Gradient đa hướng và Áp sát giá trị cực hạn (Chống Gradient nổ)
    O->>M: Lan truyền ngược cập nhật Trọng số liên tục
    
    opt Mỗi chu kỳ theo quy mô Epoch định sẵn
        T->>M: Gọi đánh giá độ vi phạm tính Phi chu trình
        alt Số vòng lặp không chịu suy giảm
            T->>T: Tác động hình phạt Cứng mạnh gấp đôi để triệt tiêu ngoan cố
        else Đồ thị dần về cấu trúc phân tầng h(W) cực nhỏ
            T->>T: Tích lũy điều chỉnh hệ số giãn xả theo chuỗi Lagrange
        end
    end
```
<p align="center"><b>Hình 2.5: Trình tự kiểm soát năng lượng tối ưu hóa ALM, liên tục kiểm tra và điều khiển xu hướng phá hoại chu trình</b></p>

Khâu kiểm soát ALM như một người cai ngục độc lập với mạng neural, chỉ theo dõi sự vi phạm chỉ số hướng tạo chu kỳ. Không có hệ thống ALM này, mô hình sẽ thỏa hiệp độ đo dự đoán bằng cách nối chằng chịt hệ thống vòng lặp tự ngụy biện.

---

## 2.4 Pha 3: Tinh lọc nhân quả Vững chắc

Với một đồ thị được rèn giũa từ một tổ hợp hàm mạng neural, rỏ rỉ cạnh giả và sự nhận lầm cạnh trung gian luôn là tác dụng phụ. Xuyên suốt Pha 3, hệ thống sử dụng một mạng rây thanh trừng (Adaptive LASSO) cực kỳ khốc liệt dựa trên ba tấm màng độc lập.

```mermaid
graph TD
    Raw["Cạnh ứng cử viên từ Mô hình Neural Sinh ra"]
    
    subgraph Gate1 ["Màng 1: Rào chắn Lực Can thiệp (Jacobian ATE)"]
        Raw --> Jacobian["Khai triển đạo hàm hàm bộ giải mã truy hồi mức độ thay đổi cục bộ"]
        Jacobian --> Cond1{"Tác động Đủ ngưỡng?"}
    end
    
    subgraph Gate2 ["Màng 2: Thử thách Sụp đổ Rừng Ngẫu nhiên"]
        Cond1 --> |Đủ ngưỡng Khởi động| RF["Hoán vị xáo trộn ngẫu nhiên toàn bộ trục nguyên nhân"]
        RF --> Cond2{"Sụt giảm khả năng dự đoán quá tỷ lệ an toàn quy định?"}
    end
    
    subgraph Gate3 ["Màng 3: Hậu kiểm Độc lập Cầu nối"]
        Cond2 --> |Quyết định Đóng góp Đủ| HistGBM["Tính chiết phần dư phụ thuộc qua mô hình học máy tăng cường biểu đồ"]
        HistGBM --> Pearson["Đối xạ phân tích tương quan Độc lập mức cao chót"]
        Pearson --> Cond3{"Phủ nhận hoàn toàn các liên kết mang tính giao hòa gián tiếp?"}
    end
    
    Cond3 --> |Hoàn toàn Thuyết phục| Final["Ấn định Cạnh Nhân quả Tuyệt đối"]
    
    Cond1 -.-> |Ngụy cạnh toán học tĩnh| Drop["Thải loại khỏi mạng nhân quả"]
    Cond2 -.-> |Tính thống kê ảo| Drop
    Cond3 -.-> |Liên kết Mượn đường (A -> B -> C)| Drop
```
<p align="center"><b>Hình 2.6: Cỗ máy Mạng rây đa cổng loại bỏ triệt để các rủi ro tính tương quan giả và hệ quả bắc cầu.</b></p>

1.  **Cường độ Causal Jacobian (Màng 1):** Cạnh tìm được không chỉ để có mà phải gây rúng động đến biến đích. Bằng phép tính vi phân xuyên suốt chức năng của bộ tiền xử lý phi tuyến, hệ thống trả về chỉ số Tác động Cố ý (Average Treatment Effect - ATE). Chỉ cạnh sở hữu điểm lực tác động vật lý dương thực tế mới được truyền sang vòm sau.
2.  **Sức chống sốc Tác động (Màng 2):** Khối kiểm định sử dụng mô hình học máy thứ ba bám sát đánh sập thử trật tự cấu trúc biến nguyên nhân đang xét. Biến nào bị xáo trộn mà đồ thị không xi nhê sụp đổ giá trị đo R-Squared, biến đó không mang trong mình giá trị nguyên nhân tiên quyết.
3.  **Cắt đứt Cầu nối Mượn đường (Màng 3):** Trong khoa học tương quan học, hệ quả truyền nhiễm qua một trạm trung gian thường gây đánh lừa trực giác sinh cạnh lấn cấn. Áp dụng cỗ máy hồi quy tăng cường đồ thị rất tiên tiến, hệ thống huấn luyện tách rời trung gian, thu thập chiết xuất dồn và nếu hệ quả kiểm nghiệm độc lập cho ra giá trị an toàn, mầm mống bắt cầu ngụy biện đã được gỡ bỏ tận chân răng.

## 2.5 Tiểu kết chương

Nội dung hệ thống học thuyết này đã vén tấm màn đen giải mật quy trình DeepANM thiết kế, xây dựng và củng cố một công trình tìm kiếm nhân quả đáng tin cậy. Ba tầng giải quyết – từ dẫn lối cấu trúc sơ khai, chuyển đối môi trường tối ưu toán học trơn mịn đa mục tiêu, đến tinh lọc tàn nhẫn và xác quyết hệ quả thực tế liên đới. DeepANM không xây dựng một khối mô hình vô danh phỏng đoán, mà chia tách rõ ràng đặc điểm nhiễu, cơ chế quy luật, cho ra một nền tảng khám phá sự vận hành cốt lõi phi tuyến của vạn vật.
