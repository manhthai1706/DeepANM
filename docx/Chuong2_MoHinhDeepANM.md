# CHƯƠNG 2: KIẾN TRÚC VÀ CƠ CHẾ VẬN HÀNH CỦA MÔ HÌNH DEEPANM

Chương này trình bày một cách hệ thống và chi tiết về cấu trúc kỹ thuật, nền tảng toán học và quy trình thực thi của mô hình **DeepANM (Deep Additive Noise Model)**. Đây là một hệ thống khám phá nhân quả (Causal Discovery) mạnh mẽ, được thiết kế để khắc phục những hạn chế của các phương pháp truyền thống trong việc xử lý dữ liệu phi tuyến, nhiễu không đồng nhất (Heterogeneous noise) và sự phức tạp của không gian trạng thái DAG.

## 2.1 Cấu trúc tổng thể của hệ thống đề xuất

Trong lý thuyết nhân quả, bài toán tìm kiếm đồ thị có hướng không chu trình (DAG) từ dữ liệu quan sát là một bài khó khăn đặc thù do tính chất **NP-Hard** của không gian tìm kiếm. Khi số lượng biến $d$ tăng lên, số lượng đồ thị khả thi tăng trưởng theo hàm siêu mũ. Để giải quyết vấn đề này, DeepANM triển khai một lộ trình **3 Pha Tương hỗ (3-Phase Synergetic Pipeline)** tích hợp sâu nhiều thuật toán học máy.

```mermaid
graph TD
    classDef highlight fill:#000,color:#fff,stroke:#333,stroke-width:2px;
    classDef sub core fill:#bbf,stroke:#333,stroke-width:2px;
    
    Data["Dữ liệu quan sát đa biến X"] --> Preprocess("Tiền xử lý đa tầng<br/>(Isolation Forest, Quantile)")
    
    subgraph phase1 ["Pha 1: Định hướng Topological"]
        Preprocess --> FastANM["Mô hình cơ sở FastANM"]
        FastANM --> Topo["Bộ phân tích TopoSort"]
    end
    
    subgraph phase2 ["Pha 2: Học cấu trúc liên tục"]
        Topo -->|"Thứ tự (Prior)"| GPPOM["Mạng mạng Neural GPPOM-HSIC"]
        GPPOM <--> |"Chấm điểm & Phạt h(W)"| Trainer["Bộ huấn luyện (Trainer / ALM)"]
    end
    
    subgraph phase3 ["Pha 3: Tinh lọc nhân quả"]
        Trainer -->|"Đồ thị thô & ATE"| Lasso["Cắt tỉa Adaptive LASSO & CI"]
    end
    
    Lasso --> Final["Đồ thị nhân quả cuối cùng (DAG)"]
```
<p align="center"><b>Hình 2.1: Lộ trình 3 pha tổng thể trong hệ thống DeepANM</b></p>

1.  **Hạn chế không gian (Pha 1):** Sử dụng các mô hình cơ sở nhanh (`FastANM`) và thuật toán sắp xếp (`TopoSort`) để xác định dòng chảy qua các phép kiểm định thống kê.
2.  **Mô hình hóa sâu (Pha 2):** Sử dụng mạng neural lõi (`GPPOM-HSIC` và `MLP`) kết hợp với bộ tối ưu hóa liên tục (`Trainer` ALM) để rèn dũa trọng số nhân quả.
3.  **Tinh chắt nhân quả (Pha 3):** Sử dụng công cụ cắt tỉa (`Adaptive LASSO`) để vô hiệu hóa những liên kết nhiễu.

---

## 2.2 Pha 1: Định hướng Cơ sở và Tiền xử lý Dữ liệu

Bước đầu tiên là thanh lọc dữ liệu và phác thảo hướng đi an toàn của luồng thông tin, đảm bảo mạng neural phía sau không lạc vào vòng lặp vô tận.

### 2.2.1 Chuẩn hóa Đa tầng và Khử Outliers

Sử dụng **Isolation Forest** để loại bỏ các điểm dị biệt và **Quantile Transformer** để tạo dáng phân phối Gaussian chuẩn, hỗ trợ tối đa cho các kiểm định độ nhạy cao.

### 2.2.2 FastANM: Khởi tạo Cấu trúc Nhanh

Thay vì để mạng mạng neural dò dẫm từ Không (Zero-knowledge), hệ thống có khả năng khởi chạy qua `FastANM` để lấy một bộ khung ban đầu. FastANM dùng Random Forest để dự báo nhanh các cạnh, sử dụng phép kiểm định tương quan khoảng cách (Distance Correlation) trên phần dư để tạo lập ma trận đồ thị mồi. 

### 2.2.3 Định hướng Topological (TopoSort)

DeepANM áp dụng chiến lược sắp xếp chìm (Greedy Sink-First). Nút "Sink" là biến không gây ra bất kỳ biến nào khác.

```mermaid
graph TD
    subgraph TopoSort ["Quy trình TopoSort HSIC"]
        X["Tập biến ứng cử viên X_rest"] --> RFF["Xấp xỉ đặc trưng RFF"]
        RFF --> Sink_Test["Kiểm định HSIC (Độc lập phần dư)"]
        Sink_Test --> |"Độc lập nhất"| Sink_Node["Chọn làm Nút Sink (cuối cùng)"]
        Sink_Node --> Update["Cập nhật ngược thứ tự nhân quả"]
        Update --> |"Loại bỏ nút"| X
    end
```
<p align="center"><b>Hình 2.2: Sơ đồ thuật toán Sắp xếp TopoSort bằng RFF-HSIC</b></p>

Kiểm định HSIC nguyên bản rất chậm ($O(N^2)$). Thông qua đặc trưng Fourier ngẫu nhiên (RFF) dựa trên định lý Bochner, bài toán được xấp xỉ xuống $O(N)$, cho phép mạng hoạt động siêu tốc trên hàng ngàn mẫu dữ liệu.

---

## 2.3 Pha 2: Mô hình hóa lõi Deep Neural SCM (GPPOM-HSIC)

Khối `GPPOM_HSIC` (Gaussian Process Partially Observable Model) và lõi `MLP` đảm nhận việc vi phân hóa một đồ thị nhân quả rời rạc thành một không gian tối ưu toán học trơn tru.

### 2.3.1 Kiến trúc Mạng Neural Nhân quả (MLP)

Module `MLP` không đơn thuần là mô hình Perceptron đa lớp, nó được chia thành 4 phân hệ phụ trách 4 chức năng sinh học thống kê khác nhau của dữ liệu.

```mermaid
graph TD
    In["Đầu vào X_masked"]
    
    subgraph Cluster_Encoder ["1. Bộ mã hóa Cơ chế (Encoder VAE)"]
        In --> E_Linear["Linear + LayerNorm + GELU"]
        E_Linear --> E_Logits["Logits Cơ chế ẩn"]
        E_Logits --> Gumbel["Lấy mẫu Gumbel-Softmax"]
        Gumbel --> Z_soft["Xác suất Cơ chế Z"]
    end
    
    subgraph Neural_SEM ["2. Khối Phương trình Cấu trúc (ANM_SEM)"]
        In --> S_Proj["Input Projection"]
        S_Proj --> Res1["Khối Res-MLP 1 (Skip Connection)"]
        Res1 --> Res2["Khối Res-MLP 2"]
        Res2 --> Mu["Luồng Tác động Trung bình Mu"]
    end
    
    subgraph Decoder_Noise ["3. Bộ Giải mã & Nhiễu Hỗn hợp"]
        Z_soft --> Noise["Mô hình Nhiễu GMM (5 đỉnh)"]
        Mu --> PNL["Bộ Decoder hậu phi tuyến (Softplus)"]
        PNL --> Y_hat["Dự đoán Phục hồi Y_hat"]
    end
```
<p align="center"><b>Hình 2.3: Phân rã cấu trúc mạng MLP. Đảm nhiệm nhận diện cơ chế ẩn (Màu xanh), học lực tác động nhân quả (Màu lục) và phân phối nhiễu phức tạp (Màu đỏ)</b></p>

Cơ chế *Heterogeneous Noise Model* cho phép xử lý dữ liệu với nhiễu đuôi nặng (heavy-tailed) hoặc đa đỉnh, vô cùng thích hợp cho dữ liệu sinh học hay tài chính.

### 2.3.2 Kiến trúc Điều phối Tổng thể (GPPOM-HSIC)

Bên trên khối MLP, GPPOM làm nhiệm vụ tổ chức dòng dữ liệu và bổ sung mạng dự báo hỗ trợ bằng tiến trình Gaussian. Không sử dụng thuật ngữ "lai tạp" (hybrid), chúng tôi định nghĩa đây là **Đầu dự báo song song**.

```mermaid
graph TD
    W_logits["Trọng số thô W_logits"] --> Gate["Cổng Gumbel-Sigmoid (ST Estimator)"]
    Gate -->|"Đạo hàm qua biến nhị phân"| W_mask["Ma trận kề W_mask"]
    X["Dữ liệu quan sát X"] --> Masking{"Sàng lọc X @ W_mask"}
    
    Masking --> MLP["Khối MLP (Phân cụm & SEM)"]
    
    subgraph SongSong ["Đầu dự báo phụ trợ (Gaussian Process)"]
        MLP -.-> |Phân cụm Z| GP_Z["RFF-GP Đặc trưng Z"]
        Masking --> GP_X["RFF-GP Đặc trưng X"]
        GP_Z --> Combine{"Hadamard Product"}
        GP_X --> Combine
        Combine --> Linear["Đầu ra tuyến tính hỗ trợ"]
    end
    
    MLP --> |Gía trị cốt lõi Mu| Sum{"Tổng hợp Dự đoán"}
    Linear --> Sum
    Sum --> Y_pred["Dự đoán cuối cùng"]
```
<p align="center"><b>Hình 2.4: Kiến trúc GPPOM-HSIC. Tích hợp trích xuất đặc trưng cơ chế ẩn và sức mạnh của Cổng Gumbel.</b></p>

Khâu quan trọng tuyệt đối ở đây là **Cổng Gumbel-Sigmoid với Straight-Through (ST) Estimator**. Thay vì phải chọn ngưỡng tĩnh cho đồ thị, cổng Gumbel ép giá trị đồ thị là biến nhị phân $\{0, 1\}$ ở lần truyền tiến (để mạng học các đường đi rõ ràng), nhưng khi truyền đạo hàm ngược, ST bẽ cong không gian giúp đạo hàm đi qua hàm Sigmoid liên tục để cập nhật $W\_{logits}$.

### 2.3.3 Tối ưu hóa đa mục tiêu với Augmented Lagrangian (Trainer)

Mạng neural có 7 mục tiêu xung đột phải thỏa mãn:
1.  **MSE:** Tái cấu trúc giá trị y.
2.  **NLL:** Hàm hợp lý cực đại cho nhiễu GMM.
3.  **HSIC_CLU:** Ép độc lập giữa nguyên nhân ($X$) và biến phân cụm ($Z$).
4.  **HSIC_PNL:** Ràng buộc cốt yếu: Phần dư phải độc lập với Nguyên nhân.
5.  **KL Divergence:** Cân bằng phân phối Gumbel cho kiến trúc VAE.
6.  **L1 / L2 Penalty:** Ép cấu trúc đồ thị trở nên sắc bén (thưa).
7.  **$h(W)$ DAGMA:** Hàm phạt log-determinant chống vi phạm chu trình đồ thị.

Quá trình giữ thăng bằng cho cả 7 hàm số được kiểm soát bởi cơ chế **Augmented Lagrangian Method (ALM)** nằm bên trong File `trainer.py`:

```mermaid
sequenceDiagram
    participant T as DeepANM Trainer
    participant O as AdamW Optimizer
    participant M as GPPOM Model

    T->>M: Cung cấp Batch Dữ liệu, Nhiệt độ Temperature
    M->>T: Tính tổn thất (Mục tiêu cơ bản + Alpha * h(W) + Rho/2 * h(W)^2)
    T->>O: Tính Gradient và Cắt ngọn (Clip <= 5.0)
    O->>M: Lan truyền ngược cập nhật W_logits và tham số
    
    opt Mỗi chu kỳ K epoch
        T->>M: Gọi đánh giá phi chu trình h(W)
        alt h(W) ngoan cố không chịu giảm
            T->>T: Tăng gấp đôi hệ số phạt cứng (Rho = Rho * 2)
        else h(W) có dấu hiệu hội tụ
            T->>T: Tích lũy điều chỉnh biên độ Lagrange (Alpha)
        end
    end
```
<p align="center"><b>Hình 2.5: Luồng xử lý điều phối Gradient và chống bùng nổ của khối Trainer</b></p>

Sự chặt chẽ của Trainer đảm bảo mô hình không bao giờ sinh ra đồ thị lặp, giải quyết xuất sắc điểm yếu truyền thống của máy học cấu trúc.

---

## 2.4 Pha 3: Tinh lọc nhân quả (Adaptive LASSO)

Ngay cả một thuật toán mạnh nhất cũng sẽ dính cạnh nhiễu hoặc cạnh gián tiếp. Module `Adaptive LASSO` là bức tường lửa 3 lớp bảo vệ chân lý cuối cùng.

```mermaid
graph TD
    Raw["Cạnh ứng cử (Pha 2 sinh ra)"]
    
    subgraph Gate1 ["Cửa 1: Thước đo Can thiệp Thực tế"]
        Raw --> Jacobian["Tính Toán Jacobian ATE"]
        Jacobian --> Cond1{"|ATE| > 0.005?"}
    end
    
    subgraph Gate2 ["Cửa 2: Độ nhạy Phi tuyến"]
        Cond1 --> |Mạnh| RF["Xáo trộn Rừng ngẫu nhiên (RF Permutation)"]
        RF --> Cond2{"Sụt giảm > 3%?"}
    end
    
    subgraph Gate3 ["Cửa 3: Cắt cầu Mượn đường (CI)"]
        Cond2 --> |Ý nghĩa| HistGBM["Tính phần dư HistGradientBoosting"]
        HistGBM --> Pearson["Kiểm tra Độc lập Pearson"]
        Pearson --> Cond3{"p-value > 0.01?"}
    end
    
    Cond3 --> |Hoàn toàn Độc lập| Final["Xác nhận Cạnh Nhân quả"]
    
    Cond1 -.-> |Yếu ớt| Drop["Loại bỏ cạnh giả"]
    Cond2 -.-> |Nhiễu loạn| Drop
    Cond3 -.-> |Bắc cầu gián tiếp| Drop
```
<p align="center"><b>Hình 2.6: Mạng rây tĩnh điện Adaptive LASSO loại bỏ toàn bộ các tương quan không phải là nguyên nhân</b></p>

1.  **Cơ chế Jacobian ATE (Cổng 1):** Cạnh tìm được phải mang lực tác động mạnh ($> 0.005$). Nó được tính vi phân xuyên thẳng cấu trúc PNL Decoder để tìm ra mức độ can thiệp thực sự (Treatment Effect).
2.  **Xáo trộn RF (Cổng 2):** Đo sự sụt giảm trên phân phối ngẫu nhiên để miễn nhiễm với các tương quan tuyến tính giả.
3.  **HistGBM CI (Cổng 3):** Những đường link kiểu $A \to B \to C$ thường sinh ảo giác $A \to C$. Thuật toán học máy `Histogram-based Gradient Boosting` sẽ tìm phần dư và loại trừ hiện tượng bắc cầu này.

## 2.5 Tiểu kết chương

Chương này đã làm rõ kiến trúc tinh vi dưới nắp capo của hệ thống DeepANM qua 3 pha xử lý bổ trợ hoàn hảo cho nhau. Việc không sử dụng khái niệm lai tạp mơ hồ mà phân chia rõ tính năng cho phép hệ thống triển khai mạng Neural SCM kết hợp Gumbel Gate giải quyết bài toán Không chu trình khét tiếng. DeepANM thể hiện mình là một mẫu hình Causal Discovery thế hệ mới: Vững chắc về lý thuyết, sâu sắc về biểu diễn và minh bạch trong đào thải giả định.
