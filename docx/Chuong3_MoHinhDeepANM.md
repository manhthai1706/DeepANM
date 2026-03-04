# CHƯƠNG 3: MÔ HÌNH DEEP ADDITIVE NOISE MODEL (DeepANM)

Chương này trình bày chi tiết về kiến trúc, tư tưởng thiết kế và thuật toán vận hành của hệ thống DeepANM (Deep Additive Noise Model) do học viên đề xuất và phát triển. Hệ thống này là sự kết giao chặt chẽ giữa Toán học tối ưu liên tục, Lý thuyết Nhân quả và Trí tuệ Nhân tạo Học Sâu (Deep Learning). Mục tiêu của DeepANM là phá vỡ bức tường hạn chế của dữ liệu nhiễu phi tuyến và cung cấp một lộ trình (pipeline) tự động hóa khép kín từ khâu tiếp nhận tập dữ liệu thô (raw observational data) cho đến định hình chính xác Ma trận cấu trúc Đồ thị Có hướng Không chu trình (DAG).

## 3.1 Kiến trúc Tổng thể (Overall Pipeline Architecture)

Để giải quyết một bài toán vô cùng phức tạp và mang tính rủi ro cao như Tối ưu NP-Hard trên dữ liệu tỷ lệ độ nhiễu cao (High Noise-to-Signal ratio), một Mạng Neural truyền thống (như Multi-layer Perceptron đơn thuần) là không đủ. Việc ép một mạng neural vừa học quan hệ hàm tương quan, vừa phải tự dò đường xem cạnh nào không được phép đi luẩn quẩn (Cyclic) thường xuyên dẫn đến các điểm tối ưu cục bộ (Local Minima tệ hại). Hệ quả là máy tính có thể vẽ ra một đồ thị gần như nối các đỉnh với nhau chằng chịt, hoặc nối ngược quy luật tự nhiên.

Nhận diện được nút thắt này, tôi đã thiết kế kiến trúc của DeepANM được phân rã thành **Hệ thống 3 Pha (3-Phase Pipeline)** chuyên biệt nhằm bảo vệ tính toàn vẹn của đồ thị phân cấp ở từng bước:

```mermaid
graph TD
    classDef highlight fill:#000,color:#fff,stroke:#333,stroke-width:2px;
    classDef sub core fill:#bbf,stroke:#333,stroke-width:2px;
    
    Data[(Dữ liệu quan sát đa biến X)] --> Preprocess(Tiền xử lý & Loại bỏ nhiễu cực trị)
    
    subgraph phase1 ["Phase 1: Tiền định hướng (TopoSort)"]
        Preprocess --> RFF[Phép chiếu Không gian ngẫu nhiên Fourier - RFF]
        RFF --> Sink[HSIC Greedy Sink-First Ordering]
    end
    
    subgraph phase2 ["Phase 2: Khớp Động cơ Sinh thái (Neural SCM Fitter)"]
        Sink -->|Thứ tự Ưu tiên C| Encoder[Mạng VAE Encoder + Gumbel Softmax]
        Sink -->|Đội hình biến| SEM[Mạng Neural Phương trình Nhân quả Cốt lõi]
        Encoder -->|Phân cụm Cơ chế Z| Combine((Trộn Ensemble))
        SEM -->|Dự đoán Trạng thái| Combine
        Combine --> Decoder[Mạng Monotonic Decoder & GMM NLL]
        Decoder --> ALM[Tối ưu hóa Hệ số Vong lặp DAGMA ALM]
    end
    
    subgraph phase3 ["Phase 3: Công cụ Cắt tỉa tinh chỉnh (Adaptive Post-Pruning)"]
        ALM -->|Ma trận Trọng số thô W_raw| DoubleGate(Double-Gate Filter)
        DoubleGate --> RF[Random Forest Permutation Importance]
        DoubleGate --> ATE[Neural ATE Jacobian Estimator]
        RF --> ALasso[Adaptive LASSO Penalties]
        ATE --> ALasso
        ALasso --> CI[Partial Residual Correlation Test]
    end
    
    CI --> Final[DAG Cấu trúc Cuối cùng]
    
    class Sink,ALM,CI highlight;
    class Encoder,SEM,Decoder core;
```

**Tổng quan dòng đời vận hành của 3 pha:**
- **Pha 1 (TopoSort - Sắp xếp Topological):** Đóng vai trò là "Nhà chiến lược". Không vội vàng nhồi trọng số vào học máy sâu, pha này dùng phân tích thống kê toán học (HSIC) để tìm ra một chuỗi (Order) các biến được sắp xếp từ Tổ tiên (Nguyên nhân Gốc) $\to$ ... $\to$ Đóng vai trò con cháu (Sink Nodes). Kết quả định hình một biên giới để chặn đường Mạng Neural không vi phạm nguyên tắc xoay vòng sau này.
- **Pha 2 (Neural SCM - Khớp Mạng Phương trình Cấu trúc):** Là cốt lõi (Trái tim của hệ thống). Một hệ thống VAE - Khớp nhiễu Giao thoa sẽ tính toán sự lây lan tín hiệu hàm ẩn và áp dụng nhân tử Augmented Lagrangian Barrier chặn đường chu kỳ (Acyclicity).
- **Pha 3 (Edge Post-Pruning - Cắt tỉa thông minh):** Vận hành như một "Bác sĩ Ngoại khoa". Thay vì để ngưỡng cố định (VD: trọng số $< 0.3$ thì bỏ cạnh vì coi nó là nhiễu), hệ thống ứng dụng Rừng ngẫu nhiên (Random Forest) và lý thuyết can thiệp Do-calculus (ATE Jacobian matrix) để lọc đi các đường dẫn gián tiếp có tính lừa dối thống kê (False Positives).

Dưới đây là thiết kế chi tiết cấp cao về mặt kỹ thuật cho từng Pha.

---

## 3.2 Pha 1: Sắp xếp Topological & Chuẩn hóa Bất đối xứng (Phase 1)

Thay vì phó mặc hoàn toàn trọng trách phân biệt "Cha - Con" cho Mạng Neural ALM, thực tế việc cung cấp cho ALM một bức tranh gợi ý về độ ưu tiên Topological (Node $A$ gần cội nguồn Nguyên nhân gốc hơn Node $B$) sẽ tăng độ ổn định của Loss Optimization lên đáng kể trên đa hình thể nhiễu.

### 3.2.1 Tiền xử lý Dữ liệu Chuẩn (Quantile Normalization & Isolation Forest)

Dữ liệu tự nhiên thường chứa các giá trị biên cực dải (Extreme Outliers). Chẳng hạn, trong dữ liệu y khoa Protein (Sachs Dataset), đôi khi phép đo phổ bị kẹt tia Laser tạo ra cường độ lệch chuẩn (Z-score $> 5.0$). 

Mạng Neural sử dụng hàm phi lồi (Non-convex) như Sigmoid và L2 Penalty (MSE Loss) cực kỳ nhạy cảm và dễ vỡ gradient bởi các nhiễu lớn bùng nổ này. Hệ thống được trang bị bộ Tiền xử lý hai lớp:

1. **Isolation Forest (Cách ly ngoại lệ):** Khởi chạy một tập các Cây quyết định ngẫu nhiên nhằm chia tách dữ liệu. Nếu một điểm dữ liệu bị tách rời quá nhanh khỏi quần thể, nó bị coi là nhiễu cực đại và loại bỏ để bảo vệ sự ổn định của quá trình huấn luyện.
2. **Quantile Transformer (Chuẩn hóa Phân phối):** Để đảm bảo các thuật toán tối ưu hoạt động mượt mà, hệ thống ép mọi biến số về phân phối Chuẩn (Gaussian) theo hình chuông. Việc này giúp loại bỏ sự chênh lệch về thang đo (Scale) và hình dạng phân phối kỳ dị của dữ liệu thô.

Mục tiêu cuối cùng của bước này là tạo ra một "sàn đấu" bằng phẳng nhất cho mô hình học máy, nơi các tín hiệu nhân quả không bị che lấp bởi các giá trị ngoại lai hay sự sai khác đơn vị quá lớn.

### 3.2.2 Thuật toán Greedy Sink-First Topological Ordering

TopoSort là khâu tìm thứ tự hoán vị $\pi$ của các biến số sao cho nếu $\pi(i) < \pi(j)$ thì $X_{\pi(j)}$ tuyệt đối không thể là Nguyên nhân gây ra $X_{\pi(i)}$. Đây là tính chất của dòng chảy thông tin DAG.
Khác với thuật toán sắp xếp dựa vào rễ (Leaf/Source First), quy trình học máy ANM bộc lộ tính độc lập phần dư cực đoan (Independence of residual signals) khi bóc vỏ biến Kết quả (Sink Nodes).

DeepANM áp dụng thuật toán **Sắp xếp Chìm tham lam (HSIC Greedy Sink-First Ordering)** dựa trên tư tưởng: "Nếu xoa bỏ hết các nguyên nhân, phần còn lại của một kết quả phải hoàn toàn ngẫu nhiên và độc lập".

**Quy trình logic:**
1.  Giả định toàn bộ các biến là tổ tiên, thử nghiệm từng biến một đóng vai trò là "nút đích" (kết quả cuối cùng).
2.  Sử dụng một mô hình học máy nhanh để dự đoán nút đích đó từ các biến còn lại.
3.  Tính toán độ độc lập thống kê (thông qua chỉ số HSIC) giữa sai số dự đoán và các biến cha.
4.  Nút nào có sai số "sạch" nhất (độc lập nhất) sẽ được chọn làm nút đích và đưa vào cuối danh sách.
5.  Loại bỏ nút đó và lặp lại cho đến khi sắp xếp xong toàn bộ hệ thống.

Kết quả của Pha 1 là một **Thứ tự Topological**, đóng vai trò là "kim chỉ nam" ngăn chặn việc mạng nơ-ron tạo ra các vòng lặp nghịch lý trong các bước tiếp theo.

---

## 3.3 Pha 2: Cỗ máy Lõi Nhân quả (Deep Neural SCM Fitter - GPPOMC)

DeepANM đưa ra cơ chế mô hình hóa Phương trình Cấu trúc (Structural Equations) cực kỳ mềm dẻo. Trong tự nhiên thực tế, cơ chế tự tác động gây ra hậu quả (Transfer Functions) giữa 2 biến không tồn tại dưới dạng 1 phương trình tuyến tính chung chung, mà thường bị xẻ nhỏ thành vô số cụm (Clusters) khác nhau. Ví dụ: Cơ chế Thuốc hạ Glucose hoạt động rất yếu trên người Béo phì so với người Bình thường do kháng Insulin (Hiện tượng Nhiễu không đồng nhất - Multimodal Heteroscedasticity).

Để giải quyết, tôi thiết kế và tích hợp lõi GPPOMC kết cấu thành một vòng lặp logic khép kín:

```mermaid
graph TD
    Input[Dữ liệu X] --> Encoder[Encoder: Phân cụm Cơ chế]
    Input --> Mask[Ma trận Trọng số W]
    Mask --> SEM[SEM: Học Phương trình Nhân quả]
    Encoder -->|Z| SEM
    SEM -->|Dự đoán| Decoder[Decoder: Tái cấu trúc & Tính Nhiễu]
    Decoder --> Output[Hàm Loss Đa mục tiêu]
    Output -->|Backprop| Mask
```

### 3.3.1 Kiến trúc các Khối Mạng (Module Architecture)

Sơ đồ mạng nơ-ron được thực hiện với kích thước Batch Tensor $(N_{batch}, d)$. Các lớp (Layer) xử lý của Mạng Neural DeepANM không phải các Lớp Fully-Connected chéo nhau giữa các Biến (Cross-Variable Mixing), mà được thiết kế dạng **Perceptron Cục bộ Song Tử (Local Perceptron for Variable Pairs)**, nhằm không phá vỡ tính Diễn dịch Tác động Nhân quả (Identifiability Causal Graph). Hệ thống có 3 cấu kiện:

1. **Khối SEM Nhân quả Trọng tâm (ANM_SEM Module)** 
   - Đầu vào: Ma trận $X$
   - Vận hành: Đối với mỗi biến $j$, thiết lập một mạng đa tầng (MLP) với Lớp ẩn (Hidden Dim = 32), Tích hợp Activation LeakyReLU phi tuyến tĩnh.
   - Đầu ra kỳ vọng của mạng: $\mu_j$. Lớp này học cách phản chiếu sự thay thế các hàm $f_j(PA(X_j))$ trong đó $PA(X_j)$ được kiểm soát bằng một ma trận mặt nạ $W_{adj}$ đè lên trước lúc feed forward:
     $$X_{\text{masked}} = X \odot \sigma(\alpha \cdot W_{\text{logits}})$$
   - Đảm bảo một khi trọng số Cạnh $W_{ij} \to 0$, $X_i$ vĩnh viễn không được đưa vào Lấy mẫu trọng số cho hàm dự đoán $\hat{f_j}$.

2. **Khối Phát hiện Cơ chế (Encoder VAE & Gumbel-Softmax Trick)**
   - Cung cấp mô hình khả năng quyết định xem một dòng dữ liệu của bệnh nhân đang chịu chi phối bởi cụm cơ chế nhân quả số $k = 1$ hay $k = 2$ (Mặc định `n_clusters = 2` cho các trạng thái đối lập như Kích hoạt / Bất hoạt sinh hóa học).
   - Encoder tiếp nhận $X$ đưa qua Linear + ReLU, trả lại ma trận log-Xác suất rời rạc $Logits = \log p(z | x)$ để xếp cụm.
   - **Gumbel-Softmax Trick (Jang et al., 2016):** Mạng Neural thông thường không thể tính đạo hàm xuyên qua quá trình chọn lọc rời rạc bằng hàm Argmax $\arg\max_k (prob)$. Để Backpropagate luồng Error cho việc chọn Cụm $z \in \{1, \dots, K\}$, kiến trúc cộng thêm biến nhiễu Gumbel(0, 1) $g_k$:
     $$z_k = \frac{\exp((\log p_k + g_k)/\tau)}{\sum_{j=1}^K \exp((\log p_j + g_j)/\tau)}$$
     Tham số nhiệt độ Nhiễu Softmax $\tau \to 0$ sẽ ép vector xác suất thành định hướng One-Hot, nhưng vẫn khả vi để trượt Gradient xuống $f_j$.

3. **Khối Đọc Nhiễu Biến Hóa (Monotonic Decoder YH)**
   - Hàm dự báo $\mu$ của SEM dựa theo Gumbel cụm số $k$ sẽ được Decoder biến đổi Monotonic (đồng biến tăng) để giả lập sự méo mó không tuần hoàn của hàm rải rác:
     $Y_{trans} = \text{Decoder}(\mu_z)$
   - Sau đó tính toán phân phối log-likelihood của phần dư nhiễu bằng thuật toán phân bổ hỗn hợp Gaussian hỗn hợp theo phân hóa cụm DECI (Heterogeneous Noise Component).

### 3.3.2 Hàm Mất Mát Đa Mục Tiêu (Objective Loss Function)

Với hàng chục ngàn trọng số Param thiết kế, GPPOMC định hướng Back-propagation thông qua hàm Loss đồ sộ hòa trộn 4 chỉ số cực tiểu hóa. Đặt $\Theta$ là toàn tập Parameters của Neural Networks và $W$ là Ma trận kề Causal Graph DAG logit. 

Hàm Loss được thiết kế để cân bằng giữa sự chính xác của mô hình và tính hợp lệ của đồ thị:
1. **Lỗi Tái cấu trúc:** Ép mô hình phải dự đoán đúng giá trị của các biến từ các biến cha của chúng.
2. **Độ hợp lý của Nhiễu (NLL):** Đảm bảo phần sai số của mô hình tuân theo các quy luật thống kê tự nhiên, không bị vặn vẹo quá mức.
3. **KL Divergence:** Ngăn chặn việc mô hình chỉ tập trung vào một cơ chế duy nhất, ép nó phải khám phá sự đa dạng của dữ liệu.
4. **Rào chắn Phi chu trình (DAGMA):** Đây là thành phần quan trọng nhất, đóng vai trò như một lực đẩy cực mạnh ngăn cản các vòng lặp (Cycles) xuất hiện trong đồ thị.

**B. Rào chắn Tối ưu Phi Chu Trình DAGMA (Acyclicity Constraint):**

Bởi vì $W \in \mathbb{R}^{d \times d}$ là hệ tọa độ liên tục (cạnh $0.4, 0.7, \dots$), thuật toán DAGMA được tích hợp vào để chặn mạng nơ-ron học đồ thị hồi tiếp tự thân.
Dựa trên giá trị nghịch lưu Log-Determinant Barrier:
$$ h_{DAGMA}(W) = - \log \det(\mathbf{I} - \alpha W \circ W) $$
Đóng vai trò là Tường thành phạt (Penalty Barrier Function).

### 3.3.3 Vòng xoáy Tối ưu Augmented Lagrangian Method (ALM)

Vì bài toán yêu cầu Cực tiểu $\mathcal{L}_{\text{toàn cục}}$ VỚI ĐIỀU KIỆN RÀNG BUỘC (Subject to constraint) $h_{DAGMA}(W) = 0$. Hai không gian này mâu thuẫn (Mạng càng tạo chu trình Cycle, Loss MSE càng nhỏ). Do vậy, tôi triển khai thuật toán ALM lừng danh (Bertsekas, 1982). Quá trình huấn luyện không chỉ dốc xuống bằng thuật toán Gradient Adam truyền thống một chiều mà gồm 2 vòng lặp (Dual-Loop Optimization):

```mermaid
sequenceDiagram
    participant ALM as Vòng lặp ALM (Ngoài)
    participant Adam as Tối ưu hóa Adam (Trong)
    participant Graph as Đồ thị Nhân quả W

    ALM->>Adam: Bắt đầu với hệ số phạt nhẹ
    loop Huấn luyện Epoch
        Adam->>Graph: Cập nhật trọng số để giảm Loss
        Graph->>Adam: Tính toán vi phạm Chu trình h(W)
    end
    Adam->>ALM: Trả về đồ thị hiện tại
    ALM->>ALM: Kiểm tra ràng buộc
    Note over ALM: Nếu vẫn còn vòng lặp -> Tăng hình phạt gấp 10 lần
    ALM->>Adam: Tiếp tục huấn luyện với rào chắn cứng hơn
```

Kỹ thuật ALM đẩy hệ số cấm chập mạch to lên theo thời gian. Giai đoạn đầu, mô hình được tự do khám phá các mối quan hệ. Nhưng dần về sau, khi hình phạt xấp xỉ vô cực, hệ thống mạng nơ-ron buộc phải tự cắt bỏ những mắt xích yếu nhất để triệt tiêu các vòng lặp, chỉ giữ lại những đường dây nhân quả cốt lõi nhất.

---

## 3.4 Pha 3: Lọc Giao Tiếp Nhảm Bằng Cơ chế Hỗn Hợp Đồng Quy (Edge Post-Pruning Gate)

Kết thúc Pha 2 Mạng Neural, chúng ta thu được một Ma trận $W_{raw} = \sigma(W_{logits}) \in \mathbb{R}^{d \times d}$. Các hệ số lúc này bơi trong vùng liên tục (Ví dụ: 0.81, 0.45, 0.05, 0.001...). Vấn đề muôn thuở của Global optimization đó là $W_{raw}$ hầu như không bao giờ là Số Zeros hoàn chỉnh. Thậm chí các Node hoàn toàn không dính dáng, Mạng vẫn gán đại giá trị $0.1$. Làm thế nào để biết đường link $0.35$ kia là "Nhân quả thật sự" hay chỉ là "Sự bù trừ sai số hồi quy của mạng"?

Hầu hết hệ thống trên thế giới dùng một Threshold Tĩnh (Tất cả cạnh $< 0.3$ cắt bỏ). Điều này hết sức ngớ ngẩn do không phân định quan trọng trong từng cấu trúc (Biến $A$ có Range 10 triệu, Biến $B$ Rate 0.5, thì $W_{ij} = 0.02$ có thể là nhân quả sống còn chấn động hệ gen). 

Dự án DeepANM triển khai bộ thiết kế lọc 2 cổng (Double-Gate) cực kỳ chặt chẽ:

```mermaid
graph TD
    Raw[Đồ thị thô Phase 2] --> Gate1[Cổng 1: Neural Jacobian ATE]
    Raw --> Gate2[Cổng 2: Random Forest Importance]
    Gate1 --> Logic{Kết hợp & Lọc}
    Gate2 --> Logic
    Logic --> CI[Cổng 3: Test Độc lập Điều kiện CI]
    CI --> Final[DAG Cuối cùng]
```

### 3.4.1 Màng Lọc Cơ Sở Jacobian (Neural ATE Score)

Dựa trên lý thuyết can thiệp của Judea Pearl, ATE đo lường sự thay đổi của biến "Con" khi chúng ta tác động trực tiếp vào biến "Cha". DeepANM ép mạng nơ-ron phải tính toán: "Nếu tôi rung lắc nhẹ giá trị của biến Cha, biến Con sẽ phản ứng bao nhiêu?". Những cạnh có phản ứng quá yếu sẽ bị coi là nhiễu và loại bỏ ngay lập tức.

### 3.4.2 Màng Lọc Permutation Importance (Random Forest)

Để kiểm chứng lại một lần nữa bằng một phương pháp hoàn toàn khác, hệ thống sử dụng Rừng ngẫu nhiên (Random Forest). Chúng ta thử xáo trộn hoàn toàn dữ liệu của biến Cha. Nếu sau khi xáo trộn mà mô hình vẫn dự đoán tốt biến Con, chứng tỏ biến Cha đó chẳng có vai trò gì cả. Chỉ những biến nào mà việc xáo trộn nó gây ra một "cú sốc" sụt giảm độ chính xác mới được giữ lại.

### 3.4.3 Loại Bỏ Liên kết Gián Tiếp (Partial Correlation)
Thuật toán gạt nhầm nhánh gián tiếp dựa vào **Cây Tăng Cường Tốc độ (HistGradientBoostingRegressor)**:
- Dự đoán $A$ bằng tập Nền $B$. Trích xuất phần dư: $\varepsilon_{A|B} = A - \text{Nonlinear Model}(A, \text{với tập } B)$
- Dự đoán $C$ bằng tập Nền $B$. Trích xuất phần dư: $\varepsilon_{C|B} = C - \text{Model}(C, \text{với tập } B)$
- Chạy hệ số Tương quan tuyến tính (Pearson Correlation test) giữa bộ biến đổi $\varepsilon_{A|B}$ và $\varepsilon_{C|B}$.
- Trả về ngưỡng tham chiếu p-value. Nếu $p > 0.05$, các phần độc lập (Residuals) thể hiện rằng chúng hoàn toàn triệt tiêu khi biết thêm Nền trung gian. Bác bỏ giả thuyết có đường truyền thông tin liên kết riêng $A \to C$. Node C sẽ được tách rời.

Tính ưu việt tuyệt đối của Pha số 3 là khả năng lọc cực đoan (Ultra-pruning) mọi tín nhiệm giả với sự giao thoa kép của Mạng Neural Thần kinh Đại vi phân (Jacobian ATE) và Các Mô hình Cây Phi Tập Trung (Ensembles). Kết cục, đồ thị $W_{bin}$ được chắt lọc chỉ còn chứa những "Xương sống Huyết mạch" thực sự đại diện cho Trật tự tự nhiên vạn vật của tập Dữ liệu, kết thúc chu trình Discover Nhân quả hoành tráng.

---

## 3.5 Tiểu kết

Chương 3 của Đồ án đã minh họa cấu trúc chi tiết, nguyên lý phân rã thành luồng (Dataflow) cấp chuyên gia của hệ thống DeepANM. Điểm cốt lõi kỹ thuật mang tính công trình của mô hình nằm ở khả năng khép kín hoàn hảo các nhược điểm của Phương pháp Tối ưu Tĩnh tại 3 điểm cắt (3-Phase Model): Bắt đầu bằng việc thu hẹp Tầm nhìn NP-Hard vô hướng thông qua định lý Bất đối xứng O(N*d) TopoSort Sink-First. Tiếp theo, hệ quy chuẩn mạng Neural VAE-SCM ứng biến Gumbel-Softmax kết hợp Hàm Log-Determinant Tối ưu Kép (ALM) cho phép rẽ nhánh Cơ chế Phi tuyến Phức Tạp. Cuối cùng, cổng Trích Cạnh Thích nghi kép Jacobian-RandomForest loại bỏ sai phạm cạnh nhân quả trực quan cao. Kiến trúc này mang dáng vóc của một công trình Toán-Tin chuẩn Mực dành riêng cho khám phá hệ thống mạng lưới Y sinh hoặc Tài chính có hệ nhiễu cực đại, sẽ được kiểm chứng bằng thực nghiệm đo đạc ở Chương số 4.
