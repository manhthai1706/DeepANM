# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

Chương này trình bày các khái niệm nền tảng và cơ sở toán học cấu thành nên quá trình Khám phá Nhân quả (Causal Discovery) từ dữ liệu quan sát. Các lý thuyết này bao gồm ngôn ngữ biểu diễn Đồ thị Có hướng Không chu trình (DAG), lý thuyết Mô hình Phương trình Cấu trúc (SEM), nguyên lý tính bất đối xứng của Mô hình Nhiễu Cộng (ANM), tiêu chuẩn đo lường sự phụ thuộc phi tuyến (HSIC), cũng như các kỹ thuật tối ưu hóa đồ thị liên tục. Đây là nền móng để xây dựng và chứng minh sự hoạt động của hệ thống DeepANM trong các chương tiếp theo.

## 2.1 Khung Cấu trúc Nhân quả (Causal Framework)

Để máy tính có thể "hiểu" và "học" được nhân quả, chúng ta cần một ngôn ngữ toán học thống nhất. Tính đến nay, hai công cụ biểu diễn toán học quyền lực nhất được sử dụng trong Trí tuệ Nhân tạo là Đồ thị DAG và Mô hình SEM.

### 2.1.1 Đồ thị có hướng không chu trình (Directed Acyclic Graph - DAG)

Trong Lý thuyết Đồ thị, một **DAG** được định nghĩa là một cặp $\mathcal{G} = (V, E)$, trong đó:
- $V = \{X_1, X_2, \dots, X_d\}$ là tập hợp các đỉnh (nodes), đại diện cho các biến / đặc trưng (features) quan sát được.
- $E \subseteq V \times V$ là tập hợp các cạnh có hướng (directed edges). Một cạnh $X_i \rightarrow X_j$ mang ý nghĩa: Biến $X_i$ là **nguyên nhân trực tiếp** gây ra $X_j$.

Một tính chất cực kỳ quan trọng của DAG là **Không có chu trình (Acyclic)**. Tức là, nếu ta xuất phát từ bất kỳ đỉnh $X_i$ nào, men theo các mũi tên có hướng, ta sẽ *không bao giờ* quay trở lại đúng vị trí của $X_i$. Đặc tả này phù hợp với dòng chảy của Thời gian và Quy luật Nhân quả trong vật lý học: Một sự kiện xảy ra ở tương lai không thể quay lại làm nguyên nhân gây ra một sự kiện ở quá khứ tạo ra nó.

Trong mã hóa tính toán, DAG được biểu diễn bằng một ma trận kề (Adjacency Matrix) trọng số $W \in \mathbb{R}^{d \times d}$. 
- $W_{ij} \neq 0$ tương đương với sự tồn tại của cạnh $X_i \rightarrow X_j$.
- $W_{ij} = 0$ nghĩa là không tồn tại quan hệ tác động trực tiếp từ $i$ đến $j$.

Tính chất phi chu trình của DAG tạo ra một trở ngại vô cùng lớn đối với Trí tuệ nhân tạo: Không gian của tất cả các đồ thị DAG khả dĩ (DAG Space) trên $d$ đỉnh mở rộng với tốc độ siêu hàm mũ (Super-exponential) quanh mốc $2^{d(d-1)/2}$. Với bài toán 20 biến, số lượng đồ thị khổng lồ đến mức các giải thuật duyệt cạn (Brute-force) là bất khả thi. Điều này khiến Khám phá Nhân quả mặc định là một bài toán **NP-Hard**.

### 2.1.2 Mô hình Phương trình Cấu trúc (Structural Equation Models - SEM)

Đồ thị DAG chỉ cho ta biết **ai là nguyên nhân của ai**, nhưng không cho biết **cách thức mà nguyên nhân tạo ra kết quả**. Mô hình SEM lấp đầy khoảng trống đó bằng cách định nghĩa các quan hệ dưới dạng các hàm số cơ học.

Giả sử $PA(X_j)$ là tập hợp tất cả các cha (nguyên nhân trực tiếp) của $X_j$ trong đồ thị DAG $\mathcal{G}$. Hệ thống SEM mô tả mỗi biến $X_j$ được sinh ra như sau:
$$X_j := f_j(PA(X_j), \varepsilon_j) \quad \text{for } j = 1, 2, \dots, d$$
Trong đó:
- $f_j$ là một hàm phi tuyến hoán đổi (cơ chế nhân quả / Causal Mechanism) không nhất thiết xác định trước.
- $\varepsilon_j$ là biến nhiễu ngoại sinh (Exogenous noise) hoặc những yếu tố ngẫu nhiên từ môi trường không đo lường được tác động trực tiếp lên $X_j$.
- Quy ước ký hiệu `:=` (phép gán) thay cho dấu `=` đại diện cho sự bất đối xứng. Nếu Cấp dưới phụ thuộc vào Sếp ($X_{cấp dưới} := f(X_{sếp})$), thì việc Cấp dưới tự ý làm việc không thể thay đổi hành vi của Sếp theo chiều ngược lại.

Trong Causal AI, mục tiêu của Khớp SCM (SCM Fitting) là từ dữ liệu đầu vào $X$, tìm ra được chính xác hình thái của danh sách các hàm $f_j$ và ma trận kề $W$ định hình tập $PA$.

### 2.1.3 Thống kê quan sát và Can thiệp thực nghiệm (Do-Calculus)

Nhà khoa học dữ liệu Judea Pearl (Đạt giải Turing 2011) đã đưa ra lý thuyết **Causal Hierarchy** phân rõ Thống kê học thông thường và Khoa học Nhân quả thành các nấc thang:

1. **Rung 1 - Tương quan / Quan sát (Seeing):** Tính xác suất điều kiện $\mathbb{E}[Y | X = x]$. "Nếu quan sát thấy hóa đơn tiền điện tăng, suy ra thời tiết đang rất nóng". Khớp trực tiếp dữ liệu.
2. **Rung 2 - Can thiệp hành động (Doing / Intervening):** Ký hiệu bằng toán tử $do(\cdot)$. Tính kỳ vọng $\mathbb{E}[Y | do(X = x)]$. "Nếu ta ép biểu giá điện tăng gấp 3, liệu thời tiết có trở nên nóng lên không?". 

Xác suất điều kiện thống kê khác biệt hoàn toàn với phân phối can thiệp. Việc thay đổi $do(X = x)$ cắt đứt hoàn toàn $X$ khỏi các nguyên nhân gốc của nó trong mô hình DAG (Mutilated Graph), khiến ta có thể đo lường Tác động Nhân quả Trung bình (Average Treatment Effect - ATE) như sau:
$$ATE = \mathbb{E}[Y | do(X = x + 1)] - \mathbb{E}[Y | do(X = x)]$$

Việc tính toán được ATE từ một hệ thống SEM học được bằng Mạng Neural là chìa khóa để phân định sức mạnh (Causal magnitude) của các trọng số cạnh học được, làm rào chắn cuối cùng loại bỏ nhiễu.

---

## 2.2 Mô hình Nhiễu Cộng (Additive Noise Models - ANM)

Bên cạnh khai thác hướng nhân quả dựa vào can thiệp hoặc thứ tự thời gian, ANM (Hoyer et al., 2009) là phương pháp nổi bật nhất để khám phá nhân quả trên dữ liệu lát cắt ngang (Cross-sectional observational data) tĩnh.

### 2.2.1 Khái niệm Nhiễu Cộng Phi tuyến (Nonlinear ANM)

ANM giới hạn dạng của phương trình cấu trúc (SEM) thành dạng cộng, sao cho tác động của các biến nguyên nhân và nhiễu ngẫu nhiên là độc lập cộng tuyến với nhau:
$$X_j := f_j(PA(X_j)) + \varepsilon_j \quad (\text{hoặc mở rộng ra PNL: } X_j := g_j(f_j(PA(X_j)) + \varepsilon_j))$$
Mấu chốt của ANM yêu cầu hai giả định mạnh:
1. Hàm $f_j$ là phi tuyến (Nonlinear).
2. Nhiễu $\varepsilon_j$ độc lập thống kê với các nguyên nhân $PA(X_j)$, tức $\varepsilon_j \perp\!\!\!\perp PA(X_j)$.

### 2.2.2 Tính Bất đối xứng của ANM (ANM Asymmetry)

Nền tảng của thuật toán nằm ở Việc chứng minh toán học phức tạp rằng: **Nếu dữ liệu thật sự được sinh ra từ $X \rightarrow Y$ bởi hệ phương trình ANM phi tuyến, thì điều ngược lại $Y \rightarrow X$ là không thể khớp được bởi phương trình ANM**.

Điều này dẫn đến thuật toán kinh điển đi tìm chiều $X \rightarrow Y$ hay $Y \rightarrow X$:
1. Coi $X$ là nguyên nhân, dùng học máy phi tuyến (Mạng Neural, Random Forest, XGBoost) học hàm $Y \approx f(X)$. Tính phần dư: $\varepsilon_Y = Y - f(X)$. 
2. Đánh giá tính độc lập thống kê: Kiểm tra xem khoảng dao động của $\varepsilon_Y$ có độc lập với $X$ không ($\varepsilon_Y \perp\!\!\!\perp X$).
3. Đổi chiều chiều ngược lại: Coi $Y$ là nguyên nhân, học $X \approx g(Y)$, tính phần dư $\varepsilon_X = X - g(Y)$ và đối chứng độc lập $\varepsilon_X \perp\!\!\!\perp Y$.
4. Lý thuyết chứng minh rằng: **Chỉ có chiều diễn tiến thật sự của tự nhiên mới duy trì được tính độc lập giữa Nguyên nhân và Phần dư.** Chiều ngược lại (Học máy buộc phải ép đường cong ngược) sẽ tạo hình dạng phần dư phụ thuộc mạnh mẽ vào vị trí điểm ảnh của biến nguyên nhân giả (Dependent Residuals).

Sự thay đổi về tính độc lập phần dư (Residual Dependence) này là hòn đá tảng cho bộ đánh giá TopoSort Sink-First. Tuy nhiên, nó yêu cầu một phép thử độc lập thống kê cực kỳ nhạy bén - HSIC, mà các tham số hiệp phương sai tuyến tính truyền thống (Covariance, Pearson Correlation) thất bại hoàn toàn.

---

## 2.3 Tiêu chuẩn Độc lập Thống kê Hilbert-Schmidt (HSIC)

Tiêu chuẩn Độc lập Hilbert-Schmidt (Hilbert-Schmidt Independence Criterion - HSIC) được Gretton (2005) đề xuất là công cụ mạnh mẽ nhất trong Máy học để kiểm tra sự độc lập của hai biến ngẫu nhiên phi tuyến $X$ và $Y$ dù chúng có độ trĩ lệch hay phân phối đa phương thức khổng lồ. 

### 2.3.1 Công thức Toán học của HSIC

Định lý toán học chỉ ra $X$ và $Y$ hoàn toàn độc lập khi và chỉ khi phân phối đồng thời (Joint distribution) bằng tích của các phân phối biên (Marginal distribution): $P_{XY} = P_X P_Y$.

Thay vì phải nội suy mật độ (Density Estimation) vốn là bài toán nan giải, HSIC chiếu dữ liệu $X$ vào không gian Hilbert vô hạn chiều $\mathcal{F}$ thông qua hàm ánh xạ đặc trưng $\phi(x)$, và $Y$ vào không gian $\mathcal{G}$ qua $\psi(y)$. Tác giả dùng chuẩn Frobenius bình phương đối với toán tử cross-covariance:

Giá trị HSIC thực nghiệm (Empirical HSIC) được tính toán thông qua ma trận Kernel của X và Y:
$$HSIC(\mathbf{X}, \mathbf{Y}) = \frac{1}{(n-1)^2} \text{Tr}(\mathbf{K}_X \mathbf{H} \mathbf{K}_Y \mathbf{H})$$
Trong đó:
- $\mathbf{X}, \mathbf{Y}$ là dữ liệu mẫu gồm $n$ điểm dữ liệu.
- $\mathbf{K}_X, \mathbf{K}_Y \in \mathbb{R}^{n \times n}$ là Ma trận Kernel (thường là Gaussian RBF: $k(x, x') = \exp(-\frac{||x-x'||^2}{2\sigma^2})$).
- $\mathbf{H} = \mathbf{I}_n - \frac{1}{n} \mathbf{1}_{n \times n}$ là ma trận chuẩn hóa tâm (Centering matrix).

Nếu $HSIC(X, Y) \approx 0$, hai biến hoàn toàn độc lập. Càng lớn hơn $0$, mức độ phụ thuộc phi tuyến càng mạnh. Hàm Loss này thường xuyên được cho vào Mạng Neural để "ép" mạng phải tìm ra khoảng không ẩn độc lập tối đa.

### 2.3.2 Khắc phục giới hạn bằng Random Fourier Features (RFF)

Tính toán ma trận $n \times n$ cho $\mathbf{K}_X$ tiêu tốn dung lượng bộ nhớ lớn và tốc độ vận hành $O(N^2)$. Đối với các mạng Deep Learning cần tính HSIC ở từng lần đạo hàm Epoch (hàng vạn vòng lặp), tốc độ này dẫn đến hiện tượng treo cổ chai.

Dựa trên Định lý Bochner, Kernel tịnh tiến RBF có thể được xấp xỉ bằng giá trị kỳ vọng của các phép chiếu sin/cos ngẫu nhiên kết hợp phân phối Gaussian (Rahimi & Recht, 2007). Thay vì chiếu đồ thị vô hạn chiều (Kernel Trick), ta rút thẳng giá trị Fourier Hữu hạn Chiều Explicit $D$:

$$\phi_{\text{RFF}}(x) = \sqrt{\frac{2}{D}} \cos(X \Omega + b)$$
- $\Omega \in \mathbb{R}^{d_x \times D}$ được lấy mẫu từ $\mathcal{N}(0, 1/\sigma^2)$.
- $b \in \mathbb{R}^{D}$ là bias lấy mẫu đều từ $[0, 2\pi]$.

Lúc này, HSIC được tính như bình phương của độ dài vector Cross-Covariance trong không gian tuyến tính $D$-chiều (với $D$ thường nhỏ, cỡ $64$ đến $128$):
$$\tilde{HSIC}(X, Y) \approx \frac{1}{n^2} ||\tilde{\Phi}_{X}^T \tilde{\Phi}_{Y}||^2_F$$
Áp dụng **FastHSIC (RFF-HSIC)** cho phép thời gian tính độ Độc lập rơi xuống $O(N \cdot D)$, giải phóng giới hạn cho việc tìm kiếm thuật toán Sắp xếp Topological khổng lồ phía trước mạng Neural.

---

## 2.4 Bài toán Khai phá Cấu trúc Liên tục (Continuous DAG)

Làm thế nào để huấn luyện Mạng Neural dò đường ra bộ trọng số kề phi chu trình $W$? Giải thuật di truyền hay tìm kiếm duyệt cạn (A*) đều bất khả thu do không gian NP-Hard.

### 2.4.1 Cuộc cách mạng NOTEARS

Mọi sự thay đổi bắt nguồn từ công bố của Zheng et al. (2018) tại hội nghị NeurIPS với giải thuật **NOTEARS**. Họ thay thế tìm kiếm rời rạc cấm cạnh bằng một điều kiện Toán học có thể tính đạo hàm:
**Một đồ thị $W$ là DAG có hướng không chu trình khi và chỉ khi hàm phạt $h(W)$ thỏa mãn:**
$$h_{NOTEARS}(W) = \text{Tr}(e^{W \circ W}) - d = 0$$

Phép toán lũy thừa ma trận (matrix exponential) đếm chính xác tổng số vòng lặp có chiều dài $1, 2, ..., \infty$ trên đồ thị. Khi $Trace(e^{W \circ W})$ có giá trị bằng chính số chiều $d$, tức là ta có đúng $d$ đường kính độ dài 0 (self-loop mặc định) và không có đường kính lớn hơn 0 nào đâm ngược về node cũ (Không có cycle). 

Công trình này chuyển Khám phá nhân quả thành một bài toán **Tối ưu hóa Trọng số Liên tục (Continuous Optimization)**, mở ra cánh cửa áp dụng Back-propagation của các mạng Deep Learning (Adamizer, Gradient Descent).

### 2.4.2 Hàm phạt thế hệ thứ ba - DAGMA

Tuy nhiên NOTEARS còn cồng kềnh với hàm $e^A$ tạo Gradient nổ. Năm 2022, Bello et al. xuất bản cải tiến Log-Determinant Barrier gọi là **DAGMA**:
Khẳng định $W$ là DAG khi và chi khi:
$$h_{DAGMA}(W) = -\ln \det(sI - W \circ W) + d \ln s = 0$$
Với tham số $s$ kiểm soát bán kính phủ.

Hệ thống hàm $h(W)$ liên tục của DAGMA được kiểm soát bằng giải thuật Augmented Lagrangian Method (ALM) theo công thức kinh điển:
$$\min_{\Theta} \mathcal{L}_{Neural}(X; \Theta) + \alpha h(W(\Theta)) + \frac{\rho}{2} h(W(\Theta))^2$$
Cập nhật nhân quả kép tăng $\rho$ vô hạn dần theo chu kỳ Epochs, dẫn đến các cạnh sai trái (ngược chu trình) dần bị Mạng neural cắt bỏ tự động thành số $0$ trên không gian tối ưu.

---

## 2.5 Phương pháp Lọc Trọng số Cạnh (Edge Pruning)

Mạng Neural rất dễ rơi vào Tối ưu cực tiểu cục bộ (Local Minima) và giữ lại các cạnh giả (cạnh tạo ra nhiễu hoặc cạnh gián tiếp). Dễ hiểu, Mạng Neural "lười" hơn con người và sẽ nối trực tiếp $A \to C$ nếu $A$ ảnh hưởng rất nhỏ đến $C$ (tác động gián tiếp $A \to B \to C$), dẫn đến ma trận DAG bị rậm rạp. Quá trình chọn cạnh chặt chẽ là bắt buộc ở đuôi quá trình phát hiện (Phase 3). 

### 2.5.1 Kỹ thuật Adaptive LASSO phi tuyến tĩnh

LASSO (Least Absolute Shrinkage and Selection Operator) bổ sung hàm Penalty cấp $L_1$ vào tác vụ hồi quy để triệt tiêu các hệ số kém quan trọng về $0$.
Nhưng LASSO rất tệ ở việc lọc cạnh có tỷ lệ đa cộng tuyến (Multicollinearity). Thuật toán **Adaptive LASSO (Zou, 2006)** đưa ra giải pháp cân bằng bằng cách trừng phạt các biến phụ thuộc dựa trên nghịch đảo độ quan trọng sơ khởi thông qua ma trận $\beta_{\text{init}}$:
$$ \arg\min_{\beta} \left( ||y - X\beta||^2_2 + \lambda \sum_j \frac{1}{|\hat{\beta}_{init, j}|^\gamma} |\beta_j| \right) $$

### 2.5.2 Phân tích tầm quan trọng bằng Random Forest (Permutation Importance)

Để thay thế $\beta_{init}$ dạng tuyến tính quá lỏng lẻo bằng một kỹ thuật mạnh, DeepANM dựa trên mô hình phi tập trung Random Forest (Breiman, 2001). 
Thuật toán đi hoán vị lộn xộn dọc (Shuffling vertically) một cột dữ liệu của biến nguyên nhân $X_c$. Nếu mô hình Random Forest Regressor bị phân rã chỉ số chính xác (R² Score R-Squared Drop) mạnh > $3\%$, tức là đặc trưng này mang tính sống còn đối với Kết quả. Ngược lại, nếu Shuffle qua lại nhưng chất lượng phán đoán $Y$ vẫn vậy, thì rõ ràng cạnh $X \to Y$ là vô giá trị trong đời thực. 

### 2.5.3 Lọc thông tin gián tiếp qua Độ tương quan Điều kiện phần dư

Mạng Neural có xu hướng báo "False Positive" tại các đường dẫn gián tiếp. Ví dụ Khói bụi $X_1 \to$ Bệnh Phổi $X_2 \to$ Ho kinh niên $X_3$. Việc liên tục thấy Khói bụi làm người ta tự động kết luận $X_1 \to X_3$.
Kỹ thuật **Conditional Independence CI** được diễn giải đơn giản:
$$X_1 \perp\!\!\!\perp X_3 \,|\, X_2$$
Nghĩa là, cho dù Khói Bụi có dày đặc thế nào ($X_1$), nhưng nếu bệnh nhân Đã Khám và Xác nhận Không mắc Bệnh Phổi ($X_2 = \text{False}$), thì tỷ lệ Ho kinh niên $X_3$ hoàn toàn không bị thay đổi.
Bằng việc mô hình hóa các biến số phi tuyến bằng Histogram-based Gradient Boosting và dùng hệ số Correlation Test Pearson lên phần dư, ta dễ dàng bác bỏ cạnh thừa $X_1 \to X_3$, chỉ giữ lại quan hệ cha-con tinh khiết.

## 2.6 Tổng kết chương

Chương 2 đã cung cấp bức tranh toàn cảnh về nền tảng lý thuyết cần thiết để cấu trúc nên một mô hình học máy khám phá nhân quả. Trọng tâm của chương nằm ở việc hệ thống hóa khái niệm Đồ thị DAG, nguyên lý phân tích bất đối xứng của Mô hình Nhiễu cộng (ANM) và cách thức chuyển đổi bài toán tìm kiếm đồ thị rời rạc thành tối ưu hóa hàm phạt liên tục thông qua DAGMA/NOTEARS. Bên cạnh đó, các công cụ toán học tối quan trọng như kỹ thuật xấp xỉ RFF để giải quyết độ trễ của HSIC, hay phương pháp chọn cạnh phi tuyến bằng Random Forest kết hợp độc lập có điều kiện cũng đã được phân tích chi tiết. Đây là những khối "lego" học thuật đặc biệt cần thiết, đóng vai trò là tiền đề tiên quyết để đồ án giải quyết hệ kiến trúc đường ống phức tạp của hệ thống mạng Neural DeepANM tại Chương 3.
