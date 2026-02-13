# DeepANM

[![Architecture](https://img.shields.io/badge/Kiến_trúc-Chi_tiết-blueviolet?style=flat-square)](ARCH.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

## Tổng quan

DeepANM (Deep Additive Noise Model) là một thư viện Python dùng để xác định mối quan hệ nhân quả từ dữ liệu quan sát. Dự án được phát triển dựa trên framework ANM-MM (Additive Noise Model - Mixture Model) của [amber0309](https://github.com/amber0309/ANM-MM), với các thay đổi chính ở phần backbone mạng nơ-ron và cách tổ chức mã nguồn.

Mục tiêu của dự án là gói gọn toàn bộ quy trình khám phá nhân quả — từ tiền xử lý dữ liệu, huấn luyện mô hình đến trích xuất đồ thị — vào trong một class `DeepANM` duy nhất, thay vì phải gọi nhiều hàm rời rạc như bản gốc.

## Giới thiệu

### Bài toán

Khám phá nhân quả (Causal Discovery) là bài toán xác định quan hệ nguyên nhân - kết quả giữa các biến từ dữ liệu quan sát thuần túy, không cần thực nghiệm can thiệp. Ví dụ: cho hai biến X và Y có tương quan, cần xác định X gây ra Y hay Y gây ra X.

### Hướng tiếp cận

DeepANM sử dụng mô hình nhiễu cộng (Additive Noise Model - ANM): nếu dữ liệu được sinh từ cơ chế `Y = f(X) + N` với `N` độc lập thống kê với `X`, thì kết luận X là nguyên nhân của Y. Mô hình dùng mạng nơ-ron để xấp xỉ hàm `f`, sau đó kiểm tra tính độc lập giữa phần dư và biến đầu vào bằng HSIC.

Đối với bài toán song biến (2 biến), hệ thống huấn luyện hai mô hình riêng biệt: một cho giả thuyết X→Y và một cho giả thuyết Y→X. Mô hình nào cho phần dư độc lập hơn với biến nguyên nhân (HSIC thấp hơn) thì giả thuyết đó được chấp nhận. Đây là cơ chế Fixed-Structure Hypothesis Testing.

### Mở rộng hỗn hợp (Mixture Model)

Trong thực tế, mối quan hệ giữa X và Y có thể do nhiều cơ chế khác nhau tạo ra (ví dụ: cùng là mối quan hệ giữa thuốc và tác dụng, nhưng cơ chế khác nhau ở từng nhóm bệnh nhân). DeepANM sử dụng một VAE head kết hợp với Gumbel-Softmax để phân cụm các cơ chế này một cách khả vi (differentiable), cho phép mô hình xử lý dữ liệu hỗn hợp mà không cần tách thủ công.

---

## Cấu trúc mã nguồn

```
deepanm/
├── core/                       # Các thành phần tính toán cốt lõi
│   ├── mlp.py                  # Backbone mạng nơ-ron (ResNet, GRN, Attention, NSF, VAE)
│   ├── gppom_hsic.py           # Module tối ưu hóa DAG + hàm mất mát tổng hợp
│   ├── hsic.py                 # Triển khai HSIC (Gamma Approximation & Permutation Test)
│   └── kernels.py              # Thư viện kernel (RBF, Matern, Rational Quadratic,...)
├── models/                     # Giao diện người dùng
│   ├── deepanm.py              # Class DeepANM chính (fit, predict_direction,...)
│   ├── trainer.py              # Vòng lặp huấn luyện (AdamW, temperature annealing)
│   └── analysis.py             # Hàm ANMMM_cd (Hypothesis Testing) và ANMMM_clu (Clustering)
```

---

## Các thành phần kỹ thuật

### 1. Backbone mạng nơ-ron (`mlp.py`)

Đây là phần thay đổi lớn nhất so với bản gốc. Backbone gồm các lớp sau:

- **Gated Residual Network (GRN):** Một mạng gating học cách chọn lọc biến đầu vào nào quan trọng. Cơ chế hoạt động: nhân đầu vào `x` với `sigmoid(gate(x))`, trong đó `gate` là một mạng 2 lớp. Ý tưởng lấy từ kiến trúc Temporal Fusion Transformers (Lim et al., 2021). Sau lớp gating là 3 khối ResBlock nối tiếp.

- **ResBlock:** Mỗi khối gồm 2 lớp Linear, LayerNorm, GELU activation và Dropout. Có residual connection `output = GELU(x + block(x))` để tránh vanishing gradient khi mạng sâu.

- **VAE head:** Hai lớp Linear riêng biệt ánh xạ đặc trưng thành `mu_z` và `logvar_z`. Biến tiềm ẩn `z` được lấy mẫu bằng reparameterization trick, sau đó đi qua softmax với temperature (Gumbel-Softmax) để tạo phân phối xác suất trên `n_clusters` cơ chế. KL divergence được thêm vào hàm mất mát để regularize.

- **Neural Spline Flow (NSF):** Một lớp `MonotonicSplineLayer` dùng để biến đổi nhiễu. Thay vì giả định nhiễu tuân theo phân phối cố định, lớp này học một hàm đơn điệu (monotonic) để biến đổi phân phối nhiễu thành dạng phức tạp hơn. Tham số hóa bằng `n_bins` bin, mỗi bin có width, height và derivative riêng.

- **PNL Transform:** Lớp `InvertibleLayer` triển khai biến đổi Post-Nonlinear: `f(x) = softplus(w) * x + b`. Dùng `softplus` để đảm bảo trọng số luôn dương → hàm luôn đơn điệu tăng → khả nghịch.

### 2. Module tối ưu hóa DAG (`gppom_hsic.py`)

Module này kết nối backbone với các ràng buộc toán học:

- **Ma trận kề `W_dag`:** Một ma trận `d×d` (d = tổng số biến) được khởi tạo bằng 0 và học trong quá trình huấn luyện. Giá trị `W[i,j]` thể hiện mức độ ảnh hưởng nhân quả từ biến `i` đến biến `j`.

- **NOTEARS penalty:** Ràng buộc không vòng (acyclicity) được biểu diễn bằng công thức `h(W) = tr(e^{W⊙W}) - d`. Khi `h(W) = 0`, đồ thị không có vòng. Giá trị này được cộng vào hàm mất mát với trọng số 2.0.

- **Random Fourier Features (RFF):** Thay vì tính kernel matrix đầy đủ O(N²), module sử dụng RFF để xấp xỉ kernel RBF với độ phức tạp O(N). Mỗi `RFFGPLayer` gồm một ma trận chiếu cố định `W` và phase `b` ngẫu nhiên, với hai tham số học được là `log_alpha` (biên độ) và `log_gamma` (bandwidth).

- **FastHSIC:** Tính HSIC giữa hai tập biến dựa trên RFF. Thay vì tính ma trận kernel đầy đủ, FastHSIC ánh xạ dữ liệu vào không gian đặc trưng qua RFF rồi tính hiệp phương sai chéo. Được dùng ở hai chỗ: (1) ép phần dư PNL độc lập với đầu vào, (2) ép phân cụm cơ chế `z` độc lập với đầu vào.

- **Hàm mất mát tổng hợp:** `L = L_reg + 2.0 * L_dag + λ * log(HSIC_clu) + 3.0 * log(HSIC_pnl) + 0.2 * KL_vae`, trong đó `L_reg` là MSE giữa dự đoán và thực tế, `L_dag` là NOTEARS penalty, hai thành phần HSIC ép tính độc lập, và `KL_vae` regularize biến tiềm ẩn.

### 3. Huấn luyện (`trainer.py`)

- **Optimizer:** AdamW với weight decay 1e-2.
- **Temperature annealing:** Nhiệt độ Gumbel-Softmax giảm tuyến tính từ 1.0 xuống 0.5 theo epoch, giúp phân cụm cơ chế dần chuyển từ soft sang hard.
- **Logging:** In loss, regression loss và HSIC mỗi 50 epoch.

### 4. Phân tích hướng nhân quả (`analysis.py`)

Hàm `ANMMM_cd(data, lda)` thực hiện quy trình:

1. Tạo model CausalFlow thứ nhất, khóa `W_dag` theo hướng X→Y (chỉ `W[0,1]=1`, còn lại bằng 0). Huấn luyện 200 epoch.
2. Tạo model CausalFlow thứ hai, khóa `W_dag` theo hướng Y→X (chỉ `W[1,0]=1`). Huấn luyện 200 epoch.
3. Với mỗi model: trích xuất phần dư, tính HSIC giữa phần dư và biến nguyên nhân, đồng thời đánh giá stability.
4. Tính điểm tổng hợp: `score = HSIC * (1 + stability * 0.5)`. Hướng nào có score thấp hơn (phần dư độc lập hơn với nguyên nhân) thì được chọn.

### 5. Tiền xử lý dữ liệu (trong test scripts)

- **QuantileTransformer** (scikit-learn): Chuẩn hóa phân phối biên (marginal distribution) của mỗi biến về dạng xấp xỉ Gaussian. Giúp mô hình hội tụ tốt hơn trên dữ liệu có phân phối lệch.
- **Isolation Forest** (scikit-learn): Loại bỏ khoảng 5% điểm ngoại lai (contamination=0.05) trước khi đưa vào mô hình. Giúp giảm ảnh hưởng của outlier lên kết quả huấn luyện.

---

## So sánh với dự án gốc (amber0309)

| Thành phần | amber0309 (Base) | CausalFlow |
| :--- | :--- | :--- |
| Cấu trúc mã | Các script và hàm riêng lẻ | Package phân lớp (`core/`, `models/`) |
| Backbone | MLP tiêu chuẩn | GRN + 3 ResBlock + LayerNorm + GELU |
| Mô hình nhiễu | Phân phối cố định | Neural Spline Flow (monotonic, n_bins=8) |
| Phạm vi đồ thị | Chủ yếu song biến | Hỗ trợ đa biến qua NOTEARS (`W_dag` d×d) |
| Tính HSIC | Kernel matrix đầy đủ O(N²) | RFF approximation O(N) |
| Phân cụm cơ chế | K-Means hậu kỳ | Gumbel-Softmax end-to-end |
| Giao diện | Gọi hàm trực tiếp | Class API (`model.fit()`, `model.predict_direction()`) |
| Phân tích thêm | Không | Counterfactual, stability check |
| Tiền xử lý | Thủ công | Tích hợp IsolationForest + QuantileTransformer |

---

## Cài đặt

```bash
pip install git+https://github.com/manhthai1706/CausalFlow.git
```

**Yêu cầu:** Python ≥ 3.8, PyTorch ≥ 1.10, scikit-learn ≥ 1.0, numpy, matplotlib.

## Sử dụng

### Xác định hướng nhân quả (song biến)
```python
from deepanm import DeepANM

# data: numpy array shape [n_samples, 2]
model = DeepANM(lda=12.0)
direction = model.predict_direction(data)
# direction = 1: cột 0 gây ra cột 1 (X→Y)
# direction = -1: cột 1 gây ra cột 0 (Y→X)
```

### Huấn luyện đa biến và trích xuất DAG
```python
# X: numpy array shape [n_samples, n_variables]
model = DeepANM(data=X, epochs=200)

# Lấy ma trận kề
W_raw, W_binary = model.get_dag_matrix(threshold=0.1)
# W_raw: ma trận trọng số thực
# W_binary: ma trận nhị phân (1 nếu |W[i,j]| > threshold)
```

### Kiểm tra stability
```python
stability_score, losses = model.check_stability(data, n_splits=3)
# stability_score: std(losses) / mean(losses), giá trị càng nhỏ càng ổn định
```

### Phân tích counterfactual (song biến)
```python
y_cf = model.predict_counterfactual(x_orig=1.0, y_orig=2.0, x_new=3.0)
# Dự đoán: nếu X thay đổi từ 1.0 sang 3.0 thì Y sẽ bằng bao nhiêu?
```

---

## Kết quả Thực nghiệm

Đánh giá trên tập dữ liệu Sachs (Protein Signaling Network), gồm 11 biến protein và 17 cạnh nhân quả đã biết. Tiền xử lý: QuantileTransformer + IsolationForest (contamination=5%).

- **Accuracy**: 70.6% (12/17 cạnh đúng hướng)
- **SHD**: 5

### So sánh hiệu năng trên Sachs

Bảng dưới đây tổng hợp kết quả từ các bài báo gốc và các benchmark công khai. SHD (Structural Hamming Distance) càng thấp càng tốt. Accuracy đo tỉ lệ cạnh xác định đúng hướng.

| Phương pháp | Loại | Accuracy (Sachs) | SHD | Nguồn |
| :--- | :--- | :--- | :--- | :--- |
| PC Algorithm | Constraint-based | ~50-55% | ~17 | Spirtes et al. (2000) |
| GES | Score-based | ~55% | ~15 | Chickering (2002) |
| ICA-LiNGAM | Functional | ~55-60% | ~14 | Shimizu et al. (2006) |
| MMHC | Hybrid | ~55% | ~16 | Tsamardinos et al. (2006) |
| CAM | Functional (Additive) | ~58% | ~13 | Bühlmann et al. (2014) |
| NOTEARS | Continuous Opt. | ~60% | > 8 | Zheng et al. (2018) |
| DAG-GNN | Deep Learning | ~60% | ~19 | Yu et al. (2019) |
| **DeepANM** | **Deep Learning (ANM)** | **70.6%** | **5** | Dự án này |

> **Ghi chú:** Các con số của phương pháp khác là giá trị tham khảo từ các bài báo gốc và tổng hợp benchmark (Vowels et al., 2022). Kết quả có thể khác nhau tùy vào cách tiền xử lý và cài đặt tham số. Kết quả của DeepANM được đo trực tiếp trên code này với tiền xử lý QuantileTransformer + IsolationForest.

---

## Tham khảo

1. **ANM-MM (amber0309).** [GitHub](https://github.com/amber0309/ANM-MM).
2. **Zheng, X., et al. (2018).** "DAGs with NO TEARS." *NeurIPS*.
3. **Durkan, C., et al. (2019).** "Neural Spline Flows." *NeurIPS*.
4. **Zhang, K., & Hyvarinen, A. (2009).** "On the Identifiability of the Post-Nonlinear Causal Model." *UAI*.
5. **Rahimi, A., & Recht, B. (2007).** "Random Features for Large-Scale Kernel Machines." *NeurIPS*.
6. **Gretton, A., et al. (2007).** "A Kernel Statistical Test of Independence." *NeurIPS*.
7. **Vaswani, A., et al. (2017).** "Attention Is All You Need." *NeurIPS*.
8. **Jang, E., et al. (2016).** "Categorical Reparameterization with Gumbel-Softmax." *ICLR*.
9. **Kingma, D. P., & Welling, M. (2013).** "Auto-Encoding Variational Bayes." *ICLR*.
10. **He, K., et al. (2016).** "Deep Residual Learning for Image Recognition." *CVPR*.
11. **Ba, J. L., et al. (2016).** "Layer Normalization." *arXiv*.
12. **Hendrycks, D., & Gimpel, K. (2016).** "Gaussian Error Linear Units (GELUs)." *arXiv*.
13. **Lim, B., et al. (2021).** "Temporal Fusion Transformers." *IJF*.
14. **Loshchilov, I., & Hutter, F. (2017).** "Decoupled Weight Decay Regularization." *ICLR*.
15. **Liu, F. T., et al. (2008).** "Isolation Forest." *ICDM*.
16. **Pedregosa, F., et al. (2011).** "Scikit-learn." *JMLR*.
17. **Paszke, A., et al. (2019).** "PyTorch." *NeurIPS*.

## License
MIT License.
