# Kịch bản Kiểm thử DeepANM (Test Scenarios)

Tập tin này mô tả toàn bộ kịch bản kiểm thử (unit tests) được định nghĩa trong `tests/test_core.py`. Các bài test được sử dụng để đảm bảo tính chính xác về mặt kiến trúc, kỹ thuật học sâu và toán học của toàn bộ framework khám phá nhân quả DeepANM.

## Danh sách Bài Test

| Tên Bài Test | Module | Mục tiêu & Cách thức Test (How & What) |
|:---|:---|:---|
| `test_mlp_components` | `src.core.mlp` | **Kiểm tra luồng Tensor của mạng MLP:** Đảm bảo Encoder sinh ra phân phối rời rạc (Gumbel), SEM tính ra dự đoán trung bình (mu), Decoder tính ra luồng biến đổi, và GMM tính toán log_prob hợp lệ không NaNs. Kích thước batch và số chiều (d) phải khớp tuyệt đối. |
| `test_heterogeneous_noise`| `src.core.mlp` | **Kiểm tra GMM Nhiễu (DECI):** Đưa nhiễu ngẫu nhiên vào mô hình GMM, kiểm tra đầu ra log-likelihood có trả về đúng 1 scalar cho mỗi sample (shape `[batch_size]`) và không bị nổ gradient (NaN/Inf). |
| `test_fast_hsic` | `src.core.gppom_hsic` | **Kiểm tra kiểm định độc lập O(N·D):** Chạy FastHSIC trên 2 biến ngẫu nhiên. Đảm bảo kết quả (bình phương chuẩn Frobenius) luôn $\ge 0$ và dạng vô hướng (scalar). |
| `test_dag_penalty` | `src.core.gppom_hsic` | **Kiểm tra hàm phạt DAGMA (Acyclicity):** Giả lập 3 ma trận trọng số (Đồ thị rỗng, Đồ thị chuỗi $A \to B \to C$, và Đồ thị có chu trình $A \to B \to A$). Hàng rào phạt $h(W)$ phải trả về $0$ cho 2 trường hợp đầu và $>0$ cho trường hợp cuối. |
| `test_score_asymmetry` | `src.core.toposort` | **Kiểm tra thuật toán TopoSort:** Đưa dữ liệu ngẫu nhiên vào `hsic_greedy_order`. Đảm bảo đầu ra là một hoán vị hợp lệ (chứa đủ toàn bộ các số từ $0$ đến $n-1$) và không bị thiếu biến. |
| `test_gppomc_core_forward`| `src.core.gppom_hsic` | **Kiểm tra lõi GPPOMC Forward:** Bơm dữ liệu giả vào lõi động cơ, kiểm tra xem nó có trả về đủ 4 thành phần loss (base_loss, mse, hsic, nll) là các vô hướng Gradient-enabled hợp lệ không. |
| `test_global_ate_matrix` | `src.core.mlp` | **Kiểm tra Neural ATE Jacobian:** Tính toán tác động nhân quả chéo trên batch. Đảm bảo ma trận xuất ra có kích thước $[d, d]$ và **đường chéo chính phải bằng 0** (một biến không tự tác động lên chính nó trong cùng mốc thời gian tĩnh). |
| `test_adaptive_lasso` | `src.utils.adaptive_lasso` | **Kiểm tra Chọn cạnh Phi tuyến:** Gọi `adaptive_lasso_dag` bằng OLS tuyến tính trên cụm dữ liệu nhỏ. Kiểm tra đầu ra phải là một ma trận DAG nhị phân hợp lệ $[d, d]$ với các phần tử chỉ gồm $\{0, 1\}$. |
| `test_deepanm_modes` | `src.models.deepanm` | **Kiểm tra vòng đời Wrapper DeepANM:** Khởi tạo DeepANM và gọi `fit()` với 2 chế độ `discovery_mode='alm'` và `discovery_mode='fast'`. Đảm bảo mô hình không văng lỗi khi hoán đổi động cơ, gọi `get_dag_matrix()` trả về ma trận hợp lệ. |
| `test_prior_constraint` | `src.models.deepanm` | **Kiểm tra Cơ chế Tri thức Miền (Exogenous):** Ép biến $X_0$ thành ngoại sinh (không có nguyên nhân). Kiểm tra xem constraint_mask và ma trận kết quả có thực sự block mọi mũi tên đâm vào $X_0$ hay không. |

## Lệnh Chạy Kiểm thử

```bash
# Chạy toàn bộ các bài test
pytest tests/test_core.py -v

# Chạy và hiển thị đầy đủ log (gỡ lỗi)
pytest tests/test_core.py -v -s
```
