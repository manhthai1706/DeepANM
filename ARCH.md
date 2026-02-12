# CausalFlow: Chi tiết Kiến trúc

Tài liệu mô tả luồng hoạt động bên trong của từng file và sơ đồ tổng thể của hệ thống.

---

## 1. Sơ đồ tổng thể hệ thống

```mermaid
graph TD
    A[Dữ liệu thô] --> B[QuantileTransformer]
    B --> C[IsolationForest]
    C --> D["CausalFlow.__init__(data) hoặc fit()"]
    D --> E[Khởi tạo GPPOMC_lnhsic_Core]
    E --> F[CausalFlowTrainer.train]
    F --> G[MLP backbone]
    G --> H["z_soft, mu, log_var, pnl_transform"]
    H --> I["masked_input = data @ |W_dag|"]
    I --> J["phi = RFF_z(z_soft) * RFF_x(masked)"]
    J --> K["y_pred = linear_head(phi)"]
    K --> L["Tính loss tổng hợp"]
    L --> M["backward + AdamW"]
    M --> N{Truy vấn kết quả}
    N --> O["get_dag_matrix()"]
    N --> P["predict_direction()"]
    N --> Q["predict_counterfactual()"]
    N --> R["check_stability()"]
```

---

## 2. `mlp.py` — Backbone mạng nơ-ron

```mermaid
graph TD
    A["Input x: batch, input_dim"] --> B["Gate = Sigmoid(Linear→GELU→Linear)"]
    B --> C["gated_x = x * gate(x)"]
    C --> D[ResBlock 1]
    D --> E[ResBlock 2]
    E --> F[ResBlock 3]
    F --> G[feat]

    G --> H["z_mean = Linear → n_clusters"]
    G --> I["z_logvar = Linear → n_clusters"]
    H --> J["z = reparameterize(mu, logvar)"]
    I --> J
    J --> K["z_soft = softmax(z / temp)"]

    G --> L["regressor = Linear → output_dim*2"]
    L --> M["chunk → mu, log_var"]

    M --> N["MonotonicSplineLayer(randn_like mu)"]
    N --> O[noise_complex]

    M --> P["InvertibleLayer: softplus(w)*x + b"]
    P --> Q[y_trans]
```

### Chi tiết ResBlock

```mermaid
graph TD
    A[x] --> B["Linear → LayerNorm → GELU → Dropout → Linear → LayerNorm"]
    A --> C["residual connection"]
    B --> C
    C --> D["output = GELU(x + block x)"]
```

---

## 3. `gppom_hsic.py` — Module tối ưu hóa DAG

```mermaid
graph TD
    A["batch_data: batch, d"] --> B[MLP backbone]
    B --> C["z_soft, kl_loss"]
    B --> D["mu, log_var"]

    A --> E["masked = data @ abs W_dag"]
    E --> F["phi_x = RFFGPLayer d, 128"]
    C --> G["phi_z = RFFGPLayer n_clusters, 128"]
    F --> H["phi = phi_z * phi_x"]
    G --> H
    H --> I["y_pred = linear_head phi"]

    I --> J["L_reg = MSE y_pred vs data"]
    
    A --> K["h_y = pnl_transform data"]
    I --> L["res_pnl = h_y - y_pred"]
    K --> L
    L --> M["L_hsic_pnl = FastHSIC data, res_pnl"]
    
    A --> N["L_hsic_clu = FastHSIC data, z_soft"]
    C --> N

    D --> O["L_kl = KL divergence"]

    P["W_dag"] --> Q["L_dag = tr exp W*W - d"]

    J --> R["total = L_reg + 2*L_dag"]
    Q --> R
    M --> S["+ 3*log L_hsic_pnl"]
    N --> T["+ lambda*log L_hsic_clu"]
    O --> U["+ 0.2*L_kl"]
    R --> V[total_loss]
    S --> V
    T --> V
    U --> V
```

---

## 4. `kernels.py` — Thư viện kernel khả vi

Tất cả kernel đều là `nn.Module` với `log_alpha`, `log_gamma` học được qua backprop.

```mermaid
graph TD
    A["x1, x2"] --> B{Chọn kernel}

    B --> C["RBFKernel
    K = alpha * exp -0.5*gamma*dist_sq
    Hỗ trợ ARD"]

    B --> D["LinearKernel
    K = alpha * x1^T x2 + bias"]

    B --> E["PolynomialKernel
    K = alpha*x1^T*x2 + bias ^ order"]

    B --> F["MaternKernel
    nu = 0.5 / 1.5 / 2.5"]

    B --> G["RationalQuadraticKernel
    K = alpha*(1+dist/(2*a*l^2))^-a"]

    B --> H["CombinedKernel
    Cộng hoặc nhân nhiều kernel"]
```

---

## 5. `hsic.py` — Kiểm tra tính độc lập

### hsic_gam — Gamma Approximation

```mermaid
graph TD
    A["X, Y numpy hoặc tensor"] --> B["sigma = sqrt median pairwise dist"]
    B --> C["K = center RBF X,X,sigma_x"]
    B --> D["L = center RBF Y,Y,sigma_y"]
    C --> E["test_stat = sum K*L / n"]
    D --> E
    E --> F["Fit Gamma alpha, beta dưới H0"]
    F --> G["p_value = 1 - CDF test_stat"]
```

### hsic_perm — Permutation Test

```mermaid
graph TD
    A["X, Y"] --> B["stat gốc = hsic_gam X, Y"]
    B --> C["Lặp 500 lần: shuffle Y, tính HSIC"]
    C --> D["p = count perm >= gốc / 500"]
```

---

## 6. `analysis.py` — Quy trình suy diễn hướng nhân quả

Hàm `ANMMM_cd` thực hiện Fixed-Structure Hypothesis Testing.

```mermaid
sequenceDiagram
    participant U as User
    participant A as ANMMM_cd
    participant M1 as Model 1
    participant M2 as Model 2
    participant H as hsic_gam

    U->>A: data N x 2
    Note over A: X = col 0, Y = col 1

    Note over A,M1: Giả thuyết 1: X gây Y
    A->>M1: W_dag = [[0,1],[0,0]], khóa
    A->>M1: fit(X, Y, 200 epochs)
    A->>M1: get_residuals
    A->>M1: check_stability
    A->>H: hsic_gam(res1, X)
    H-->>A: stat1

    Note over A,M2: Giả thuyết 2: Y gây X
    A->>M2: W_dag = [[0,0],[1,0]], khóa
    A->>M2: fit(Y, X, 200 epochs)
    A->>M2: get_residuals
    A->>M2: check_stability
    A->>H: hsic_gam(res2, Y)
    H-->>A: stat2

    Note over A: score = stat * (1 + stab*0.5)
    alt score1 < score2
        A-->>U: 1 là X gây Y
    else
        A-->>U: -1 là Y gây X
    end
```

---

## 7. `trainer.py` — Vòng lặp huấn luyện

```mermaid
sequenceDiagram
    participant U as User
    participant F as CausalFlow.fit
    participant T as Trainer
    participant C as Core

    U->>F: fit(X, epochs=200, lr=2e-3)
    F->>T: AdamW lr=2e-3, wd=1e-2
    F->>T: train(X, batch_size=64)

    loop epoch 1 đến 200
        Note over T: temp = max(0.5, 1 - epoch/200)
        loop mỗi batch
            T->>C: forward(batch, temp)
            C-->>T: total_loss, reg, hsic
            T->>T: backward + step
        end
        opt epoch chia hết 50
            T->>T: in log loss
        end
    end
    T-->>F: history
```

---

## 8. `causalflow.py` — CausalFlow class API

### `__init__`

```mermaid
graph TD
    A["CausalFlow(x_dim, lda, data)"] --> B{data?}
    B -- có --> C["infer x_dim → tạo Core → fit()"]
    B -- không --> D["Core = None, chờ fit sau"]
```

### `fit()`

```mermaid
graph TD
    A["fit(X, Y, epochs, lr)"] --> B{Core đã tạo?}
    B -- chưa --> C["Tạo Core từ X.shape"]
    C --> D["Trainer.train(X)"]
    B -- rồi --> D
```

### `predict_direction()`

```mermaid
graph TD
    A["predict_direction(data)"] --> B{data?}
    B -- có --> C["Gọi ANMMM_cd(data, lda)"]
    B -- không --> D["So W_dag: W 0,1 vs W 1,0"]
```

### `get_dag_matrix()`

```mermaid
graph TD
    A["threshold = 0.1"] --> B["W = W_dag.detach.numpy"]
    B --> C["W_bin = abs W > threshold"]
```

### `get_residuals()`

```mermaid
graph TD
    A["X"] --> B["MLP(X) → z_soft"]
    B --> C["phi = RFF_z * RFF_x → y_pred"]
    C --> D["res = pnl_transform(X) - y_pred"]
```

### `check_stability()`

```mermaid
graph TD
    A["X, n_splits=3"] --> B["Chia X thành 3 phần"]
    B --> C["Tính loss trên mỗi phần"]
    C --> D["score = std losses / mean losses"]
```

### `predict_counterfactual()`

```mermaid
graph TD
    A["x_orig, y_orig, x_new"] --> B["pred_orig = GP_head(x_orig, y_orig)"]
    B --> C["pred_new = GP_head(x_new, y_orig)"]
    C --> D["y_cf = y_orig - pred_orig + pred_new"]
```

