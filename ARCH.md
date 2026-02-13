# DeepANM: Chi tiết Kiến trúc

Tài liệu mô tả luồng hoạt động bên trong của từng file và sơ đồ tổng thể của hệ thống.

---

## 1. Sơ đồ tổng thể hệ thống

```mermaid
graph TD
    subgraph PP ["Tiền xử lý"]
        A[Dữ liệu thô] --> B[QuantileTransformer]
        B --> C[IsolationForest]
    end

    C --> D["DeepANM.fit()"]

    subgraph TRAIN ["Huấn luyện - GPPOMC_lnhsic_Core"]
        D --> E[MLP backbone]
        E --> F["z_soft — phân cụm cơ chế"]
        E --> G["mu, log_var — dự đoán"]
        D --> H["masked = data @ abs W_dag"]
        F --> I["phi = RFF_z * RFF_x"]
        H --> I
        I --> J["y_pred = linear_head phi"]
        J --> K["Loss = MSE + DAG + HSIC + KL"]
        K --> L["backward + AdamW"]
    end

    L --> M{Model đã train}

    M --> N["get_dag_matrix()"]
    M --> O["predict_direction()"]
    M --> P["predict_counterfactual()"]
    M --> Q["check_stability()"]
```

---

## 2. `mlp.py` — Backbone mạng nơ-ron

```mermaid
graph TD
    A["Input x: batch, input_dim"] --> B["Gate = Sigmoid Linear→GELU→Linear"]
    B --> C["gated_x = x * gate x"]
    C --> D[ResBlock 1]
    D --> E[ResBlock 2]
    E --> F[ResBlock 3]
    F --> G[feat]

    E -.- RB

    subgraph RB ["Chi tiết mỗi ResBlock"]
        RB1["x"] --> RB2["Linear→LN→GELU→Drop→Linear→LN"]
        RB1 --> RB3["+ residual"]
        RB2 --> RB3
        RB3 --> RB4["GELU x + block x"]
    end

    G --> H["z_mean = Linear → n_clusters"]
    G --> I["z_logvar = Linear → n_clusters"]
    H --> J["z = reparameterize mu, logvar"]
    I --> J
    J --> K["z_soft = softmax z / temp"]

    G --> L["regressor = Linear → output_dim*2"]
    L --> M["chunk → mu, log_var"]

    M --> N["MonotonicSplineLayer randn_like mu"]
    N --> O[noise_complex]

    M --> P["InvertibleLayer: softplus w * x + b"]
    P --> Q[y_trans]
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
    participant F as DeepANM.fit
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

## 8. `DeepANM.py` — DeepANM class API

```mermaid
graph TD
    A["DeepANM(x_dim, lda, data)"] --> B{data?}
    B -- có --> C["infer x_dim → tạo Core → fit()"]
    B -- không --> D["Core = None"]
    C --> E["Model đã train"]
    D --> F["fit(X, Y, epochs, lr)"]
    F --> G{Core đã tạo?}
    G -- chưa --> H["Tạo Core từ X.shape"]
    H --> I["Trainer.train X"]
    G -- rồi --> I
    I --> E

    E --> J["predict_direction(data)"]
    J --> K{data?}
    K -- có --> L["ANMMM_cd: train 2 model, so HSIC"]
    K -- không --> M["So W_dag 0,1 vs W_dag 1,0"]

    E --> N["get_dag_matrix(threshold)"]
    N --> O["W = W_dag.numpy"]
    O --> P["W_bin = abs W > thr"]

    E --> Q["get_residuals(X)"]
    Q --> R["MLP → z → phi → y_pred"]
    R --> S["res = pnl_transform X - y_pred"]

    E --> T["check_stability(X, n_splits)"]
    T --> U["Chia X, tính loss mỗi phần"]
    U --> V["score = std / mean"]

    E --> W["predict_counterfactual(x_orig, y_orig, x_new)"]
    W --> X["pred_orig = GP x_orig"]
    X --> Y["pred_new = GP x_new"]
    Y --> Z["y_cf = y - pred_orig + pred_new"]
```


