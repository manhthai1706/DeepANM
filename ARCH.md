# CausalFlow: Chi tiết Kiến trúc

Tài liệu mô tả luồng hoạt động bên trong của từng file và sơ đồ tổng thể của hệ thống.

---

## 1. Sơ đồ tổng thể hệ thống

```mermaid
graph TD
    subgraph Preprocessing [Tiền xử lý - ngoài model]
        RAW[Dữ liệu thô numpy array] --> QT[QuantileTransformer]
        QT --> IF[IsolationForest - loại outlier 5%]
    end

    subgraph CausalFlowClass [causalflow.py - CausalFlow class]
        IF --> INIT["__init__(data=X) hoặc fit(X)"]
        INIT --> CORE_INIT[Khởi tạo GPPOMC_lnhsic_Core]
        CORE_INIT --> TRAINER[CausalFlowTrainer.train]

        subgraph CoreModule [gppom_hsic.py - GPPOMC_lnhsic_Core]
            TRAINER --> FWD[forward: batch_data]
            FWD --> MLP_CALL[MLP backbone]
            FWD --> MASK["masked_input = data @ |W_dag|"]
            MLP_CALL --> Z_SOFT[z_soft: phân cụm cơ chế]
            MLP_CALL --> MU_VAR["mu, log_var: dự đoán"]
            MLP_CALL --> PNL_T[pnl_transform: biến đổi PNL]
            MASK --> RFF_X["gp_phi_x(masked_input)"]
            Z_SOFT --> RFF_Z["gp_phi_z(z_soft)"]
            RFF_X --> PHI["phi = phi_z * phi_x"]
            RFF_Z --> PHI
            PHI --> Y_PRED["y_pred = linear_head(phi)"]
        end

        subgraph LossCalc [Hàm mất mát tổng hợp]
            Y_PRED --> L_REG["L_reg = MSE(y_pred, data)"]
            PNL_T --> L_HSIC_PNL["L_hsic_pnl = FastHSIC(data, h_y - y_pred)"]
            Z_SOFT --> L_HSIC_CLU["L_hsic_clu = FastHSIC(data, z_soft)"]
            MLP_CALL --> L_KL["L_kl = KL divergence VAE"]
            CORE_INIT --> L_DAG["L_dag = tr(exp(W*W)) - d"]
            L_REG --> TOTAL["L = L_reg + 2*L_dag + λ*log(L_hsic_clu)
            + 3*log(L_hsic_pnl) + 0.2*L_kl"]
            L_HSIC_PNL --> TOTAL
            L_HSIC_CLU --> TOTAL
            L_KL --> TOTAL
            L_DAG --> TOTAL
        end

        TOTAL --> BACKWARD[backward + AdamW step]

        subgraph UserAPI [API cho người dùng]
            BACKWARD --> GET_DAG["get_dag_matrix() → W, W_binary"]
            BACKWARD --> PRED_DIR["predict_direction(data)"]
            BACKWARD --> PRED_CF["predict_counterfactual(x_orig, y_orig, x_new)"]
            BACKWARD --> CHK_STAB["check_stability(X, n_splits)"]
        end
    end
```

---

## 2. `mlp.py` — Backbone mạng nơ-ron

File này chứa toàn bộ kiến trúc trích xuất đặc trưng.

```mermaid
graph TD
    subgraph MLP_Class [MLP - forward pass]
        INPUT["x: (batch, input_dim)"] --> BACKBONE

        subgraph BACKBONE [MultivariateCausalBackbone]
            INPUT2["x"] --> GATE["Gate: Linear→GELU→Linear→Sigmoid"]
            INPUT2 --> MUL["gated_x = x * gate(x)"]
            GATE --> MUL
            MUL --> RES1[ResBlock 1]
            RES1 --> RES2[ResBlock 2]
            RES2 --> RES3[ResBlock 3]
        end

        subgraph ResBlockDetail [Mỗi ResBlock]
            RB_IN[x] --> RB_BLOCK["Linear→LayerNorm→GELU→Dropout→Linear→LayerNorm"]
            RB_IN --> RB_ADD["output = GELU(x + block(x))"]
            RB_BLOCK --> RB_ADD
        end

        RES3 --> HEAD_VAE

        subgraph HEAD_VAE [VAE Head]
            FEAT[feat] --> Z_MEAN["z_mean = Linear(feat)  → n_clusters"]
            FEAT --> Z_LOGVAR["z_logvar = Linear(feat) → n_clusters"]
            Z_MEAN --> REPARAM["z = mu + eps * exp(0.5*logvar)"]
            Z_LOGVAR --> REPARAM
            REPARAM --> SOFTMAX["z_soft = softmax(z / temperature)"]
        end

        subgraph HEAD_REG [Regression Head]
            FEAT2[feat] --> REG_OUT["regressor = Linear(feat) → output_dim * 2"]
            REG_OUT --> CHUNK["chunk → mu, log_var"]
        end

        subgraph HEAD_NSF [Neural Spline Flow]
            NOISE_IN["randn_like(mu)"] --> SPLINE["MonotonicSplineLayer"]
            SPLINE --> NOISE_OUT["noise_complex"]
        end

        subgraph HEAD_PNL [PNL Transform]
            MU_IN[mu] --> INV["InvertibleLayer: softplus(w)*x + b"]
            INV --> Y_TRANS["y_trans"]
        end
    end

    subgraph Output [Dict output]
        SOFTMAX --> OUT_Z["z_soft"]
        CHUNK --> OUT_MU["mu, log_var"]
        HEAD_VAE --> OUT_KL["kl_loss = -0.5 * sum(1+logvar-mu²-exp(logvar))"]
        NOISE_OUT --> OUT_NOISE["noise_complex"]
        Y_TRANS --> OUT_YTRANS["y_trans"]
    end
```

---

## 3. `gppom_hsic.py` — Module tối ưu hóa DAG

File này kết nối backbone MLP với các ràng buộc nhân quả.

```mermaid
graph TD
    subgraph GPPOMC [GPPOMC_lnhsic_Core]
        subgraph Params [Tham số học được]
            W_DAG["W_dag: Parameter(d, d) — ma trận kề"]
            MLP_MOD[MLP backbone]
            RFF_PHI_Z["gp_phi_z: RFFGPLayer(n_clusters, 128)"]
            RFF_PHI_X["gp_phi_x: RFFGPLayer(d, 128)"]
            LIN_HEAD["linear_head: Linear(128, d)"]
            FAST_H1["fast_hsic: FastHSIC(d, n_clusters)"]
            FAST_H2["pnl_hsic: FastHSIC(d, d)"]
        end

        subgraph Forward [forward pass]
            DATA["batch_data (batch, d)"] --> MLP_MOD
            MLP_MOD --> OUT_Z2["z_soft, kl_loss"]
            MLP_MOD --> OUT_MU2["mu, log_var"]

            DATA --> MASK2["masked_input = data @ abs(W_dag)"]
            MASK2 --> RFF_PHI_X
            OUT_Z2 --> RFF_PHI_Z
            RFF_PHI_X --> PHI2["phi = phi_z(z) * phi_x(masked)"]
            RFF_PHI_Z --> PHI2
            PHI2 --> LIN_HEAD
            LIN_HEAD --> Y_P2["y_pred_gp"]
        end

        subgraph Losses [Các thành phần loss]
            Y_P2 --> MSE["loss_reg = MSE(y_pred, data)"]

            W_DAG --> DAG_PEN["loss_dag = tr(exp(W⊙W)) - d"]

            DATA --> PNL_RES["res_pnl = pnl_transform(data) - y_pred"]
            Y_P2 --> PNL_RES
            PNL_RES --> HSIC_PNL["loss_hsic_pnl = pnl_hsic(data, res_pnl)"]

            DATA --> HSIC_CLU["loss_hsic_clu = fast_hsic(data, z_soft)"]
            OUT_Z2 --> HSIC_CLU

            OUT_MU2 --> KL2["kl_loss từ VAE"]
        end

        MSE --> TOTAL2["total = reg + 2*dag + λ*log(hsic_clu) + 3*log(hsic_pnl) + 0.2*kl"]
        DAG_PEN --> TOTAL2
        HSIC_PNL --> TOTAL2
        HSIC_CLU --> TOTAL2
        KL2 --> TOTAL2
    end
```

---

## 4. `kernels.py` — Thư viện kernel khả vi

Các kernel dùng cho Gaussian Process, tất cả đều là `nn.Module` với tham số học được.

```mermaid
graph TD
    subgraph Kernels [Kernel Library]
        RBF["RBFKernel
        K = α * exp(-0.5γ||x-y||²)
        Hỗ trợ ARD (mỗi chiều 1 gamma)"]

        LIN["LinearKernel
        K = α * xᵀy + bias"]

        POLY["PolynomialKernel
        K = (α * xᵀy + bias)^order"]

        MATERN["MaternKernel
        ν=1.5 hoặc 2.5
        Học được: log_alpha, log_gamma"]

        RQ["RationalQuadraticKernel
        K = α * (1 + ||x-y||²/(2α_rk*l²))^(-α_rk)"]

        COMB["CombinedKernel
        Cộng hoặc nhân nhiều kernel"]
    end

    subgraph Shared [Tham số chung]
        LOG_A["log_alpha: nn.Parameter — biên độ"]
        LOG_G["log_gamma: nn.Parameter — bandwidth"]
    end

    Shared --> RBF
    Shared --> MATERN
    Shared --> RQ
```

---

## 5. `hsic.py` — Kiểm tra tính độc lập

Dùng trong bước cuối của `ANMMM_cd` để so sánh HSIC giữa hai giả thuyết.

```mermaid
graph TD
    subgraph HSIC_GAM [hsic_gam - Gamma Approximation]
        X_IN["X, Y: numpy hoặc tensor"] --> SIGMA["Tính sigma = median heuristic"]
        SIGMA --> K_MAT["K = RBF(X,X,σ_x) → center"]
        SIGMA --> L_MAT["L = RBF(Y,Y,σ_y) → center"]
        K_MAT --> STAT["test_stat = sum(K*L) / n"]
        L_MAT --> STAT
        STAT --> GAMMA_FIT["Fit Gamma(α,β) dưới H0"]
        GAMMA_FIT --> P_VAL["p_value = 1 - CDF(test_stat)"]
    end

    subgraph HSIC_PERM [hsic_perm - Permutation Test]
        X_IN2["X, Y"] --> BASE["Tính HSIC gốc"]
        BASE --> LOOP["Lặp 500 lần: shuffle Y, tính HSIC"]
        LOOP --> P_VAL2["p_value = count(perm >= gốc) / 500"]
    end
```

---

## 6. `analysis.py` — Quy trình suy diễn hướng nhân quả

Hàm `ANMMM_cd` thực hiện Fixed-Structure Hypothesis Testing.

```mermaid
sequenceDiagram
    participant User
    participant ANMMM as ANMMM_cd(data, lda)
    participant CF1 as CausalFlow 1
    participant CF2 as CausalFlow 2
    participant HSIC as hsic_gam

    User->>ANMMM: data (N, 2)
    Note over ANMMM: Tách X = data[:,0], Y = data[:,1]

    ANMMM->>CF1: Tạo CausalFlow, khóa W_dag = [[0,1],[0,0]]
    ANMMM->>CF1: fit(X, Y, epochs=200)
    ANMMM->>CF1: get_residuals → res1
    ANMMM->>CF1: check_stability → stab1
    ANMMM->>HSIC: hsic_gam(res1[:,1], X) → stat1

    ANMMM->>CF2: Tạo CausalFlow, khóa W_dag = [[0,0],[1,0]]
    ANMMM->>CF2: fit(Y, X, epochs=200)
    ANMMM->>CF2: get_residuals → res2
    ANMMM->>CF2: check_stability → stab2
    ANMMM->>HSIC: hsic_gam(res2[:,1], Y) → stat2

    Note over ANMMM: score1 = stat1 * (1 + stab1*0.5)
    Note over ANMMM: score2 = stat2 * (1 + stab2*0.5)
    alt score1 < score2
        ANMMM-->>User: return 1 (X→Y)
    else score2 <= score1
        ANMMM-->>User: return -1 (Y→X)
    end
```

---

## 7. `trainer.py` — Vòng lặp huấn luyện

```mermaid
sequenceDiagram
    participant User
    participant CF as CausalFlow.fit()
    participant Trainer as CausalFlowTrainer
    participant Core as GPPOMC_lnhsic_Core

    User->>CF: fit(X, epochs=200, lr=2e-3)
    CF->>Trainer: Khởi tạo AdamW(lr, weight_decay=1e-2)
    CF->>Trainer: train(X, epochs, batch_size=64)

    loop epoch = 1 → 200
        Note over Trainer: temperature = max(0.5, 1.0 - epoch/200)
        loop Mỗi batch
            Trainer->>Core: forward(batch, temperature)
            Core-->>Trainer: total_loss, reg_loss, hsic_loss
            Trainer->>Trainer: loss.backward()
            Trainer->>Trainer: optimizer.step()
        end
        alt epoch % 50 == 0
            Trainer->>Trainer: In loss trung bình
        end
    end
    Trainer-->>CF: return history
```

---

## 8. `causalflow.py` — Giao diện CausalFlow class

```mermaid
graph TD
    subgraph CausalFlowAPI [CausalFlow - nn.Module]
        INIT["__init__(x_dim, y_dim, lda, data=None)"]
        INIT --> CHECK{"data != None?"}
        CHECK -- Có --> AUTO["Tự infer x_dim → khởi tạo Core → fit()"]
        CHECK -- Không --> LAZY["Core = None, chờ fit() sau"]

        FIT["fit(X, Y=None, epochs, lr)"]
        FIT --> CORE_CHECK{"Core đã tạo?"}
        CORE_CHECK -- Chưa --> DYN["Khởi tạo Core từ X.shape"]
        CORE_CHECK -- Có --> TRAIN["CausalFlowTrainer.train(X)"]
        DYN --> TRAIN

        PRED_DIR["predict_direction(data)"]
        PRED_DIR --> HAS_DATA{"data != None?"}
        HAS_DATA -- Có --> CALL_ANMMM["Gọi ANMMM_cd(data, lda)"]
        HAS_DATA -- Không --> READ_W["Đọc W_dag: W[0,1] vs W[1,0]"]

        GET_DAG["get_dag_matrix(threshold=0.1)"]
        GET_DAG --> W_RAW["W = W_dag.detach().numpy()"]
        GET_DAG --> W_BIN["W_bin = (|W| > threshold)"]

        GET_RES["get_residuals(X, use_pnl=True)"]
        GET_RES --> PASS_MLP["MLP(X) → z, phi → y_pred"]
        GET_RES --> CALC_RES["residuals = pnl_transform(X) - y_pred"]

        CHK["check_stability(X, n_splits=3)"]
        CHK --> SPLIT["Chia X thành n_splits phần"]
        CHK --> EVAL_LOSS["Tính loss trên mỗi phần"]
        CHK --> STAB_SCORE["stability = std(losses) / mean(losses)"]

        CF["predict_counterfactual(x_orig, y_orig, x_new)"]
        CF --> PRED_ORIG["y_pred_orig = GP_head(x_orig, y_orig)"]
        CF --> PRED_NEW["y_pred_new = GP_head(x_new, y_orig)"]
        CF --> Y_CF["y_cf = y_orig - y_pred_orig + y_pred_new"]
    end
```
