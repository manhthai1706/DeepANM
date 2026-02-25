import sys
import os
import numpy as np
import pandas as pd
import torch

from deepanm import DeepANM, plot_dag

def evaluate_graph(W_pred, W_true):
    """
    Evaluate structural metrics compared to true graph.
    W_pred, W_true are binary adjacency matrices (n x n).
    W[i, j] = 1 means edge from i to j.
    """
    N = W_true.shape[0]
    
    tp = 0
    fp = 0
    fn = 0
    shd = 0
    
    for i in range(N):
        for j in range(N):
            if i == j: continue
            
            p = W_pred[i, j]
            t = W_true[i, j]
            p_rev = W_pred[j, i]
            t_rev = W_true[j, i]
            
            if p == 1 and t == 1:
                tp += 1
            elif p == 1 and t == 0:
                # Phân biệt fp hoàn toàn (không có cả cạnh xuôi ngược) và reversed
                if t_rev == 1:
                    # Đây là reversed, sẽ chỉ cộng 1 mistake vào tổng SHD sau
                    pass 
                else:
                    fp += 1
            elif p == 0 and t == 1:
                if p_rev == 1:
                    pass
                else:
                    fn += 1
                    
    # Lặp lại qua ma trận tam giác trên để tính chuẩn SHD
    # SHD (Structural Hamming Distance) = Khác biệt bộ xương (thừa/thiếu) + Sai hướng (reversed)
    shd_calc = 0
    reversed_edges = 0
    
    for i in range(N):
        for j in range(i+1, N):
            p1, p2 = W_pred[i, j], W_pred[j, i]
            t1, t2 = W_true[i, j], W_true[j, i]
            
            # Skeleton
            has_edge_p = (p1 == 1 or p2 == 1)
            has_edge_t = (t1 == 1 or t2 == 1)
            
            if has_edge_p and not has_edge_t:
                shd_calc += 1 # Extra edge
            elif not has_edge_p and has_edge_t:
                shd_calc += 1 # Missing edge
            elif has_edge_p and has_edge_t:
                # Có cạnh, check xem có trùng hướng không
                if p1 != t1 or p2 != t2:
                    shd_calc += 1 # Sai hướng (reversed)
                    reversed_edges += 1

    precision = tp / (tp + fp + reversed_edges) if (tp + fp + reversed_edges) > 0 else 0
    recall = tp / (tp + fn + reversed_edges) if (tp + fn + reversed_edges) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'SHD': shd_calc,
        'TP': tp,
        'FP': fp + reversed_edges, # FP ở đây đại diện cho tất cả các cạnh dự đoán sai (bao gồm cả sai hướng)
        'FN': fn + reversed_edges, # FN đại diện cho các cạnh đúng không dự đoán được
        'Reversed': reversed_edges,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }

def run_sachs_evaluation():
    print("="*60)
    print(" Sachs (2005) Protein Network - Causal Discovery Test")
    print("="*60)
    
    # 1. Load Data
    data_path = 'datasets/sachs/data/sachs.2005.continuous.txt'
    df = pd.read_csv(data_path, sep='\t')
    headers = df.columns.tolist()
    data = df.values
    
    # Sử dụng 2000 mẫu để Cân bằng giữa Tốc độ và Độ chính xác (Tập data tổng có hơn 7400 điểm)
    data = data[:2000] 
    print(f"Loaded Sachs Data: {data.shape[0]} samples, {data.shape[1]} variables")
    print(f"Nodes: {headers}")
    
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    # 2. Extract Ground Truth
    gt_edges = [
        ("erk", "akt"), ("mek", "erk"), ("pip2", "pkc"), ("pip3", "akt"),
        ("pip3", "pip2"), ("pip3", "plc"), ("pka", "akt"), ("pka", "erk"),
        ("pka", "jnk"), ("pka", "mek"), ("pka", "p38"), ("pka", "raf"),
        ("pkc", "jnk"), ("pkc", "mek"), ("pkc", "p38"), ("pkc", "pka"),
        ("pkc", "raf"), ("plc", "pip2"), ("plc", "pkc"), ("raf", "mek")
    ]
    
    N = len(headers)
    W_true = np.zeros((N, N))
    for u, v in gt_edges:
        if u in headers and v in headers:
            W_true[headers.index(u), headers.index(v)] = 1
            
    # KHÔNG CẦN TIỀN XỬ LÝ THỦ CÔNG: Hệ thống DeepANM phiên bản mới đã tự động hóa mọi thứ!
    # Bạn chỉ cần truyền trực tiếp Dữ Liệu Thô (Raw Data) vào mô hình.
    data_raw = df.values
    
    # 3. Model Training Siêu Tối Ưu (Lightweight & Fast)
    model = DeepANM(
        n_clusters=2,     # 2 cụm cơ chế là đủ để bắt Heteroscedasticity, chạy cực nhanh
        hidden_dim=32,    # Bóp nhỏ mạng MLP do data chỉ có 11 chiều, tránh over-fitting nhiễu
        lda=0.2,          # Tăng trọng số ép độc lập HSIC lên một chút để bù đắp việc giảm capacity
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    n_bootstraps = 8      # Tăng số vòng bootstrap để lọc cạnh nhiễu (Stability Selection)
    print(f"Starting {n_bootstraps} Bootstraps ALM Training (Lightweight mode)...")
    
    # ----------------------------------------------------
    # BẬT TÍNH NĂNG TỰ ĐỘNG LỌC OUTLIERS & CHUẨN HÓA GAUSSIAN
    # ----------------------------------------------------
    prob_matrix, avg_W_norm = model.fit_bootstrap(
        data_raw,               
        n_bootstraps=n_bootstraps,
        threshold=0.01,   
        epochs=150,       
        lr=1e-2,          
        batch_size=128,   
        verbose=True,
        apply_quantile=True,  # Ép toàn bộ biến phân phối lệch về hình Quả chuông (Hỗ trợ FastHSIC)
        apply_isolation=True  # Rạch ròi 5% dữ liệu quấy rối, rác, outlier sinh học đo lường sai
    )
    
    # 4. Filter Edges (Dùng ngưỡng 30% mẫu cho Causal Graph)
    W_pred = (prob_matrix > 0.3).astype(int)
    
    # 5. Evaluate Metrics
    metrics = evaluate_graph(W_pred, W_true)
    
    print("\n" + "="*60)
    print(" GROUND TRUTH CHẤM ĐIỂM (EVALUATION METRICS)")
    print("="*60)
    print(f" * SHD (Khoảng cách cấu trúc): {metrics['SHD']}")
    print(f" * F1 Score: {metrics['F1']:.4f}")
    print(f" * TP (Đoán trúng): {metrics['TP']}")
    print(f" * FP (Cảnh báo giả / Sai hướng): {metrics['FP']}")
    print(f" * FN (Mất mát): {metrics['FN']}")
    print(f" * Cạnh Reverse (Lộn chiều): {metrics['Reversed']}")
    print(f" * Precision: {metrics['Precision']:.4f}")
    print(f" * Recall: {metrics['Recall']:.4f}")
    print("="*60)
    
    # 6. Vẽ Đồ thị Causal DAG: So sánh giữa Kết quả Dự đoán và Ground Truth
    import os
    os.makedirs('results', exist_ok=True)
    
    print("\n[Visualizer] Đang dựng File ảnh so sánh đồ thị Sachs...")
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle("DeepANM Causal Discovery - Sachs (2005) Dataset", fontsize=18, fontweight='bold', y=1.01)
    
    # Trái: Đồ thị Dự đoán (sử dụng prob_matrix mời nhất có giá trị thực sự giữa 0.0 - 1.0)
    plot_dag(
        W_matrix=prob_matrix,          # prob_matrix có giá trị tỷ lệ thực, nếu cạnh xuất hiện 30% sẽ là 0.3+
        labels=headers,
        title="[Predicted] (Bootstrap Prob)",
        threshold=0.3,                 # Ngưỡng 30% là tiêu chuẩn Stability Selection 
        ax=ax1,
        node_size=2200,
        font_size=10
    )
    
    # Phải: Ground Truth để so sánh trực quan
    plot_dag(
        W_matrix=W_true,
        labels=headers,
        title="[Ground Truth] Sachs (2005)",
        threshold=0.5,                 # Giá trị trong W_true là 0 hoặc 1, ngưỡng 0.5 để lấy cạnh thực sự
        ax=ax2,
        node_size=2200,
        font_size=10
    )

    plt.tight_layout()
    plt.savefig("results/sachs_causal_graph.png", bbox_inches='tight', dpi=200)
    plt.close()
    print("Mời bạn mở file 'results/sachs_causal_graph.png' để xem so sánh trực quan!")
    
if __name__ == '__main__':
    run_sachs_evaluation()
