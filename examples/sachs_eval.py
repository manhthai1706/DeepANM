import sys
import os
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deepanm import DeepANM

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
    data_norm = (data - data_mean) / (data_std + 1e-8)
    
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
            
    # 3. Model Training Siêu Tối Ưu (Lightweight & Fast)
    model = DeepANM(
        n_clusters=2,     # 2 cụm cơ chế là đủ để bắt Heteroscedasticity, chạy cực nhanh
        hidden_dim=32,    # Bóp nhỏ mạng MLP do data chỉ có 11 chiều, tránh over-fitting nhiễu
        lda=0.2,          # Tăng trọng số ép độc lập HSIC lên một chút để bù đắp việc giảm capacity
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    n_bootstraps = 8      # Tăng số vòng bootstrap để lọc cạnh nhiễu (Stability Selection)
    print(f"Starting {n_bootstraps} Bootstraps ALM Training (Lightweight mode)...")
    
    prob_matrix, avg_W_norm = model.fit_bootstrap(
        data_norm,               
        n_bootstraps=n_bootstraps,
        threshold=0.01,   # Loại bỏ mũi tên quá mỏng sớm để tập trung ATE
        epochs=150,       # Thuật toán ALM hội tụ rất nhanh, không cần tới 300 epochs
        lr=1e-2,          # Đẩy tốc độ học lên cao do mạng đã thu nhỏ
        batch_size=128,   # Tăng batch size tối ưu mảng GPU
        verbose=True 
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
    
if __name__ == '__main__':
    run_sachs_evaluation()
