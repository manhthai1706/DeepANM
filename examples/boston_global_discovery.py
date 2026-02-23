import sys
import os
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deepanm import DeepANM

def run_boston_discovery():
    print("="*60)
    print(" Boston Housing Dataset - Causal Discovery Test")
    print(" (With Bootstrap Stability Selection / Lọc cạnh qua Bootstrap)")
    print("="*60)
    
    # Đọc dữ liệu Boston Housing
    df = pd.read_csv('datasets/boston-housing/data/boston-housing.continuous.txt', sep='\t')
    headers = df.columns.tolist()
    data = df.values
    
    print(f"Loaded Boston Data: {data.shape[0]} samples, {data.shape[1]} features")
    print(f"Features: {headers}")
    print("-" * 60)
    
    # Chuẩn hóa dữ liệu toàn cục (Standardization)
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    data_norm = (data - data_mean) / (data_std + 1e-8)
    
    # Khởi tạo mô hình logic không tự chạy fit
    model = DeepANM(
        n_clusters=2,     
        hidden_dim=64,    
        lda=10.0,
        device='cpu'
    )
    
    # 1. Khai báo Prior Knowledge (Các Biến Ngoại Sinh bị cấm Reverse Causation)
    exogenous_vars = ['CHAS', 'ZN', 'RAD', 'TAX', 'B']
    exog_idx = [headers.index(v) for v in exogenous_vars]
    model.set_exogenous(exog_idx)
    print(f"=> 🔒 Đã khóa cấu trúc Prior Exogenous: {exogenous_vars}\n")
    
    # Số lần Bootstrap
    n_bootstraps = 10
    
    # 2. Huấn luyện Bootstrap Tự Động từ DeepANM Engine
    prob_matrix, avg_W_norm = model.fit_bootstrap(
        data_norm,               
        n_bootstraps=n_bootstraps,
        threshold=0.005, # Nới lỏng Threshold cắt tỉa thô trong mỗi lần Bootstrap để tránh sót cạnh (False Negative)
        epochs=300, 
        lr=5e-3,          
        verbose=True 
    )
    
    print("\n" + "="*60)
    print(f" DANH SÁCH CÁC CẠNH NHÂN QUẢ TIỀM NĂNG (Xếp theo độ tự tin Bootstrap)")
    print("="*60)
    
    edges_found = 0
    # Gom danh sách để sort
    all_edges = []
    
    for i in range(len(headers)):
        for j in range(len(headers)):
            if prob_matrix[i, j] > 0.3 and i != j: # Hiển thị các rule chiếm ưu thế >30% mẫu
                # Lấy Derivative
                real_derivative = avg_W_norm[i, j] * (data_std[j] / data_std[i])
                prob_pct = prob_matrix[i, j] * 100.0
                all_edges.append((prob_pct, headers[i], headers[j], real_derivative))
                edges_found += 1
                
    # Sort theo Tự tin giảm dần
    all_edges.sort(key=lambda x: x[0], reverse=True)
    for edge in all_edges:
        print(f"  [>] {edge[1]:<10} ---> {edge[2]:<10} (Tự tin: {edge[0]:3.0f}% | Derivative: {edge[3]:8.3f})")
                
    print(f"\n=> Tổng kết: Tìm thấy {edges_found} đường nhân quả cốt lõi vững chắc.")

if __name__ == '__main__':
    run_boston_discovery()
