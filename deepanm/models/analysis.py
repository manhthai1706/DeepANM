"""
ANM-MM (ANM Mixture Model) Analysis Tools
Công cụ phân tích mô hình hỗn hợp ANM
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

from deepanm.models.deepanm import DeepANM
from deepanm.core.hsic import hsic_gam

def draw_clu(data, label, name):
    """Visualize clustering results / Trực quan hóa kết quả phân cụm"""
    colours = ['#F13C20','#E27D60','#BC986A','#4056A1','#D79922','#379683','#379683'] # Cluster colors / Màu đại diện cụm
    markers = ['o', 'p', 's', 'v', '^', '<', '>'] # Cluster markers / Ký hiệu điểm cụm
    label_list = np.unique(label) # Unique clusters / Danh sách cụm duy nhất
    plt.figure(figsize=(8, 6)) # Create figure / Tạo khung hình
    
    is_1d = data.shape[1] == 1 # Check if data is 1D / Kiểm tra dữ liệu 1 chiều
    
    for i, ilabel in enumerate(label_list): # Loop through labels / Lặp qua các nhãn
        cu_indices = [idx for idx, x in enumerate(label) if x == ilabel] # Get indices / Lấy chỉ số
        cu_data = data[cu_indices, :] # Extract cluster data / Trích xuất dữ liệu cụm
        
        if is_1d: # Plot 1D data / Vẽ dữ liệu 1 chiều
            y_vals = np.zeros_like(cu_data[:, 0]) # Zero Y values / Giá trị Y bằng 0
            plt.scatter(cu_data[:, 0], y_vals, c=colours[i % len(colours)], 
                        marker=markers[i % len(markers)], label=f'Cluster {ilabel}', alpha=0.6)
        else: # Plot 2D data / Vẽ dữ liệu 2 chiều
            plt.scatter(cu_data[:, 0], cu_data[:, 1], c=colours[i % len(colours)], 
                        marker=markers[i % len(markers)], label=f'Cluster {ilabel}', alpha=0.6)

    if is_1d: # Format 1D plot / Định dạng biểu đồ 1 chiều
        plt.yticks([]) # Hide Y axis / Ẩn trục Y
        plt.xlabel('Latent Variable Z') # Label X / Nhãn trục X
    
    plt.title(name) # Graph title / Tiêu đề biểu đồ
    plt.legend() # Show legend / Hiển thị chú thích
    plt.grid(True, alpha=0.2) # Show grid / Hiển thị lưới

def ANMMM_cd(data, lda):
    """Causal Direction Inference / Suy luận hướng nhân quả"""
    X = data[:,0].reshape(-1 ,1) # Extract X / Trích xuất X
    Y = data[:,1].reshape(-1 ,1) # Extract Y / Trích xuất Y
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # Select device / Chọn thiết bị

    # Test hypothesis X -> Y / Kiểm tra giả thuyết X gây ra Y
    print("\n[Testing Hypothesis: X --> Y]")
    cf1 = DeepANM(x_dim=1, y_dim=1, n_clusters=2, lda=lda, device=device)
    with torch.no_grad():
        mask1 = torch.zeros(2, 2).to(device) # Create mask / Tạo mặt nạ
        mask1[0, 1] = 1.0 # Set edge X->Y / Đặt cạnh X->Y
        cf1.core.W_dag.data = mask1 # Apply structure / Áp dụng cấu trúc
        cf1.core.W_dag.requires_grad = False # Lock DAG / Khóa đồ thị
        
    cf1.fit(X, Y, epochs=200, verbose=False) # Train / Huấn luyện
    combined1 = np.hstack([X, Y]) # Combine data / Gộp dữ liệu
    stab1, _ = cf1.check_stability(combined1) # Check stability / Kiểm tra ổn định
    res1 = cf1.get_residuals(combined1) # Get residuals / Lấy phần dư
    stat1, _, p1 = hsic_gam(res1[:, 1:2], X) # HSIC score / Điểm HSIC

    # Test hypothesis Y -> X / Kiểm tra giả thuyết Y gây ra X
    print("[Testing Hypothesis: Y --> X]")
    cf2 = DeepANM(x_dim=1, y_dim=1, n_clusters=2, lda=lda, device=device)
    with torch.no_grad():
        mask2 = torch.zeros(2, 2).to(device) # Create mask / Tạo mặt nạ
        mask2[1, 0] = 1.0 # Set edge Y->X / Đặt cạnh Y->X
        cf2.core.W_dag.data = mask2 # Apply structure / Áp dụng cấu trúc
        cf2.core.W_dag.requires_grad = False # Lock DAG / Khóa đồ thị
        
    cf2.fit(Y, X, epochs=200, verbose=False) # Train / Huấn luyện
    combined2 = np.hstack([Y, X]) # Combine data / Gộp dữ liệu
    stab2, _ = cf2.check_stability(combined2) # Check stability / Kiểm tra ổn định
    res2 = cf2.get_residuals(combined2) # Get residuals / Lấy phần dư
    stat2, _, p2 = hsic_gam(res2[:, 1:2], Y) # HSIC score / Điểm HSIC

    # Scoring and decision / Chấm điểm và đưa ra quyết định
    print(f"X->Y HSIC: {stat1:.6f} | Y->X HSIC: {stat2:.6f}")
    score1 = stat1 * (1.0 + stab1 * 0.5) # Final score X->Y / Điểm cuối X->Y
    score2 = stat2 * (1.0 + stab2 * 0.5) # Final score Y->X / Điểm cuối Y->X
    
    if score1 < score2: # Lower score wins / Điểm thấp hơn thắng
        return 1, cf1 # Returns X->Y / Trả về X->Y
    else:
        return -1, cf2 # Returns Y->X / Trả về Y->X

def ANMMM_clu(data, label_true, ilda):
    """Mechanism Clustering / Phân cụm cơ chế nhân quả"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = data[:,0].reshape(-1 ,1)
    Y = data[:,1].reshape(-1 ,1)
    nclu = len(np.unique(label_true)) # Target cluster count / Số cụm mục tiêu

    # Initialize and fit DeepANM / Khởi tạo và chạy DeepANM
    print(f"\n--- End-to-End Clustering (lda={ilda}, clusters={nclu}) ---")
    cf = DeepANM(x_dim=1, y_dim=1, n_clusters=nclu, lda=ilda, device=device)
    cf.fit(X, Y, epochs=200)

    # Predict categorical labels / Dự báo nhãn phân loại
    combined = np.hstack([X, Y])
    clu_label = cf.predict_clusters(combined) # Final labels / Nhãn cuối cùng

    ari = metrics.adjusted_rand_score(label_true, clu_label) # Compute ARI / Tính chỉ số ARI
    print(f'\nARI of DeepANM: {ari:.4f}')

    # Visualizations / Trực quan hóa
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(device)
        Y_tensor = torch.from_numpy(Y).float().to(device)
        xy = torch.cat([X_tensor, Y_tensor], dim=1)
        # Mechanism space / Không gian cơ chế
        z_viz = cf.core.MLP(xy)['z_soft'].cpu().numpy()

    draw_clu(z_viz, label_true, 'Latent Space (Mechanism Space)') # Draw Z space / Vẽ KG Z
    draw_clu(data, clu_label, 'Final Clustering Results') # Draw Results / Vẽ kết quả
    plt.show()

    return clu_label # Result / Kết quả
