import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Đảm bảo import được module `deepanm` từ thư mục gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import DeepANM, plot_dag

def generate_nonlinear_data(n_samples=2000):
    """
    Sinh Cấu trúc Đồ thị (DAG) Phi tuyến tính phức tạp
    X1 -> X2
    X1 -> X3
    X2 -> X4
    X3 -> X4
    (Mô phỏng 1 viên kim cương: X1 sinh ra X2, X3. X2, X3 sinh ra X4)
    """
    np.random.seed(42)
    
    # X1: Biến Độc lập gốc (Uniform, Không phải Gaussian)
    X1 = np.random.uniform(-3, 3, n_samples)
    
    # X2: Phụ thuộc X1 qua hàm bậc 3 (Cubic) + Nhiễu Gamma (Lệch)
    noise_X2 = np.random.gamma(shape=2.0, scale=0.5, size=n_samples) - 1.0
    X2 = 0.5 * (X1 ** 3) + noise_X2
    
    # X3: Phụ thuộc X1 qua hàm Lượng giác (Sin) + Nhiễu Gaussian
    noise_X3 = np.random.normal(0, 0.5, n_samples)
    X3 = 2.0 * np.sin(X1 * 2) + noise_X3
    
    # X4: Phụ thuộc X2 (Hàm Mũ) và X3 (Tuyến tính) + Nhiễu Mix (Bimodal)
    noise_X4 = np.choose(np.random.binomial(1, 0.5, n_samples), [
        np.random.normal(-2, 0.5, n_samples),
        np.random.normal(2, 0.5, n_samples)
    ])
    X4 = np.exp(-np.abs(X2)) * X3 + noise_X4
    
    # X5: Biến Rác (Nhiễu loạn hệ thống, độc lập hoàn toàn)
    X5 = np.random.randn(n_samples) * 2
    
    # Trả về ma trận tổng hợp [N, 5]
    return np.column_stack((X1, X2, X3, X4, X5))

def run_synthetic_test():
    print("="*60)
    print(" BÀI TEST CHINH PHỤC DỮ LIỆU PHI TUYẾN (NON-LINEAR)")
    print(" Cấu trúc ẩn: X1 -> X2, X1 -> X3, X2 -> X4, X3 -> X4. X5 độc lập.")
    print("="*60)
    
    data = generate_nonlinear_data(2000)
    labels = ["X1 (Root)", "X2 (Cubic)", "X3 (Sin)", "X4 (Mixed)", "X5 (Noise)"]
    
    # Ma Trận Ground Truth Sự thật
    W_true = np.zeros((5, 5))
    W_true[0, 1] = 1 # X1 -> X2
    W_true[0, 2] = 1 # X1 -> X3
    W_true[1, 3] = 1 # X2 -> X4
    W_true[2, 3] = 1 # X3 -> X4
    
    print("\n[Bước 1] Khởi tạo mô hình DeepANM siêu tốc...")
    model = DeepANM(
        n_clusters=3,    # Dùng 3 cụm GMM để xử lý dứt điểm nhiễu Bimodal của X4
        hidden_dim=64,   # Tăng độ dày mạng lên 64 để hấp thụ hàm Sin và Mũ (Exp)
        lda=0.2          # Cường độ độc lập HSIC
    )
    
    print("[Bước 2] Bắt đầu Training và Lọc cơ chế (Bootstrap)...")
    import time
    start_time = time.time()
    
    # KHÔNG dùng Quantile/Isolation để ép hệ thống dùng sức mạnh DECI Flow GMM nguyên thủy
    prob_matrix, avg_W_norm = model.fit_bootstrap(
        data,               
        n_bootstraps=5,
        epochs=200,      
        lr=5e-3,          
        batch_size=256,
        verbose=True,
        apply_quantile=True,  # Bật Quantile Transform để dẹp ngay chênh lệch Scale
        apply_isolation=True  # Bật Isolation Forest dẹp Outlier từ Sinh ngẫu nhiên Hàm mũ
    )
    training_time = time.time() - start_time
    print(f"\n=> Training hoàn tất trong {training_time:.2f} giây.")
    
    # Lọc Cạnh Nhị phân bằng ngưỡng 40% (xuất hiện ít nhất 2/5 lần bootstrap)
    W_pred = (prob_matrix >= 0.4).astype(int)
    
    print("\n[Kết quả Ma trận Dự đoán (W_pred)]")
    print(W_pred)
    
    print("\n[Ground Truth Sự thật (W_true)]")
    print(W_true)
    
    # Đếm số cạnh đoán trúng
    correct_edges = np.sum((W_pred == 1) & (W_true == 1))
    false_positives = np.sum((W_pred == 1) & (W_true == 0))
    false_negatives = np.sum((W_pred == 0) & (W_true == 1))
    
    print("\n[Đánh giá]")
    print(f" - Cạnh đúng (True Positive): {correct_edges}/4")
    print(f" - Cạnh sai lầm (False Positive): {false_positives}")
    print(f" - Cạnh bỏ sót (False Negative): {false_negatives}")

    # -------- VẼ ĐỒ THỊ ---------
    print("\n[Trực Quan Hóa]")
    import matplotlib.pyplot as plt
    os.makedirs('results', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("DeepANM Discovery vs Phi Tuyến Tính (Sin, Cubic, Exp)", fontsize=16, fontweight='bold')
    
    plot_dag(
        W_matrix=prob_matrix, 
        labels=labels,
        title="[Predicted] DeepANM (Bootstrap Prob)",
        threshold=0.3, ax=ax1, node_size=2500, font_size=10
    )
    
    plot_dag(
        W_matrix=W_true, 
        labels=labels,
        title="[Ground Truth] Sự thật",
        threshold=0.5, ax=ax2, node_size=2500, font_size=10
    )

    plt.tight_layout()
    plt.savefig("results/synthetic_nonlinear.png", dpi=150)
    plt.close()
    print("=> Ảnh báo cáo đã lưu tại: results/synthetic_nonlinear.png")

if __name__ == "__main__":
    run_synthetic_test()
