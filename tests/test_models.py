import numpy as np # Thư viện tính toán mảng / Array computation library
import pytest # Thư viện kiểm thử / Testing framework
import os # Thư viện giao tiếp hệ điều hành / OS interface library
import matplotlib # Thư viện vẽ biểu đồ / Plotting library
# Use non-interactive backend for testing / Sử dụng backend không tương tác để chạy kiểm thử
matplotlib.use('Agg')

# Import các mô hình và công cụ hỗ trợ / Import models and utilities
from src.models.deepanm import DeepANM
from src.models.fast_baseline import FastANM
from src.models.lite_baseline import LiteANM
from src.utils.visualize import plot_dag
from src.utils.adaptive_lasso import adaptive_lasso_dag


def test_fastanm_pipeline():
    """Verify FastANM logic works with and without RF/CI pruning
    Kiểm tra luồng hoạt động của FastANM khi có/không có bộ lọc RF/CI"""
    np.random.seed(42) # Thiết lập seed để tái lập kết quả / Set seed for reproducibility
    X = np.random.randn(30, 3) # Dữ liệu ngẫu nhiên / Random data
    
    model = FastANM()
    # Chạy mô hình cơ bản (không dùng RF/CI) / Run basic model (no RF/CI)
    W_fast = model.fit(X, verbose=False, use_rf=False, use_ci_pruning=False)
    assert W_fast.shape == (3, 3) # Kiểm tra kích thước đồ thị / Check graph shape
    
    # Test with Random Forest and Conditional Independence / Kiểm tra với Random Forest và Độc lập có điều kiện
    W_fast_rf = model.fit(X, verbose=False, use_rf=True, use_ci_pruning=True)
    assert W_fast_rf.shape == (3, 3)


def test_liteanm_pipeline():
    """Verify LiteANM overrides fit defaults properly
    Kiểm tra LiteANM ghi đè các tham số mặc định chính xác"""
    np.random.seed(42)
    X = np.random.randn(30, 3)
    
    model = LiteANM()
    # 2 epochs is enough to test if the training loop works / 2 epoch là đủ để kiểm tra vòng lặp huấn luyện
    history = model.fit(X, epochs=2, batch_size=10, verbose=False)
    assert 'loss' in history # Đảm bảo có lưu vết tổn thất / Ensure loss history is recorded
    assert len(history['loss']) == 2 # Số lượng bản ghi phải khớp với số epoch / Records must match epochs
    
    # Kiểm tra phương pháp Bootstrap để chọn cạnh ổn định / Test Bootstrap for edge stability selection
    prob, ate = model.fit_bootstrap(X, n_bootstraps=2, epochs=2, batch_size=10, verbose=False)
    assert prob.shape == (3, 3) # Ma trận xác suất / Probability matrix
    assert ate.shape == (3, 3)  # Ma trận tác động nhân quả / ATE matrix


def test_deepanm_bootstrap_and_ate():
    """Verify DeepANM bootstrap selection, ATE estimation and clustering predict
    Kiểm tra lựa chọn bootstrap, ước lượng ATE và dự báo cụm của DeepANM"""
    np.random.seed(42)
    X = np.random.randn(30, 3)
    
    model = DeepANM(n_clusters=2, hidden_dim=16) # Khởi tạo mô hình 2 cụm cơ chế / Init model with 2 clusters
    
    # Fit Bootstrap with fast mode internally / Chạy Bootstrap tích hợp chế độ discovery nhanh
    prob, ate_matrix = model.fit_bootstrap(
        X, n_bootstraps=2, epochs=2, batch_size=10, 
        verbose=False, discovery_mode="fast"
    )
    
    assert prob.shape == (3, 3)
    assert ate_matrix.shape == (3, 3)
    
    # Predict clusters based on latent mechanisms / Dự báo cụm dựa trên các cơ chế ẩn
    clusters = model.predict_clusters(X)
    assert clusters.shape == (30,) # Mỗi mẫu có 1 nhãn cụm / Each sample has 1 cluster label
    assert set(clusters).issubset({0, 1}) # Nhãn phải thuộc {0, 1} / Labels must be in {0, 1}


def test_visualize_plot_dag(tmp_path):
    """Verify plot_dag function generates output without crashing
    Kiểm tra chức năng vẽ đồ thị DAG tạo ra file mà không gây lỗi"""
    # Ma trận kề giả định / Dummy weight matrix
    W = np.array([
        [0.0, 1.5, 0.0],
        [0.0, 0.0, -0.8],
        [0.0, 0.0, 0.0]
    ])
    
    save_file = tmp_path / "test_graph.png" # Đường dẫn file tạm / Temporary file path
    
    # Chạy hàm vẽ đồ thị và lưu file / Run plotting and save file
    plot_dag(W_matrix=W, labels=['X', 'Y', 'Z'], threshold=0.1, save_path=str(save_file))
    
    assert os.path.exists(save_file) # Kiểm tra file có tồn tại / Check file existence
    assert os.path.getsize(save_file) > 0 # File không được rỗng / File must not be empty


def test_adaptive_lasso_rf_ci():
    """Verify Adaptive LASSO with Random Forest and Partial Correlation Pruning
    Kiểm tra Adaptive LASSO phối hợp với Random Forest và lọc tương quan từng phần"""
    np.random.seed(42)
    # Tạo chuỗi nhân quả / Create causal chain
    X0 = np.random.randn(50)
    X1 = X0 + np.random.randn(50) * 0.1
    X2 = X1 + np.random.randn(50) * 0.1
    X = np.column_stack([X0, X1, X2])
    
    order = [0, 1, 2] # Thứ tự topo / Topological order
    # use_rf=True invokes sklearn RF permutation importance / Kích hoạt độ quan trọng RF của sklearn
    # use_ci_pruning=True invokes HistGradientBoosting conditional tests / Kích hoạt kiểm định điều kiện qua HistGB
    W_bin = adaptive_lasso_dag(X, order, use_rf=True, use_ci_pruning=True)
    
    assert W_bin.shape == (3, 3)
    assert np.all(np.tril(W_bin) == 0) # Ràng buộc tam giác dưới phải bằng 0 / Lower triangle must be 0
