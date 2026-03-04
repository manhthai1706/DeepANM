import torch # Thư viện tính toán Tensor / Tensor computation library
import numpy as np # Thư viện tính toán mảng / Array computation library
import pytest # Thư viện kiểm thử / Testing framework

# Import các thành phần nội bộ của mô hình / Import internal model components
from src.core.mlp import MLP, HeterogeneousNoiseModel, Encoder, ANM_SEM, Decoder
from src.core.gppom_hsic import GPPOMC_lnhsic_Core, FastHSIC
from src.core.toposort import hsic_greedy_order
from src.utils.adaptive_lasso import adaptive_lasso_dag
from src.models.deepanm import DeepANM

def test_mlp_components():
    """Verify input/output shapes of internal MLP components (Encoder, SEM, Decoder, MNN)
    Kiểm tra kích thước đầu vào/đầu ra của các thành phần MLP (Encoder, SEM, Decoder, MNN)"""
    batch_size = 10 # Kích thước lô / Batch size
    input_dim = 5 # Số lượng biến / Number of variables
    n_clusters = 3 # Số lượng cụm cơ chế / Number of mechanism clusters
    
    # Check Encoder & Gumbel Selection / Kiểm tra bộ mã hóa và lựa chọn Gumbel
    encoder = Encoder(input_dim, hidden_dim=16, n_mechanisms=n_clusters)
    x = torch.randn(batch_size, input_dim) # Dữ liệu giả / Dummy data
    feat, z_soft, kl_loss = encoder(x) # Chạy forward / Run forward pass
    assert z_soft.shape == (batch_size, n_clusters) # Kiểm tra kích thước xác suất cụm / Check cluster probability shape
    assert kl_loss.dim() == 0  # Kiểm tra tổn thất KL là một vô hướng / Check KL loss is a scalar
    
    # Check SEM (Structural Equation Model) / Kiểm tra Mô hình phương trình cấu trúc
    sem = ANM_SEM(input_dim, hidden_dim=16, output_dim=input_dim)
    mu = sem(x) # Dự báo giá trị trung bình / Predict mean values
    assert mu.shape == (batch_size, input_dim) # Kiểm tra kích thước đầu ra / Check output shape
    
    # Check Decoder (Monotonic Transformation) / Kiểm tra bộ giải mã (Biến đổi đơn điệu)
    decoder = Decoder(input_dim)
    y_trans = decoder(x) # Biến đổi thuận / Forward transformation
    assert y_trans.shape == (batch_size, input_dim) # Kiểm tra kích thước / Check shape
    x_inv = decoder.inverse(y_trans) # Biến đổi nghịch / Inverse transformation
    # Inverse should ideally get back to exactly similar space, check shapes / Nghịch đảo nên thu hồi lại không gian tương tự
    assert x_inv.shape == (batch_size, input_dim)

    # Full MLP integration / Kiểm tra tích hợp MLP toàn phần
    mlp = MLP(input_dim=input_dim, hidden_dim=32, output_dim=input_dim, n_clusters=n_clusters)
    out = mlp(x) # Chạy toàn bộ mạng / Run full network
    assert out['z_soft'].shape == (batch_size, n_clusters) # Kiểm tra xác suất cụm / Check cluster probabilities
    assert out['mu'].shape == (batch_size, input_dim) # Kiểm tra giá trị trung bình / Check mean values
    assert torch.allclose(out['z_soft'].sum(dim=1), torch.ones(batch_size)) # Tổng xác suất các cụm phải bằng 1 / Sum of probabilities must be 1
    
    # NLL is always computed natively using noise proxy / NLL được tính toán dựa trên nhiễu trung gian
    assert 'log_prob_noise' in out # Đảm bảo có log xác suất nhiễu / Ensure log prob noise exists
    assert out['log_prob_noise'].shape == (batch_size,) # Kiểm tra kích thước xác suất / Check probability shape


def test_heterogeneous_noise():
    """Verify DECI-inspired GMM output / Kiểm tra đầu ra GMM dựa trên cảm hứng từ DECI"""
    batch_size = 20
    dim = 4
    n_components = 5 # Số thành phần hỗn hợp / Number of mixture components
    
    # Khởi tạo mô hình nhiễu hỗn hợp / Initialize heterogeneous noise model
    noise_model = HeterogeneousNoiseModel(dim=dim, n_components=n_components)
    fake_noise = torch.randn(batch_size, dim) # Tạo nhiễu giả / Create dummy noise
    log_prob = noise_model.compute_log_prob(fake_noise) # Tính log xác suất / Compute log probability
    
    assert log_prob.shape == (batch_size,) # Kiểm tra kích thước đầu ra / Check output shape
    assert not torch.isnan(log_prob).any() # Không được chứa giá trị NaN / Should not contain NaN
    assert not torch.isinf(log_prob).any() # Không được chứa giá trị vô cùng / Should not contain Inf


def test_fast_hsic():
    """Verify RFF-based HSIC approximation / Kiểm tra xấp xỉ HSIC dựa trên RFF"""
    x_dim = 3
    z_dim = 2
    n_samples = 50
    
    # Khởi tạo bộ tính HSIC nhanh với 64 đặc trưng ngẫu nhiên / Initialize Fast HSIC with 64 random features
    fast_hsic = FastHSIC(x_dim, z_dim, n_features=64)
    X = torch.randn(n_samples, x_dim)
    Z = torch.randn(n_samples, z_dim)
    
    hsic_val = fast_hsic(X, Z) # Tính giá trị HSIC / Compute HSIC value
    assert hsic_val.item() >= 0.0 # HSIC phải không âm / HSIC must be non-negative
    assert not torch.isnan(hsic_val) # Không được là NaN / Should not be NaN


def test_dag_penalty():
    """Verify DAGMA Log-determinant DAG acyclicity penalty h(W)
    Kiểm tra hàm phạt tính chu trình h(W) của DAGMA dựa trên Log-determinant"""
    # Khởi tạo lõi GPPOM / Initialize GPPOM core
    core = GPPOMC_lnhsic_Core(x_dim=2, y_dim=0, n_clusters=2, hidden_dim=16, lda=1.0, device='cpu')
    
    # Graph 1: Empty graph is acyclic / Đồ thị rỗng không có chu trình
    core.W_val.data = torch.ones(2, 2)
    core.W_logits.data = torch.zeros(2, 2)
    assert core.get_dag_penalty(W_mask=torch.zeros(2, 2)).item() < 1e-6 
    
    # Graph 2: Chain 0 -> 1 is acyclic / Chuỗi 0 -> 1 không có chu trình
    assert core.get_dag_penalty(W_mask=torch.tensor([[0.0, 1.0], [0.0, 0.0]])).item() < 1e-6
    
    # Graph 3: Cycle 0 -> 1 -> 0 / Chu trình 0 -> 1 -> 0
    assert core.get_dag_penalty(W_mask=torch.tensor([[0.0, 1.0], [1.0, 0.0]])).item() > 0.5 # Penalty must be high / Phạt phải cao
    
    # Graph 4: Self loop (filtered by zero-diagonal) / Tự lặp (đã bị lọc bởi đường chéo bằng 0)
    assert core.get_dag_penalty(W_mask=torch.tensor([[1.0, 0.0], [0.0, 1.0]])).item() < 1e-6


def test_score_asymmetry():
    """Verify TopoSort logic using random noise / Kiểm tra tính toán TopoSort bằng nhiễu ngẫu nhiên"""
    n_vars = 4
    n_samples = 50
    X = np.random.randn(n_samples, n_vars)
    # Xác định thứ tự nhân quả bằng thuật toán tham lam HSIC / Determine causal order using greedy HSIC
    order = hsic_greedy_order(X, n_rff=64, verbose=False)
    
    # Must contain 0 to n_vars-1 / Phải chứa đủ chỉ số các biến
    assert len(order) == n_vars
    assert set(order) == set(range(n_vars))


def test_gppomc_core_forward():
    """Verify the Core Engine correctly routes variables and computes all loss components
    Kiểm tra lõi động cơ điều hướng biến chính xác và tính đủ các thành phần tổn thất"""
    batch_size = 10
    d = 4
    # Khởi tạo lõi mô hình DeepANM / Initialize DeepANM core engine
    core = GPPOMC_lnhsic_Core(x_dim=d, y_dim=0, n_clusters=2, hidden_dim=16, lda=1.0, device='cpu')
    
    batch_data = torch.randn(batch_size, d)
    # Trả về các giá trị tổn thất thành phần / Returns loss components
    base_loss, loss_reg, loss_hsic_clu, loss_nll = core(batch_data)
    
    for loss in [base_loss, loss_reg, loss_hsic_clu, loss_nll]:
        assert isinstance(loss, torch.Tensor) # Phải là Tensor của PyTorch / Must be PyTorch Tensor
        assert loss.dim() == 0  # Phải là một số vô hướng / Must be a scalar
        assert not torch.isnan(loss) # Không được là NaN / Should not be NaN


def test_global_ate_matrix():
    """Verify computation of Neural ATE Jacobian with Diagonals strictly zeroed
    Kiểm tra tính toán ma trận ATE Jacobian với đường chéo được gán bằng 0 tuyệt đối"""
    batch_size = 20
    d = 3
    mlp = MLP(input_dim=d, hidden_dim=16, output_dim=d, n_clusters=2)
    x = torch.randn(batch_size, d)
    
    # Tính toán ma trận tác động nhân quả trung bình / Compute Average Treatment Effect matrix
    ate_matrix = mlp.get_global_ate_matrix(x)
    assert ate_matrix.shape == (d, d) # Kiểm tra kích thước ma trận / Check matrix shape
    
    diagonal = torch.diag(ate_matrix) # Lấy đường chéo / Get diagonal
    assert torch.all(diagonal == 0.0) # Đường chéo phải bằng 0 (không có tự nhân quả) / Diagonal must be 0 (no self-causality)


def test_adaptive_lasso():
    """Verify Edge Selection Module works logically with given structural rules
    Kiểm tra module lựa chọn cạnh hoạt động logic với các ràng buộc cấu trúc"""
    np.random.seed(42)
    n_samples = 50
    n_vars = 3
    # Tạo chuỗi nhân quả: 0 -> 1 -> 2 / Create causal chain: 0 -> 1 -> 2
    X0 = np.random.randn(n_samples)
    X1 = X0 + np.random.randn(n_samples) * 0.1
    X2 = X1 + np.random.randn(n_samples) * 0.1
    X = np.column_stack([X0, X1, X2])
    
    order = [0, 1, 2] # Thứ tự định trước / Defined order
    # For speed in test, use OLS instead of Random Forest / Dùng OLS thay vì RF để tăng tốc kiểm thử
    W_bin = adaptive_lasso_dag(X, order, use_rf=False, use_ci_pruning=False)
    
    # Checks dimensions and content / Kiểm tra kích thước và nội dung
    assert W_bin.shape == (n_vars, n_vars)
    # Check strict topological constraint (lower triangle must be 0) / Kiểm tra ràng buộc topo (tam giác dưới phải bằng 0)
    assert np.all(np.tril(W_bin) == 0)


def test_deepanm_modes():
    """Verify Full Pipeline fits via different modes without crashing
    Kiểm tra pipeline vận hành ở các chế độ khác nhau mà không gây lỗi hệ thống"""
    d = 3
    n_samples = 40
    data = np.random.randn(n_samples, d)
    
    # ALM Mode (pure deep learning) / Chế độ ALM (thuần học sâu)
    model1 = DeepANM()
    model1.fit(data, epochs=2, batch_size=20, verbose=False, discovery_mode="alm")
    W1, W_bin1 = model1.get_dag_matrix() # Lấy ma trận kết quả / Get result matrices
    assert W1.shape == (d, d)
    
    # Fast Mode (TopoSort + ALASSO + Neural SCM gate checking) / Chế độ Fast (TopoSort + ALASSO + Lọc Neural SCM)
    model2 = DeepANM()
    model2.fit(data, epochs=2, batch_size=20, verbose=False, discovery_mode="fast")
    # Provide data for internal Jacobians computation / Cung cấp dữ liệu để tính toán Jacobian nội bộ
    ATE2, W_bin2 = model2.get_dag_matrix(X=data)
    assert ATE2.shape == (d, d)


def test_prior_constraint():
    """Verify the Exogenous prior constraint mechanism works properly to exclude incoming causal arrows
    Kiểm tra cơ chế ràng buộc ngoại sinh hoạt động chính xác để loại bỏ các mũi tên nhân quả đi vào"""
    d = 3
    model = DeepANM(x_dim=d)
    
    # By default, constraint mask allows any edge except diagonal / Mặc định mặt nạ cho phép mọi cạnh trừ đường chéo
    # Mark node 1 as Exogenous (meaning NO incoming edges) / Đánh dấu nút 1 là ngoại sinh (KHÔNG có cạnh đi vào)
    model.set_exogenous([1])
    
    W_mask = model.core.constraint_mask.cpu().numpy() # Lấy mặt nạ ràng buộc / Get constraint mask
    
    # Column 1 should be entirely 0.0 (no parent -> 1) / Cột 1 phải hoàn toàn bằng 0 (không có cha -> 1)
    assert np.all(W_mask[:, 1] == 0.0)
