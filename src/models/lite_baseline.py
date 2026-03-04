"""
LiteANM: Lightweight variant of DeepANM for faster inference.
Reduces n_clusters=1, hidden_dim=16, and default epochs=50.
Suitable for medium-sized datasets or quick exploratory runs.

LiteANM: Biến thể nhẹ của DeepANM để thực hiện suy diễn nhanh hơn.
Giảm số cụm xuống 1, chiều ẩn còn 16 và số epoch mặc định là 50.
Phù hợp cho các bộ dữ liệu vừa hoặc chạy thử nghiệm khám phá nhanh.
"""
from src.models.deepanm import DeepANM

class LiteANM(DeepANM):
    """
    Optimized DeepANM subclass with reduced model capacity for fast prototyping.
    Lớp con của DeepANM được tối ưu hóa với dung lượng mô hình nhỏ để chạy thử nghiệm nhanh.

    Key Optimizations / Các tối ưu chính:
    - n_clusters=1: Single Gaussian noise instead of GMM mixture. / Nhiễu Gaussian đơn thay cho GMM hỗn hợp.
    - hidden_dim=16: Faster neural network backbone. / Mạng neural nền tảng nhanh hơn.
    - Fast defaults: Higher batch size, fewer training iterations. / Mặc định nhanh: Lô lớn hơn, ít vòng lặp hơn.
    """
    def __init__(self, x_dim=None, device=None):
        """Invoke DeepANM constructor with lite parameters. / Gọi hàm tạo DeepANM với các thông số nhẹ."""
        super().__init__(x_dim=x_dim, n_clusters=1, hidden_dim=16, lda=0.5, device=device)
        
    def fit(self, X, epochs=50, batch_size=128, lr=5e-3, verbose=True,
            apply_quantile=False, apply_isolation=False, discovery_mode="alm", _precomputed_order=None, **kwargs):
        """
        Execute training with fast-track defaults.
        Thực hiện huấn luyện với các giá trị mặc định được rút gọn.
        """
        return super().fit(
            X, epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose,
            apply_quantile=apply_quantile, apply_isolation=apply_isolation,
            discovery_mode=discovery_mode, _precomputed_order=_precomputed_order, **kwargs
        )

    def fit_bootstrap(self, X, n_bootstraps=3, epochs=50, batch_size=128, lr=5e-3,
                      verbose=True, apply_quantile=False, apply_isolation=False, discovery_mode="alm", **kwargs):
        """
        Run stability selection with lower bootstrap iteration count for speed.
        Chạy lựa chọn ổn định với số vòng lặp bootstrap ít hơn để đạt tốc độ.
        """
        return super().fit_bootstrap(
            X, n_bootstraps=n_bootstraps, epochs=epochs, batch_size=batch_size,
            lr=lr, verbose=verbose, apply_quantile=apply_quantile,
            apply_isolation=apply_isolation, discovery_mode=discovery_mode, **kwargs
        )
