"""
Lite Baseline Logic
A lighter version of the DeepANM neural network for faster execution 
on medium-sized datasets.
"""
from src.models.deepanm import DeepANM

class LiteANM(DeepANM):
    """
    DeepANM "Lite" (Học Sâu Rút Gọn).
    Vẫn sử dụng Neural Network và Phase 2, nhưng cấu hình được thu nhỏ:
    - Không dùng Gaussian Mixture Model (n_clusters = 1) -> Giảm tham số
    - Mạng nơ-ron nhỏ hơn (hidden_dim = 16)
    - Epochs mặc định thấp hơn (50 epochs)
    
    Phù hợp cho các bộ dữ liệu kích thước trung bình và khi muốn kết quả ổn định
    hơn FastANM nhưng không muốn chờ quá lâu.
    """
    def __init__(self, x_dim=None, device=None):
        # Rút gọn n_clusters=1 (Normal Noise thay vì GMM), hidden_dim=16
        super().__init__(x_dim=x_dim, n_clusters=1, hidden_dim=16, lda=0.5, device=device)
        
    def fit(self, X, epochs=50, batch_size=128, lr=5e-3, verbose=True, 
            apply_quantile=False, apply_isolation=False, _precomputed_order=None):
        """Override fit() with lighter defaults: 50 epochs, larger batch_size."""
        return super().fit(
            X, epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose,
            apply_quantile=apply_quantile, apply_isolation=apply_isolation,
            _precomputed_order=_precomputed_order
        )

    def fit_bootstrap(self, X, n_bootstraps=3, epochs=50, batch_size=128, lr=5e-3, 
                      verbose=True, apply_quantile=False, apply_isolation=False):
        """Override fit_bootstrap() with fewer bootstraps and epochs."""
        return super().fit_bootstrap(
            X, n_bootstraps=n_bootstraps, epochs=epochs, batch_size=batch_size, 
            lr=lr, verbose=verbose, apply_quantile=apply_quantile, 
            apply_isolation=apply_isolation
        )
