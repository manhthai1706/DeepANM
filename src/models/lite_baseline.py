"""
LiteANM: Lightweight variant of DeepANM for faster inference.
Reduces n_clusters=1, hidden_dim=16, and default epochs=50.
Suitable for medium-sized datasets or quick exploratory runs.

LiteANM: Biến thể nhẹ của DeepANM để suy diễn nhanh hơn.
Giảm n_clusters=1, hidden_dim=16 và epochs mặc định=50.
Phù hợp cho tập dữ liệu trung bình hoặc chạy khám phá nhanh.
"""
from src.models.deepanm import DeepANM

class LiteANM(DeepANM):
    """
    DeepANM "Lite": smaller network capacity for speed.
    - n_clusters=1 (single Gaussian noise, no GMM)
    - hidden_dim=16 (fewer parameters)
    - Default 50 epochs and larger batch size

    Still uses the full Neural pipeline (discovery_mode='alm' by default).
    For best accuracy, use DeepANM with discovery_mode='fast'.

    DeepANM "Lite": dung lượng mạng nhỏ hơn để đạt tốc độ.
    - n_clusters=1 (nhiễu Gaussian đơn, không dùng GMM)
    - hidden_dim=16 (ít tham số hơn)
    - Mặc định 50 epochs và batch lớn hơn

    Vẫn dùng pipeline Neural đầy đủ (discovery_mode='alm' mặc định).
    Để đạt độ chính xác tốt nhất, dùng DeepANM với discovery_mode='fast'.
    """
    def __init__(self, x_dim=None, device=None):
        super().__init__(x_dim=x_dim, n_clusters=1, hidden_dim=16, lda=0.5, device=device)
        
    def fit(self, X, epochs=50, batch_size=128, lr=5e-3, verbose=True,
            apply_quantile=False, apply_isolation=False, discovery_mode="alm", _precomputed_order=None, **kwargs):
        """Override fit() with lighter defaults: 50 epochs, larger batch_size.
        Ghi đè fit() với mặc định nhẹ hơn: 50 epochs, batch_size lớn hơn."""
        return super().fit(
            X, epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose,
            apply_quantile=apply_quantile, apply_isolation=apply_isolation,
            discovery_mode=discovery_mode, _precomputed_order=_precomputed_order, **kwargs
        )

    def fit_bootstrap(self, X, n_bootstraps=3, epochs=50, batch_size=128, lr=5e-3,
                      verbose=True, apply_quantile=False, apply_isolation=False, discovery_mode="alm", **kwargs):
        """Override fit_bootstrap() with fewer bootstraps and lighter epochs.
        Ghi đè fit_bootstrap() với ít bootstrap hơn và epochs nhẹ hơn."""
        return super().fit_bootstrap(
            X, n_bootstraps=n_bootstraps, epochs=epochs, batch_size=batch_size,
            lr=lr, verbose=verbose, apply_quantile=apply_quantile,
            apply_isolation=apply_isolation, discovery_mode=discovery_mode, **kwargs
        )
