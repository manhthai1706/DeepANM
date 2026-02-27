"""Debug: xem TopoSort tìm ra thứ tự nào trên synthetic_demo data"""
import sys
sys.path.insert(0, '.')
import numpy as np
from deepanm.core.toposort import hsic_greedy_order

np.random.seed(42)
n = 500

# Sinh lại dữ liệu giống synthetic_demo.py
def gen_noise(noise_type, n):
    if noise_type == 'gaussian': return np.random.normal(0, 0.5, n)
    if noise_type == 'gamma': return np.random.gamma(2, 0.5, n) - 1.0
    if noise_type == 'bimodal': return np.random.normal(-1, 0.3, n) * (np.random.rand(n) > 0.5) + np.random.normal(1, 0.3, n) * (np.random.rand(n) <= 0.5)

X1 = gen_noise('gaussian', n)
X2 = X1**3 + gen_noise('gamma', n)
X3 = np.sin(X1) + gen_noise('bimodal', n)
X4 = np.exp(np.clip(X2 * 0.5, -5, 5)) + X3 * 0.7 + gen_noise('gaussian', n)
X5 = gen_noise('gaussian', n)

data = np.column_stack([X1, X2, X3, X4, X5])
print("Ground Truth: X0→X1, X0→X2, X1→X3, X2→X3, X4 độc lập")
print("(0-indexed: 0→1, 0→2, 1→3, 2→3)")
print()

order = hsic_greedy_order(data, verbose=True)
print()
print(f"DISCOVERED ORDER: {' → '.join(f'X{i}' for i in order)}")
print(f"(0-indexed causal_order list: {order})")
print()
print("PHÂN TÍCH: Nếu đúng, thứ tự phải có X0 trước X1, X2, và X1/X2 trước X3. X4 bất kỳ.")
