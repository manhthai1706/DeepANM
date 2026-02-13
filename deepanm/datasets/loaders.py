import numpy as np
import os
import requests
from io import BytesIO

def load_sachs():
    """
    Load Sachs dataset. 
    Ưu tiên load từ local folder 'data/sachs' (nếu người dùng git clone) 
    Fallback tải từ GitHub (nếu người dùng pip install).
    """
    # 1. Thử load từ local path (giả sử cấu trúc repo)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Đi lên 2 cấp: deepanm/datasets/loaders.py -> deepanm/datasets -> deepanm -> root
    repo_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    local_data_path = os.path.join(repo_root, 'data', 'sachs')
    
    if os.path.exists(local_data_path):
        try:
            data = np.load(os.path.join(local_data_path, 'continuous', 'data1.npy'))
            headers = np.load(os.path.join(local_data_path, 'sachs-header.npy'))
            return data, headers
        except Exception:
            pass # Fallback to GitHub

    # 2. Tải từ GitHub
    # Nếu người dùng chưa đổi tên repo trên GitHub, thử cả 2 URL
    urls = [
        "https://raw.githubusercontent.com/manhthai1706/DeepANM/main/data/sachs",
        "https://raw.githubusercontent.com/manhthai1706/CausalFlow/main/data/sachs"
    ]
    
    for base_url in urls:
        try:
            data_resp = requests.get(f"{base_url}/continuous/data1.npy")
            header_resp = requests.get(f"{base_url}/sachs-header.npy")
            
            if data_resp.status_code == 200 and header_resp.status_code == 200:
                data = np.load(BytesIO(data_resp.content))
                headers = np.load(BytesIO(header_resp.content))
                return data, headers
        except Exception:
            continue

    print("!! Error: Không thể tải dữ liệu. Vui lòng kiểm tra kết nối mạng hoặc folder 'data/sachs'.")
    return None, None

def load_tubingen_pair(pair_id=1):
    """Load cặp dữ liệu từ Tübingen Cause-Effect Pairs"""
    # Tübingen thường tải Online trực tiếp nên không cần local fallback phức tạp
    # Nhưng ta cần import đúng loader từ package nếu có thể, hoặc dùng helper cục bộ
    try:
        from data.tubingen_loader import TubingenLoader
        loader = TubingenLoader()
        return loader.load_pair(pair_id)
    except ImportError:
        # Fallback nếu folder data không nằm trong python path
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        if repo_root not in sys.path:
            sys.path.append(repo_root)
        from data.tubingen_loader import TubingenLoader
        loader = TubingenLoader()
        return loader.load_pair(pair_id)
