"""
Synthetic Data Evaluation Test
Generates data using data_gen.py and evaluates CausalFlow's performance.
"""
import numpy as np
import torch
import sys
import os
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import IsolationForest

# Add root to path to import data_gen and causalflow
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_gen import gen_D
from deepanm import ANMMM_cd

def test_synthetic_direction():
    print("=" * 70)
    print("SYNTHETIC DATA CAUSAL DIRECTION EVALUATION")
    print("=" * 70)
    
    # 1. Configuration for synthetic data
    # [dist_x, f1_type, dist_noise, f2_type, n_points]
    # Configuration 1: Normal X, Identity mapping, Normal noise
    config_1 = [2, 0, 2, 0, 500] 
    
    print("\n[Step 1] Generating Synthetic Data...")
    print(f"Config: {config_1}")
    data, _ = gen_D([config_1])
    
    # In gen_each: return torch.cat([x, y], dim=1)
    # So True Direction is X -> Y (Index 0 -> Index 1)
    
    # 2. Preprocessing (Standard Sklearn Pipeline)
    print("[Step 2] Preprocessing Data...")
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(data), 500), random_state=42)
    iso = IsolationForest(contamination=0.05, random_state=42)
    
    data_norm = qt.fit_transform(data)
    clean_mask = iso.fit_predict(data_norm)
    data_clean = data_norm[clean_mask == 1]
    
    print(f"Data points after cleaning: {len(data_clean)}")
    
    # 3. Run Causal Discovery
    print("[Step 3] Running Causal Discovery...")
    try:
        # X is data[:,0], Y is data[:,1]. We expect direction = 1 (X->Y)
        direction, _ = ANMMM_cd(data_clean, lda=12.0)
        
        print("\n" + "-" * 40)
        if direction == 1:
            print("RESULT: CORRECT (X -> Y detected)")
            success = True
        else:
            print("RESULT: WRONG (Y -> X detected)")
            success = False
        print("-" * 40)
        
    except Exception as e:
        print(f"ERROR: {e}")
        success = False
        
    return success

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    test_synthetic_direction()
