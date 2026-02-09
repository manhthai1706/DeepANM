
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from ANMMM import ANMMM_cd_advanced

def test_sachs_causality():
    print("=== SACHS BIOLOGICAL DATASET TEST ===")
    
    # 1. Load Data
    data = np.load('c:/Users/manht/Downloads/CausalFlow/sachs/continuous/data1.npy')
    headers = np.load('c:/Users/manht/Downloads/CausalFlow/sachs/sachs-header.npy')
    
    # We'll test Plcg -> PIP3 (Known Causal Link, NO CONFOUNDERS)
    # Index 2: Plcg, Index 4: PIP3
    idx_cause = 0
    idx_effect = 1
    
    
    X = data[:, idx_cause].reshape(-1, 1)
    Y = data[:, idx_effect].reshape(-1, 1)
    
    print(f"Testing Relationship: {headers[idx_cause]} vs {headers[idx_effect]}")
    print(f"Ground Truth: {headers[idx_cause]} --> {headers[idx_effect]}")
    
    # 2. Preprocessing (Deep Learning standard)
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    Y_norm = scaler.fit_transform(Y)
    
    combined_data = np.hstack([X_norm, Y_norm])
    
    # 3. Run Advanced Causal Inference
    # Increase lda for real-world noisy data
    direction, analyzer = ANMMM_cd_advanced(combined_data, lda=15.0)
    
    # 4. Result Interpretation
    inferred_cause = headers[idx_cause] if direction == 1 else headers[idx_effect]
    inferred_effect = headers[idx_effect] if direction == 1 else headers[idx_cause]
    
    print(f"\n[FINAL RESULT]")
    print(f"Inferred: {inferred_cause} --> {inferred_effect}")
    
    if direction == 1:
        print(">>> SUCCESS: Match with biological ground truth!")
    else:
        print(">>> MISMATCH: Inferred reverse direction.")

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    test_sachs_causality()
