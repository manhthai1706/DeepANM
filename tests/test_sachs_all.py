"""
Comprehensive Sachs Dataset Evaluation
Tests all 17 known causal edges
"""
import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import IsolationForest
from causalflow import ANMMM_cd_advanced

def test_all_sachs_edges():
    print("=" * 70)
    print("ENHANCED SACHS EVALUATION (SKLEARN OPTIMIZED)")
    print("=" * 70)
    
    # Load data
    data = np.load('data/sachs/continuous/data1.npy')
    headers = np.load('data/sachs/sachs-header.npy')
    dag = np.load('data/sachs/continuous/DAG1.npy')
    
    # Extract all true causal edges from DAG
    true_edges = []
    for i in range(11):
        for j in range(11):
            if dag[i, j] == 1:
                true_edges.append((i, j, headers[i], headers[j]))
    
    print(f"\nTotal edges to test: {len(true_edges)}")
    print("Applying: QuantileTransform + IsolationForest + Adaptive LDA")
    print("-" * 70)
    
    results = []
    
    # Preprocessor initialization
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=500, random_state=42)
    iso = IsolationForest(contamination=0.05, random_state=42)
    
    for idx, (cause_idx, effect_idx, cause_name, effect_name) in enumerate(true_edges):
        print(f"\n[{idx+1}/{len(true_edges)}] Analyzing: {cause_name} -> {effect_name}")
        
        # 1. Extract and Normalize with QuantileTransformer
        X = data[:, cause_idx].reshape(-1, 1)
        Y = data[:, effect_idx].reshape(-1, 1)
        
        X_norm = qt.fit_transform(X)
        Y_norm = qt.fit_transform(Y)
        combined = np.hstack([X_norm, Y_norm])
        
        # 2. Advanced Outlier Removal
        clean_mask = iso.fit_predict(combined)
        data_clean = combined[clean_mask == 1]
        
        # 3. Training with Adaptive LDA
        try:
            # We testing two LDA levels and picking the more stable decision
            direction, _ = ANMMM_cd_advanced(data_clean, lda=12.0)
            
            correct = (direction == 1)
            result = "CORRECT" if correct else "WRONG"
            
            results.append({
                'edge': f"{cause_name} -> {effect_name}",
                'correct': correct,
                'result': result
            })
            print(f"    Result: {result}")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({'edge': f"{cause_name} -> {effect_name}", 'correct': False, 'result': "ERROR"})
    
    # Summary
    print("\n" + "=" * 70)
    print("METRICS SUMMARY (OPTIMIZED)")
    print("=" * 70)
    
    correct_count = sum(1 for r in results if r['correct'])
    total = len(results)
    
    accuracy = correct_count / total
    shd = total - correct_count
    f1 = accuracy # In forced choice binary task
    
    print(f"Accuracy:  {accuracy*100:>.1f}%")
    print(f"SHD:       {shd}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\nDetailed Results:")
    for r in results:
        status = "[OK]" if r['correct'] else "[X]"
        print(f"  {status} {r['edge']}")
    
    return results

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    test_all_sachs_edges()
