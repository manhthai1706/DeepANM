"""
Comprehensive Sachs Dataset Evaluation
Tests all 17 known causal edges
"""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from ANMMM import ANMMM_cd_advanced

def test_all_sachs_edges():
    print("=" * 70)
    print("COMPREHENSIVE SACHS DATASET EVALUATION")
    print("=" * 70)
    
    # Load data
    data = np.load('sachs/continuous/data1.npy')
    headers = np.load('sachs/sachs-header.npy')
    dag = np.load('sachs/continuous/DAG1.npy')
    
    # Extract all true causal edges from DAG
    true_edges = []
    for i in range(11):
        for j in range(11):
            if dag[i, j] == 1:
                true_edges.append((i, j, headers[i], headers[j]))
    
    print(f"\nTotal edges to test: {len(true_edges)}")
    print("-" * 70)
    
    results = []
    
    for idx, (cause_idx, effect_idx, cause_name, effect_name) in enumerate(true_edges):
        print(f"\n[{idx+1}/{len(true_edges)}] Testing: {cause_name} -> {effect_name}")
        
        # Extract pair
        X = data[:, cause_idx].reshape(-1, 1)
        Y = data[:, effect_idx].reshape(-1, 1)
        
        # Preprocess
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
        Y_norm = scaler.fit_transform(Y)
        combined = np.hstack([X_norm, Y_norm])
        
        # Run causal discovery
        try:
            direction, _ = ANMMM_cd_advanced(combined, lda=15.0)
            
            if direction == 1:
                result = "CORRECT"
                correct = True
            else:
                result = "WRONG"
                correct = False
                
            results.append({
                'edge': f"{cause_name} -> {effect_name}",
                'correct': correct,
                'result': result
            })
            print(f"    Result: {result}")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                'edge': f"{cause_name} -> {effect_name}",
                'correct': False,
                'result': "ERROR"
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    correct_count = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = correct_count / total * 100
    
    print(f"\nAccuracy: {correct_count}/{total} = {accuracy:.1f}%")
    print("\nDetailed Results:")
    for r in results:
        status = "[OK]" if r['correct'] else "[X]"
        print(f"  {status} {r['edge']}")
    
    return results, accuracy

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    results, accuracy = test_all_sachs_edges()
