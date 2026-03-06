"""
Ablation Study: DeepANM on Sachs Dataset.
Tests 4 configurations to show the contribution of each component.

Level 1: TopoSort + OLS only             (Baseline linear)
Level 2: TopoSort + RF                   (+ Non-linear edge selection)
Level 3: TopoSort + RF + CI Pruning      (+ Conditional Independence pruning)
Level 4: TopoSort + RF + CI + SCM Filter (Full pipeline)
"""
import os, sys
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.deepanm import DeepANM

def evaluate(W, gt_edges, n_vars):
    """Compute all evaluation metrics."""
    found_edges = []
    tp = fp = 0
    for i in range(n_vars):
        for j in range(n_vars):
            if W[i, j] == 1:
                found_edges.append((i, j))
                if (i, j) in gt_edges:
                    tp += 1
                else:
                    fp += 1
    fn = len(gt_edges) - tp
    
    reversals = extra = missing = 0
    for u, v in gt_edges:
        if W[u, v] == 0:
            if W[v, u] == 1: reversals += 1
            else: missing += 1
    for i, j in found_edges:
        if (i, j) not in gt_edges and (j, i) not in gt_edges:
            extra += 1
    
    shd = reversals + extra + missing
    total = n_vars * (n_vars - 1)
    tn = total - tp - fp - fn
    
    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    nhd = shd / total if total > 0 else 0
    
    return {
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'SHD': shd, 'Missing': missing, 'Extra': extra, 'Reversed': reversals,
        'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'NHD': nhd,
        'Edges': len(found_edges)
    }

def main():
    print("="*70)
    print(" ABLATION STUDY: DeepANM Component Contributions (Sachs Dataset)")
    print("="*70)
    
    # Load Data
    url = 'https://raw.githubusercontent.com/FenTechSolutions/CausalDiscoveryToolbox/master/cdt/data/resources/cyto_full_data.csv'
    df = pd.read_csv(url)
    labels = df.columns.tolist()
    n_vars = len(labels)
    
    print(f"Data: {df.shape[0]} samples, {n_vars} variables")
    
    # Ground Truth
    gt_edges_str = [
        ('praf', 'pmek'), ('plcg', 'PIP2'), ('PIP3', 'PIP2'), ('PIP3', 'plcg'), 
        ('pmek', 'p44/42'), ('p44/42', 'pakts473'), 
        ('PKA', 'praf'), ('PKA', 'pmek'), ('PKA', 'p44/42'), ('PKA', 'pakts473'), ('PKA', 'P38'), ('PKA', 'pjnk'),
        ('PKC', 'praf'), ('PKC', 'pmek'), ('PKC', 'P38'), ('PKC', 'pjnk')
    ]
    label_to_idx = {l.lower(): i for i, l in enumerate(labels)}
    gt_edges = [(label_to_idx[u.lower()], label_to_idx[v.lower()]) for u, v in gt_edges_str]
    
    # Define 4 ablation configurations
    configs = [
        {
            'name': 'Level 1: TopoSort + OLS',
            'desc': 'Linear baseline (no RF, no CI, no SCM)',
            'use_rf': False, 'use_ci_pruning': False, 'use_scm_filter': False
        },
        {
            'name': 'Level 2: TopoSort + RF',
            'desc': '+ Non-linear edge selection',
            'use_rf': True, 'use_ci_pruning': False, 'use_scm_filter': False
        },
        {
            'name': 'Level 3: TopoSort + RF + CI',
            'desc': '+ Conditional Independence pruning',
            'use_rf': True, 'use_ci_pruning': True, 'use_scm_filter': False
        },
        {
            'name': 'Level 4: Full Pipeline',
            'desc': '+ Neural SCM Filter (Adaptive ATE Gate)',
            'use_rf': True, 'use_ci_pruning': True, 'use_scm_filter': True
        },
    ]
    
    results = []
    
    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"  {cfg['name']}")
        print(f"  {cfg['desc']}")
        print(f"{'='*70}")
        
        model = DeepANM()
        start = time.time()
        
        prob_matrix, avg_W = model.fit_bootstrap(
            df, n_bootstraps=1, apply_quantile=True,
            discovery_mode='fast', verbose=False,
            use_rf=cfg['use_rf'], use_ci_pruning=cfg['use_ci_pruning'],
            use_scm_filter=cfg['use_scm_filter']
        )
        
        W = (prob_matrix > 0).astype(int)
        elapsed = time.time() - start
        
        metrics = evaluate(W, gt_edges, n_vars)
        metrics['Time'] = elapsed
        metrics['Config'] = cfg['name']
        results.append(metrics)
        
        print(f"  Time: {elapsed:.1f}s | Edges: {metrics['Edges']}")
        print(f"  TP: {metrics['TP']}/{len(gt_edges)}  FP: {metrics['FP']}  FN: {metrics['FN']}")
        print(f"  SHD: {metrics['SHD']}  (Missing: {metrics['Missing']}, Extra: {metrics['Extra']}, Rev: {metrics['Reversed']})")
        print(f"  Accuracy: {metrics['Accuracy']:.1%}  Precision: {metrics['Precision']:.1%}  Recall: {metrics['Recall']:.1%}  F1: {metrics['F1']:.1%}")
    
    # Summary Table
    print(f"\n{'='*70}")
    print("  ABLATION SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Configuration':<30s} {'TP':>3s} {'FP':>3s} {'SHD':>4s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'NHD':>6s} {'Time':>6s}")
    print("-" * 90)
    for r in results:
        print(f"{r['Config']:<30s} {r['TP']:>3d} {r['FP']:>3d} {r['SHD']:>4d} {r['Accuracy']:>5.1%} {r['Precision']:>5.1%} {r['Recall']:>5.1%} {r['F1']:>5.1%} {r['NHD']:>5.1%} {r['Time']:>5.0f}s")
    
    print(f"\n  Ground Truth: {len(gt_edges)} edges | Total possible: {n_vars*(n_vars-1)} directed edges")

    # ----- Visualization -----
    print("\nGenerating SHD comparison visualization...")
    os.makedirs('results', exist_ok=True)
    
    levels = [r['Config'].split(':')[0] for r in results]
    shd_scores = [r['SHD'] for r in results]
    f1_scores = [r['F1'] * 100 for r in results] # in percentage
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar plot for SHD (lower is better, primary axis)
    color = 'tab:red'
    ax1.set_xlabel('Ablation Levels', fontweight='bold')
    ax1.set_ylabel('Structural Hamming Distance (SHD)', color=color, fontweight='bold')
    bars = ax1.bar(levels, shd_scores, color=color, alpha=0.7, width=0.4, label='SHD (Lower is better)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontweight='bold', color=color)
        
    # Line plot for F1 (higher is better, secondary axis)
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('F1 Score (%)', color=color, fontweight='bold')
    line = ax2.plot(levels, f1_scores, color=color, marker='o', linestyle='-', linewidth=2, markersize=8, label='F1 Score (Higher is better)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add values next to points
    for i, txt in enumerate(f1_scores):
        ax2.annotate(f"{txt:.1f}%", (levels[i], f1_scores[i] + 1), ha='center', fontweight='bold', color=color)
        
    plt.title('Ablation Study: Component Contributions to DeepANM Model Performance', fontweight='bold', fontsize=14)
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    plt.tight_layout()
    save_path = "results/ablation_study_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved SHD vs F1 comparison plot to: {save_path}")

if __name__ == '__main__':
    main()
