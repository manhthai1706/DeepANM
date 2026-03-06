"""
Causal Graph Visualization Module.
Renders the discovered DAG as a directed graph with edge colors
proportional to causal effect strength (ATE or bootstrap probability).
Requires: networkx, matplotlib

Module Trực quan hóa Đồ thị Nhân quả.
Vẽ DAG đã khám phá dưới dạng đồ thị có hướng với màu cạnh
tỷ lệ thuận với cường độ tác động nhân quả (ATE hoặc xác suất bootstrap).
Yêu cầu: networkx, matplotlib
"""

import numpy as np
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except ImportError:
    nx = None
    plt = None

def plot_dag(W_matrix, labels=None, GT_matrix=None, title="DeepANM Causal Discovery Graph",
             threshold=0.1, ax=None, save_path=None, node_size=2000,
             font_size=10, figure_size=(12, 9)):
    """
    Render a weighted causal DAG as a directed graph. / Vẽ DAG nhân quả có trọng số.
    Supports comparison with Ground Truth if GT_matrix is provided. / Hỗ trợ so sánh với Ground Truth.
    """
    if nx is None or plt is None:
        raise ImportError("Install required packages: pip install networkx matplotlib")

    # Filter edges below threshold
    W_filtered = np.where(np.abs(W_matrix) > threshold, W_matrix, 0)
    G = nx.DiGraph(W_filtered)

    n_nodes = W_matrix.shape[0]
    if labels is None:
        labels = [f"X{i}" for i in range(n_nodes)]
    
    # Map node indices to labels
    G = nx.relabel_nodes(G, {i: labels[i] for i in range(n_nodes)})

    # Initialize GT graph if provided
    G_gt = None
    if GT_matrix is not None:
        G_gt = nx.DiGraph(np.abs(GT_matrix) > 0.1)
        G_gt = nx.relabel_nodes(G_gt, {i: labels[i] for i in range(n_nodes)})

    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size)
        show_plot = True

    if len(G.edges) == 0:
        ax.text(0.5, 0.5, "No Edges Found Above Threshold", horizontalalignment='center',
                verticalalignment='center', fontsize=20, color='red', transform=ax.transAxes)
        ax.set_title(title)
        ax.axis('off')
        if show_plot:
            if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=300)
            else: plt.show()
        return

    # Categorize edges and determine styles
    edges = list(G.edges())
    edge_colors = []
    edge_styles = []
    
    # Colors matching the image reference
    COLOR_CORRECT = "#c0392b"    # Red
    COLOR_INDIRECT = "#2980b9"   # Blue
    COLOR_REVERSED = "#f39c12"   # Yellow/Orange
    COLOR_UNEXPLAINED = "#27ae60" # Green

    if G_gt is not None:
        for u, v in edges:
            if G_gt.has_edge(u, v):
                edge_colors.append(COLOR_CORRECT)
                edge_styles.append("solid")
            elif G_gt.has_edge(v, u):
                edge_colors.append(COLOR_REVERSED)
                edge_styles.append("solid")
            elif nx.has_path(G_gt, u, v):
                edge_colors.append(COLOR_INDIRECT)
                edge_styles.append("dashed")
            else:
                edge_colors.append(COLOR_UNEXPLAINED)
                edge_styles.append("dashed")
    else:
        # Default behavior without GT: Use coloring based on weight (ATE/Prob)
        weights = [abs(G[u][v]['weight']) for u, v in edges]
        max_w = max(weights) if weights else 1.0
        norm = mcolors.Normalize(vmin=0, vmax=max_w)
        cmap = plt.cm.winter_r
        edge_colors = [cmap(norm(w)) for w in weights]
        edge_styles = ["solid"] * len(edges)

    # Node positions
    pos = nx.spring_layout(G, k=1.8, iterations=150, seed=42)

    # Draw nodes (Oval-like style from image)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
                           node_color='#d6eaf8', # Light blue fill
                           edgecolors='#2e86c1', # Dark blue border
                           linewidths=2.0)
    
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size,
                            font_weight='bold', font_color='#1b2631')
    
    # Draw edges with specific patterns
    for i, (u, v) in enumerate(edges):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, 
                               arrowstyle='-|>', arrowsize=20,
                               edge_color=edge_colors[i],
                               style=edge_styles[i],
                               width=2.5,
                               connectionstyle="arc3,rad=0.1",
                               node_size=node_size, alpha=0.9)

    ax.set_title(title, fontsize=18, fontweight='bold', pad=30)
    ax.axis('off')

    # Add Legend if GT comparison is active
    if G_gt is not None:
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color=COLOR_CORRECT, lw=2, linestyle='-'),
            Line2D([0], [0], color=COLOR_INDIRECT, lw=2, linestyle='--'),
            Line2D([0], [0], color=COLOR_REVERSED, lw=2, linestyle='-'),
            Line2D([0], [0], color=COLOR_UNEXPLAINED, lw=2, linestyle='--')
        ]
        ax.legend(custom_lines, ['Correct Edge', 'Indirect Edge', 'Reversed Edge', 'Unexplained Edge'],
                  loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize=11, title="Edge Categories")
    else:
        # Add colorbar for weight-based coloring
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Causal Effect Strength (ATE / Prob)', rotation=270, labelpad=15, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=False)
        print(f"Causal graph comparison saved to: {save_path}")
        
    if show_plot:
        if save_path is None: plt.show()
        else: plt.close()
