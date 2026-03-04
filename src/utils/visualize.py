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

def plot_dag(W_matrix, labels=None, title="DeepANM Causal Discovery Graph",
             threshold=0.1, ax=None, save_path=None, node_size=2000,
             font_size=10, figure_size=(10, 8)):
    """
    Render a weighted causal DAG as a directed graph.

    Parameters
    ----------
    W_matrix    : (n, n) adjacency matrix; W[i,j] ≠ 0 means edge i → j.
    labels      : list of variable names (defaults to ['X0', 'X1', ...])
    title       : plot title
    threshold   : minimum |W| to display an edge (weaker edges are hidden)
    ax          : matplotlib Axes to draw on (None creates a new figure)
    save_path   : file path to save the figure (None shows interactively)
    node_size   : node circle size
    font_size   : label font size inside nodes
    figure_size : (width, height) of the figure in inches

    Vẽ DAG nhân quả có trọng số dưới dạng đồ thị có hướng.

    Tham số
    -------
    W_matrix    : ma trận kề (n, n); W[i,j] ≠ 0 nghĩa là cạnh i → j.
    labels      : danh sách tên biến (mặc định ['X0', 'X1', ...])
    title       : tiêu đề biểu đồ
    threshold   : |W| tối thiểu để hiển thị cạnh (cạnh yếu hơn bị ẩn)
    ax          : Axes matplotlib để vẽ (None tạo figure mới)
    save_path   : đường dẫn lưu hình (None hiển thị tương tác)
    node_size   : kích thước node
    font_size   : cỡ chữ nhãn bên trong node
    figure_size : (rộng, cao) của figure tính bằng inch
    """
    if nx is None or plt is None:
        raise ImportError("Install required packages: pip install networkx matplotlib")

    # Filter edges below threshold / Lọc cạnh dưới ngưỡng
    W_filtered = np.where(np.abs(W_matrix) > threshold, W_matrix, 0)
    G = nx.DiGraph(W_filtered)

    # Map node indices to variable names / Ánh xạ chỉ số node thành tên biến
    n_nodes = W_matrix.shape[0]
    if labels is None:
        labels = [f"X{i}" for i in range(n_nodes)]
    G = nx.relabel_nodes(G, {i: labels[i] for i in range(n_nodes)})

    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size)
        show_plot = True

    # Handle empty graph / Xử lý đồ thị rỗng
    if len(G.edges) == 0:
        ax.text(0.5, 0.5, "No Edges Found Above Threshold", horizontalalignment='center',
                verticalalignment='center', fontsize=20, color='red', transform=ax.transAxes)
        ax.set_title(title)
        ax.axis('off')
        if show_plot and save_path is None:
            plt.show()
        elif save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        return

    # Edge weights for color and width scaling / Trọng số cạnh để điều chỉnh màu và độ rộng
    edges = G.edges(data=True)
    weights = [abs(data['weight']) for u, v, data in edges]
    
    max_w = max(weights) if weights else 1.0
    min_w = min(weights) if weights else 0.0
    norm = mcolors.Normalize(vmin=0 if max_w == min_w else min_w,
                             vmax=max_w + (1e-5 if max_w == min_w else 0))
    
    # Color scheme: winter_r (blue → teal gradient)
    # Sơ đồ màu: winter_r (gradient xanh dương → xanh ngọc)
    cmap = plt.cm.winter_r
    edge_colors = [cmap(norm(w)) for w in weights]
    
    # Arrow width proportional to causal effect strength
    # Độ rộng mũi tên tỷ lệ với cường độ tác động nhân quả
    edge_widths = [1.5 + (w / (max_w + 1e-9)) * 3.5 for w in weights]

    # Spring layout (Fruchterman-Reingold) avoids node overlap in symmetric graphs
    # Bố cục spring (Fruchterman-Reingold) tránh chồng lấp node trong đồ thị đối xứng
    pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42, weight=None)

    # Draw nodes with white fill and dark border / Vẽ node trắng viền đậm
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
                           node_color='white', alpha=1.0,
                           edgecolors='#2c3e50', linewidths=2.5)
    
    # Draw labels / Vẽ nhãn
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size,
                            font_weight='bold', font_color='black')
    
    # Draw directed edges with curved arcs / Vẽ cạnh có hướng với cung cong
    nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='-|>', arrowsize=22,
                           edge_color=edge_colors, width=edge_widths,
                           node_size=node_size,  # Needed for correct arrowhead placement / Cần để mũi tên đặt đúng chỗ
                           connectionstyle="arc3,rad=0.1", alpha=0.85)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Colorbar for causal effect magnitude / Thanh màu cho cường độ tác động nhân quả
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Causal Effect Strength (ATE / Prob)', rotation=270, labelpad=15, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=False)
        print(f"Causal graph saved to: {save_path}")
        
    if show_plot and save_path is None:
        plt.show()
    
    if show_plot and save_path is not None:
        plt.close()  # Free memory when saving only / Giải phóng bộ nhớ khi chỉ lưu file
