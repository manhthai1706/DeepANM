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
    Render a weighted causal DAG as a directed graph. / Vẽ DAG nhân quả có trọng số.

    Parameters / Tham số
    ----------
    W_matrix    : (n, n) adjacency matrix; W[i,j] ≠ 0 means edge i → j. / ma trận kề (n, n).
    labels      : list of variable names (defaults to ['X0', 'X1', ...]) / danh sách tên biến.
    title       : plot title / tiêu đề biểu đồ.
    threshold   : minimum |W| to display an edge (weaker edges are hidden) / ngưỡng tối thiểu hiển thị cạnh.
    ax          : matplotlib Axes to draw on / Axes matplotlib để vẽ.
    save_path   : file path to save the figure / đường dẫn lưu hình.
    node_size   : node circle size / kích thước node.
    font_size   : label font size inside nodes / cỡ chữ nhãn node.
    figure_size : (width, height) of the figure in inches / kích thước hình tính bằng inch.
    """
    if nx is None or plt is None:
        raise ImportError("Install required packages: pip install networkx matplotlib")

    # Filter edges below threshold / Lọc các cạnh có trọng số tuyệt đối dưới ngưỡng
    W_filtered = np.where(np.abs(W_matrix) > threshold, W_matrix, 0)
    G = nx.DiGraph(W_filtered) # Tạo đối tượng đồ thị có hướng / Create directed graph object

    # Map node indices to variable names / Ánh xạ chỉ số node thành tên biến thực tế
    n_nodes = W_matrix.shape[0]
    if labels is None:
        labels = [f"X{i}" for i in range(n_nodes)]
    G = nx.relabel_nodes(G, {i: labels[i] for i in range(n_nodes)})

    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size) # Tạo figure mới nếu ax chưa có / Create new figure if ax is None
        show_plot = True

    # Handle empty graph cases / Xử lý trường hợp đồ thị rỗng (không có cạnh trên ngưỡng)
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

    # Extract edge weights for color and width scaling / Lấy trọng số cạnh để ánh xạ màu và độ rộng
    edges = G.edges(data=True)
    weights = [abs(data['weight']) for u, v, data in edges]
    
    max_w = max(weights) if weights else 1.0
    min_w = min(weights) if weights else 0.0
    # Normalize weights for colormap / Chuẩn hóa trọng số cho bảng màu
    norm = mcolors.Normalize(vmin=0 if max_w == min_w else min_w,
                             vmax=max_w + (1e-5 if max_w == min_w else 0))
    
    # Color scheme: winter_r (blue → teal gradient for professional look)
    # Sơ đồ màu: winter_r (biến thiên xanh dương → xanh ngọc cho giao diện chuyên nghiệp)
    cmap = plt.cm.winter_r
    edge_colors = [cmap(norm(w)) for w in weights]
    
    # Arrow thickness proportional to causal effect strength / Độ rộng mũi tên tỷ lệ với cường độ nhân quả
    edge_widths = [1.5 + (w / (max_w + 1e-9)) * 3.5 for w in weights]

    # Spring layout (Fruchterman-Reingold) algorithm for optimal node placement
    # Thuật toán bố cục spring (Fruchterman-Reingold) để tối ưu vị trí node
    pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42, weight=None)

    # Draw nodes with white fill and dark slate border / Vẽ các node với nền trắng và viền đá đậm
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
                           node_color='white', alpha=1.0,
                           edgecolors='#2c3e50', linewidths=2.5)
    
    # Draw variable labels inside the nodes / Vẽ nhãn tên biến bên trong các node
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size,
                            font_weight='bold', font_color='black')
    
    # Draw curved arcs instead of straight lines to avoid overlap / Vẽ các cung cong để tránh chồng lấp
    nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='-|>', arrowsize=22,
                           edge_color=edge_colors, width=edge_widths,
                           node_size=node_size, # Ensures arrowheads stop at node border / Đảm bảo mũi tên dừng sát viền node
                           connectionstyle="arc3,rad=0.1", alpha=0.85)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off') # Hide coordinate axes / Ẩn trục tọa độ
    
    # Add colorbar for magnitude reference / Thêm thanh màu để tham chiếu cường độ
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Causal Effect Strength (ATE / Prob)', rotation=270, labelpad=15, fontweight='bold')

    plt.tight_layout()

    # Save to file if path provided / Lưu ra file nếu có đường dẫn
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=False)
        print(f"Causal graph saved to: {save_path}") # In thông báo lưu file thành công
        
    if show_plot and save_path is None:
        plt.show() # Show interactive window / Hiển thị cửa sổ tương tác
    
    if show_plot and save_path is not None:
        plt.close() # Free resources after saving / Giải phóng tài nguyên sau khi lưu
