"""
Causal Graph Visualization Module / Module vẽ đồ thị nhân quả
Requires: networkx, matplotlib
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
    Vẽ đồ thị có hướng (DAG) biểu diễn quan hệ nhân quả học được.
    
    Args:
        W_matrix (np.ndarray): Ma trận kề trọng số (ví dụ: ATE Jacobian Matrix hoặc Thresholded DAG).
                                W[i, j] khác 0 nghĩa là có cạnh mũi tên từ i -> j.
        labels (list of str): Tên của các biến/node (nếu có).
        title (str): Tiêu đề của biểu đồ.
        threshold (float): Ngưỡng để lọc các cạnh quá mỏng (chỉ vẽ cạnh |W| > threshold).
        ax (matplotlib.axes.Axes): Trục tọa độ để vẽ (giúp nhúng vào plot lớn hơn), None thì tự tạo figure mới.
        save_path (str): Đường dẫn lưu hình ảnh (ví dụ: 'result.png'). Nếu None thì chỉ hiện lên màn hình.
        node_size (int): Kích thước của Node tròn.
        font_size (int): Cỡ chữ của Label bên trong Node.
        figure_size (tuple): Kích thước bức ảnh.
    """
    if nx is None or plt is None:
        raise ImportError("Để vẽ đồ thị, bạn cần cài đặt: pip install networkx matplotlib")

    # Lọc ma trận kề dựa trên ngưỡng threshold
    W_filtered = np.where(np.abs(W_matrix) > threshold, W_matrix, 0)
    
    # Tạo đồ thị có hướng (DiGraph) từ ma trận đã lọc
    G = nx.DiGraph(W_filtered)

    # Đổi tên các đỉnh tĩnh thành Label thực tế (vd: 'X1', 'Akt', 'Mek')
    n_nodes = W_matrix.shape[0]
    if labels is None:
        labels = [f"X{i}" for i in range(n_nodes)]
    
    mapping = {i: labels[i] for i in range(n_nodes)}
    G = nx.relabel_nodes(G, mapping)

    # Cấu tạo ảnh
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size)
        show_plot = True

    # Nếu đồ thị không có cạnh nào
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

    # Lấy danh sách trọng số (Trị tuyệt đối của Marginal ATE hoặc Bootstrap Prob) để chỉnh màu cạnh
    edges = G.edges(data=True)
    weights = [abs(data['weight']) for u, v, data in edges]
    
    # Chuẩn hóa trọng số (Min-max) để nội suy màu sắc và độ dày một cách bắt mắt
    max_w = max(weights) if weights else 1.0
    min_w = min(weights) if weights else 0.0
    
    if max_w == min_w: 
        norm = mcolors.Normalize(vmin=0, vmax=max_w + 1e-5)
    else:
        norm = mcolors.Normalize(vmin=min_w, vmax=max_w)
    
    # Phối màu mũi tên: Hệ màu Mát (Từ Xanh biển nhạt đến Xanh ngọc đậm)
    cmap = plt.cm.winter_r # Reversed winter (xanh dương sang xanh lá mát) hoặc có thể dùng plt.cm.GnBu
    edge_colors = [cmap(norm(w)) for w in weights]
    
    # Độ rộng mũi tên tăng dần theo cường độ ATE
    edge_widths = [1.5 + (w / (max_w + 1e-9)) * 3.5 for w in weights]

    # Tính toán bố cục
    # Đổi sang spring_layout (Fruchterman-Reingold) có gắn thêm lực đẩy ngẫu nhiên nhẹ, 
    # giúp các node có cấu trúc đối xứng (ví dụ X2, X3 đều nhận từ X1 và trỏ về X4) không bị đè lên nhau.
    pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42, weight=None)

    # Tô màu Node bằng màu Trắng (White) có viền sẫm màu
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, 
                           node_color='white', alpha=1.0,
                           edgecolors='#2c3e50', linewidths=2.5)
    
    # Vẽ chữ (Labels) màu Đen mượt mà rõ nét
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size, 
                            font_weight='bold', font_color='black')
    
    # Vẽ vòng cung Mũi tên (Dẹp vòng cung để nét cắt đẹp)
    nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='-|>', arrowsize=22, 
                           edge_color=edge_colors, width=edge_widths,
                           node_size=node_size, # Rất quan trọng: Báo cho NetworkX biết mép Node ở đâu để cắm mũi tên vào
                           connectionstyle="arc3,rad=0.1", alpha=0.85)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off') # Ẩn khung trục Oxyz
    
    # Thêm Colorbar giải thích cường độ Tác động Nhân quả (Causal Effect Magnitude)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Causal Effect Strength (ATE / Prob)', rotation=270, labelpad=15, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=False)
        print(f"Causal graph successfully saved to: {save_path}")
        
    if show_plot and save_path is None:
        plt.show()
    
    if show_plot and save_path is not None:
        plt.close() # Dọn dẹp RAM nếu chỉ lưu file ảnh
