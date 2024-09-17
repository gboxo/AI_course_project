# Create a gif

import torch
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import imageio
import os



# Create a graph with the distance matrix of each dictionary entry



def compute_running_mean(matrices):
    running_mean = []
    for i, matrix in enumerate(matrices):
        if i == 0:
            running_mean.append(matrix)
        else:
            running_mean.append((i*running_mean[-1] + matrix) / (i + 1))
    return running_mean

def create_graph_from_matrix(matrix):
    G = nx.Graph()
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            # Use the inverse of the distance as the edge weight
            # Adding a small epsilon to avoid division by zero
            G.add_edge(i, j, weight=(matrix[i, j] + 1e-6))
    return G
def plot_graph(G, pos, iteration, output_file):
    plt.figure(figsize=(10, 10))
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    # Create a custom colormap
    colors = ['#87CEFA', '#98FB98', '#FFA07A']  # Light sky blue, pale green, light salmon
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

    # Normalize edge weights for coloring
    min_weight, max_weight = min(weights), max(weights)
    norm = plt.Normalize(min_weight, max_weight)

    # Normalize edge widths
    min_width, max_width = 0.5, 3.0  # You can adjust these values
    width_norm = plt.Normalize(min_weight, max_weight)
    edge_widths = [width_norm(w) * (max_width - min_width) + min_width for w in weights]

    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightgray')
    nx.draw_networkx_labels(G, pos)
    
    # Draw edges with normalized widths
    nx.draw_networkx_edges(G, pos, 
                           edge_color=[cmap(norm(w)) for w in weights], 
                           width=edge_widths)

    plt.title(f"Graph Evolution - Iteration {iteration}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def create_gif(png_files, output_gif, duration=1):
    images = []
    duration_ms = int(duration * 1000)
    for filename in png_files:
        images.append(imageio.imread(filename))

    imageio.mimsave(output_gif, images, duration=duration_ms, loop=0)

if __name__ == '__main__':
    dist_comp = torch.load("app_data/feat_pairwise_dist_comp.pt")
    keys = list(dist_comp.keys())
    keys.sort()
    dist_comp = {k: dist_comp[k] for k in keys}
    matrices = [tensor.numpy() for tensor in dist_comp.values()]
    running_means = compute_running_mean(matrices)

# Generate a fixed layout for consistent node positions
    G = create_graph_from_matrix(running_means[0])
    pos = nx.spring_layout(G)

# Create a temporary directory to store PNG files
    temp_dir = 'temp_png_files'
    os.makedirs(temp_dir, exist_ok=True)

    png_files = []
    for i, matrix in enumerate(running_means):
        G = create_graph_from_matrix(matrix)
        output_file = os.path.join(temp_dir, f"graph_evolution_{i:03d}.png")
        plot_graph(G, pos, list(dist_comp.keys())[i], output_file)
        png_files.append(output_file)

# Create the GIF
    output_gif = "graph_evolution.gif"
    create_gif(png_files, output_gif,duration = 1)

# Clean up temporary PNG files
    for file in png_files:
        os.remove(file)
    os.rmdir(temp_dir)

    print(f"GIF has been created: {output_gif}")


