import random
import networkx as nx
import matplotlib.pyplot as plt
import re

def extract_vertical_position(node_name):
    match = re.search(r'blocks\.(\d+)', node_name)
    return int(match.group(1)) if match else 0

def extract_horizontal_position(node_name):
    match = re.search(r'\[.*?(\d+)', node_name)
    return int(match.group(1)) if match else 0
def visualize_circuit(circuit):
    # Create a new directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    for node, data in circuit["nodes"].items():
        G.add_node(node, **data)

    # Add edges to the graph
    for edge in circuit["edges"]:
        source, target, data = edge
        G.add_edge(source, target, **data)

    # Set up the plot
    plt.figure(figsize=(20, 16))
    
    # Calculate node sizes based on attribution
    node_sizes = [data['attribution'] * 1000 for node, data in G.nodes(data=True)]
    
    # Calculate edge widths based on attribution
    edge_widths = [data['attribution'] * 5 for (u, v, data) in G.edges(data=True)]

    # Create custom positions for nodes with jitter
    pos = {}
    for node in G.nodes():
        x = extract_horizontal_position(node.reduction) + random.uniform(-0.3, 0.3)
        y = extract_vertical_position(node.hook_point) + random.uniform(-0.6, 0.6)  # Not negated now
        pos[node] = (x, y)

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', arrows=True)
    
    # Adjust label positions slightly above nodes
    label_pos = {node: (x, y + 0.1) for node, (x, y) in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=8)

    # Add a title
    plt.title("Circuit Visualization (Sorted with Jitter)")
    
    # Show the plot
    plt.axis('off')
    plt.tight_layout()
    plt.show()

