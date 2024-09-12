
from sae_lens import SAE, SAEConfig, HookedSAETransformer
import torch
from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset
from collect_activations import ActivationsColector
import tqdm
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import networkx as nx
from scipy.sparse import csr_matrix, dok_matrix

def precompute_distance_matrix(X, metric='cosine'):
    print("Precomputing distance matrix...")

    if metric == 'cosine':
        # Normalize X to unit vectors
        X_norm = X / X.norm( dim=1, keepdim=True)
        # Compute cosine distance as 1 - cosine similarity
        mul =  (X_norm @ X_norm.T)

        distance_matrix = 1 - mul
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    return distance_matrix

# Helper function to calculate modularity
def modularity(G, communities, m):
    Q = 0.0
    degrees = G.sum(axis=1).A1  # Sum along rows (axis=1) to get degrees
    for community in communities.values():
        subgraph = G[np.ix_(community, community)]
        lc = subgraph.sum()  # Total weight within community
        dc = degrees[community].sum()  # Sum of degrees in the community
        Q += (lc / (2 * m)) - (dc / (2 * m)) ** 2
    return Q

# Louvain method implementation
def louvain_method(G):
    print(G.shape)
    n = G.shape[0]  # Number of nodes
    m = G.sum() / 2  # Total weight of all edges

    # Initially, assign each node to its own community
    communities = {i: [i] for i in range(n)}
    print(communities)
    current_modularity = modularity(G, communities, m)
    improvement = True

    while improvement:
        improvement = False
        for node in tqdm.tqdm(range(n)):

            # Find the best community for the node
            current_community = [k for k, v in communities.items() if node in v][0]
            best_community = current_community
            best_modularity = current_modularity

            # Remove the node from its current community
            communities[current_community].remove(node)
            if not communities[current_community]:
                del communities[current_community]

            # Calculate modularity gain for all other communities
            for community in communities.values():
                community.append(node)
                new_modularity = modularity(G, communities, m)
                if new_modularity > best_modularity:
                    best_modularity = new_modularity
                    best_community = community.copy()
                community.remove(node)

            # If moving the node improves modularity, do it
            if best_community != current_community:
                improvement = True
                communities[best_community].append(node)
                current_modularity = best_modularity
            else:
                communities[current_community].append(node)

    return communities
def distance_to_similarity(X, method="inverse"):

    if method == "inverse":
        # Simple inverse distance conversion
        similarity = 1 / (1 + X)  # Prevent division by zero
    elif method == "gaussian":
        # Gaussian kernel conversion
        sigma = np.std(X)
        similarity = np.exp(-X**2 / (2 * sigma**2))
    return similarity




# Function to transform the graph into a sparse adjacency matrix
def graph_to_sparse_adjacency_matrix(graph):
    # Convert a NetworkX graph to a sparse adjacency matrix
    n = len(graph.nodes())
    adj_matrix = dok_matrix((n, n), dtype=np.float32)
    
    for u, v, w in graph.edges(data=True):
        adj_matrix[u, v] = w.get('weight', 1.0)  # Default weight is 1
        adj_matrix[v, u] = w.get('weight', 1.0)
    
    return adj_matrix.tocsr()  # Convert to CSR format for efficient access

# Example usage
if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = HookedSAETransformer.from_pretrained("gpt2", device = device)
    data = load_dataset("NeelNanda/pile-10k", split = "train")
    tokens = tokenize_and_concatenate(data, model.tokenizer, max_length=128)

    acts = ActivationsColector(model, tokens, ["blocks.1.attn.hook_z","blocks.2.attn.hook_z",],"Features","../activations/","../final_dicts/version_2/", cat_activations=True, quantize = True ,average = True, load = True, version = 2, filename = "activations_Features_True.h5", device = device, stop_at_layer=3)

    all_activations = []
    for batch, batch_dict in acts.activations.items():
        for doc, doc_list in batch_dict.items():
            for hook, hook_dict in doc_list.items():
                for pos, act in hook_dict.items():
                    if len(act.shape) == 3:
                        act = act.flatten()
                        all_activations.append(act)
                    elif len(act.shape) == 2:
                        act = torch.tensor(act.flatten())
                        all_activations.append(act)

# %%
    X = torch.stack(all_activations)
    X = X[:,:24576]
    y = X.contiguous().type(torch.float32)
    distance_matrix = precompute_distance_matrix(y, metric='cosine')
    distance_matrix = distance_matrix.cpu().numpy()
    # Create a sample graph using networkx
    S = distance_to_similarity(distance_matrix, method="gaussian")
    
    # Convert the graph to a sparse adjacency matrix
    sparse_S = csr_matrix(S)
    
    # Run the Louvain method on the sparse adjacency matrix
    communities = louvain_method(sparse_S)
    
    # Output the detected communities
    for community_id, members in communities.items():
        print(f"Community {community_id}: {members}")






