# %%
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



# %%


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

def hierarchical_clustering(X, n_clusters=100, metric='cosine'):
    X = X.cpu().numpy()
    Z = linkage(X, method='average')
    cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    return cluster_labels

from sklearn.cluster import MiniBatchKMeans
def kmeans_clustering(X, n_clusters=100):
    clusters = MiniBatchKMeans(n_clusters=n_clusters, random_state=0).fit(X)
    return clusters.labels_



# %%
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
    k_values = range(500,1000,50)
    silhouette_scores = []
    for k in k_values:
        cluster_labels = kmeans_clustering(y, n_clusters=k)
        print(len(cluster_labels))
        silhouette_avg = silhouette_score(y, cluster_labels)
        print(f"Silhouette score for k={k} is {silhouette_avg}")
        silhouette_scores.append(silhouette_avg)

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.xticks(k_values)
    plt.grid()
    plt.show()


    #distance_matrix = precompute_distance_matrix(y, metric='cosine')
    #cluster_labels = hierarchical_clustering(distance_matrix, n_clusters=100, metric='cosine')



