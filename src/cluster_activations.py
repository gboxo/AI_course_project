

import h5py

from typing import List,Dict, Tuple,Any, Optional, Literal
import html
import pandas as pd
import json
import os
from sae_lens import SAE, SAEConfig, HookedSAETransformer
from matplotlib import pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from jaxtyping import Int, Float 
from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset
from torch.utils.data import DataLoader, dataset
from tqdm import tqdm
from collections import defaultdict


import h5py



from sklearn.cluster import SpectralClustering

class SpectralClusteringAnalyzer:
    def __init__(self, activations: Dict,
                 clusters_dir: str):
        self.activations = activations
        self.clusters_dir = clusters_dir
        self.cluster_labels = None

    def flatten_activations(self) -> np.ndarray:
        """Flatten the nested activations dictionary into a 2D array."""
        all_activations = []
        for batch, batch_dict in self.activations.items():
            for doc, doc_list in batch_dict.items():
                for hook, hook_dict in doc_list.items():
                    for pos, act in hook_dict.items():
                        all_activations.append(act.flatten())
        return np.array(all_activations)

    def _create_versioned_dir(self,base_dir) -> str:
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        version = 0
        while True:
            versioned_dir = os.path.join(base_dir, f"version_{version}")
            if not os.path.exists(versioned_dir):
                os.makedirs(versioned_dir)
                return versioned_dir
            version += 1
    def perform_clustering(self, n_clusters: int, affinity: str = 'nearest_neighbors'):
        """Perform spectral clustering on the activations."""
        # Flatten the activations to a 2D array for clustering
        flattened_activations = self.flatten_activations()
        
        # Perform spectral clustering
        clustering = SpectralClustering(n_clusters=n_clusters, affinity=affinity)
        self.cluster_labels = clustering.fit_predict(flattened_activations)

    def get_cluster_labels(self) -> np.ndarray:
        """Get the cluster labels assigned to the activations."""
        if self.cluster_labels is None:
            raise ValueError("Clustering has not been performed yet. Call 'perform_clustering' first.")
        return self.cluster_labels

    def save_cluster_labels(self):
        """Save the cluster labels to an HDF5 file with the same structure as activations."""
        if self.cluster_labels is None:
            raise ValueError("Clustering has not been performed yet. Call 'perform_clustering' first.")
        
        # Create a mapping from the flattened structure to the original structure
        self.versioned_dir = self._create_versioned_dir(self.clusters_dir)
        filename = os.path.join(self.versioned_dir, "cluster_labels.h5")
        index = 0
        with h5py.File(filename, "w") as f:
            for batch, batch_dict in self.activations.items():
                batch_group = f.create_group(batch)
                for doc, doc_list in batch_dict.items():
                    doc_group = batch_group.create_group(str(doc))
                    for hook, hook_dict in doc_list.items():
                        hook_group = doc_group.create_group(hook)
                        for pos, act in hook_dict.items():
                            # Here we save the cluster label corresponding to the current position
                            hook_group.create_dataset(str(pos), data=self.cluster_labels[index])
                            index += 1
