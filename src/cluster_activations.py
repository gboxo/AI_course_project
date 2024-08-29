
import h5py
import numpy as np
import os
from sklearn.cluster import SpectralClustering, KMeans

class ClusteringAnalyzer:
    def __init__(self, activations: dict, clusters_dir: str):
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
                        if len(act.shape) == 3:
                            act = act.flatten()
                            all_activations.append(act)
                        elif len(act.shape) == 2:
                            act = act.flatten()
                            all_activations.append(act)
        return np.array(all_activations)

    def _create_versioned_dir(self, base_dir) -> str:
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        version = 0
        while True:
            versioned_dir = os.path.join(base_dir, f"version_{version}")
            if not os.path.exists(versioned_dir):
                os.makedirs(versioned_dir)
                return versioned_dir
            version += 1

    def preprocess_data(self, flattened_activations: np.ndarray) -> np.ndarray:
        """Preprocess the data by computing angular similarity."""
        # Compute cosine similarity
        norms = np.linalg.norm(flattened_activations, axis=1, keepdims=True)
        cosine_sim = np.dot(flattened_activations, flattened_activations.T) / (norms @ norms.T)

        # Clip to avoid numerical issues
        cosine_sim = np.clip(cosine_sim, -1, 1)  # Ensure values are within the valid range for arccos

        # Compute angular similarity
        angular_similarity = 1 - (np.arccos(cosine_sim) / np.pi)

        # Clip angular similarity to ensure it is within [0, 1]
        angular_similarity = np.clip(angular_similarity, 0, 1)

        return angular_similarity

    def perform_clustering(self, n_clusters: int, method: str = 'spectral'):
        """Perform clustering on the activations."""
        # Flatten the activations to a 2D array for clustering
        flattened_activations = self.flatten_activations()
        
        if method == 'spectral':
            # Preprocess data for spectral clustering
            processed_data = self.preprocess_data(flattened_activations)
            # Perform spectral clustering
            clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
            self.cluster_labels = clustering.fit_predict(processed_data)
        
        elif method == 'kmeans':
            # Perform K-Means clustering
            clustering = KMeans(n_clusters=n_clusters)
            self.cluster_labels = clustering.fit_predict(flattened_activations)

        else:
            raise ValueError("Invalid clustering method. Choose 'spectral' or 'kmeans'.")

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

