"""
For each Feature

1) Get the computational trace (80% if the attr by pos)
2) Get the explanations
3) Compute per comp distance between the traces of the examples of the same feature
4) Get the top components by layer and type


"""



"""
For all features

1) Get the pairwise cos sim between component traces
2) Get the feaure pairwise cos sim for encoder and decoder
3) Compute a grpah of the accuulated distance
4) Create a gif of the evolution of the graph as we add components

"""




"""
comparison_dict = {}
for i in range(len(feats)):
    for j in range(i+1,len(feats)):
        feat1 = feats[i]
        feat2 = feats[j]
        ex_comp_dict = defaultdict(dict) 
        for ex_idx,ex_trace in comp_traces[feat1].items():
            for ex_idx2,ex_trace2 in comp_traces[feat2].items():
                for comp in ex_trace.keys():
                    dist = torch.nn.functional.cosine_similarity(ex_trace[comp],ex_trace2[comp],dim = 0)
                    ex_comp_dict[comp][(ex_idx,ex_idx2)] = dist
        comparison_dict[(feat1,feat2)] = ex_comp_dict

"""


from sae_training.sparse_autoencoder import SparseAutoencoder
import json
from sae_lens import SAE, SAEConfig, HookedSAETransformer
import torch
import numpy as np
import torch.nn as nn
from jaxtyping import Int, Float 
from datasets import load_dataset
from torch.utils.data import DataLoader, dataset
import tqdm
from collections import defaultdict
from sae_utils import get_attention_sae_dict
from transformer_lens.ActivationCache import ActivationCache
from sae_utils import get_attention_sae_out_dict
import os



def get_explanation(path):
    with open(path, "r") as f:
        data = json.load(f)

    if len(data["explanations"]) == 0:
        return "No explanation"
    explanations = data["explanations"][0]["description"]
    
    if explanations is None:
        return "No explanation"
    return explanations


def get_computational_trace(path):
    # Modigy to get top p
    with open(path, "rb") as f:
        data = torch.load(f)
    all_traces = {} 
    for key,val in data.items():
        all_traces[key] = val.sum(dim = 1)
    return all_traces

    

def return_wenc_w_dec(sae,feature_id):
    W_enc = sae.W_enc[feature_id]
    W_dec = sae.W_dec[feature_id]
    








if __name__ == "__main__":

    transcoder_template  = "/media/workspace/gpt-2-small-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"
    i = 5
    tc = SparseAutoencoder.load_from_pretrained(f"{transcoder_template.format(i)}.pt").eval()

    all_file_names_acts = os.listdir("app_data")
    with open("full_dataset_filter.json","r") as f:
        full_dataset = json.load(f)
    feats_with_acts = [int(file_name.split("_")[0]) for file_name in full_dataset]
    feats_with_data = [int(file_name.split("_")[2]) for file_name in all_file_names_acts]
    feats = list(set(feats_with_acts).intersection(set(feats_with_data)))
    unique_feats = list(set(feats))

    feats_file_dict = {feat: [file_name for file_name in all_file_names_acts if int(file_name.split("_")[2]) == feat] for feat in feats}
    x = torch.load("app_data/"+feats_file_dict[feats[0]][0],map_location = "cpu")
    components = list(x["Mean trace"].keys())
    all_file_names_dataset = os.listdir("../../dataset")
    explanations_dict = {int(file_name.split(".")[0]):get_explanation("../../dataset/"+file_name) for file_name in all_file_names_dataset}
    explanations_dict = {key:val for key,val in explanations_dict.items() if val != "No explanation"}    # Convert to set and    
    explanations_dict = {key:val for key,val in explanations_dict.items() if key in feats}    # Convert to set and    
    feats = list(explanations_dict.keys())
    np.random.seed(42)
    feats = np.random.choice(feats,50,replace = False)
    feats_file_dict = {feat: [file_name for file_name in all_file_names_acts if int(file_name.split("_")[2]) == feat] for feat in feats}
    explanations_dict = {key:val for key,val in explanations_dict.items() if key in feats}    # Convert to set and    
    with open("explanations_dict.json","w") as f:
        json.dump(explanations_dict,f)




if False:
    # Get the computational trace for each feature
    comp_traces = defaultdict(lambda: defaultdict(dict))

# Directory where the files are located

# Iterate through the files in the directory
    for file_name in all_file_names_acts:
        # Extract n and m from the file name
        parts = file_name.split("_")  # Example: "mean_trace_n_m.pt"
        n = int(parts[2])  # Extract n
        m = int(parts[3].split(".")[0])  # Extract m
        
        # Load the tensor dictionary from the file
        file_path = os.path.join("app_data", file_name)
        trace_dict = torch.load(file_path, map_location="cpu")  # Load the .pt file
        
        # Iterate through each component in the loaded dictionary
        for comp, tensor in trace_dict.items():
            # Store the tensor in the nested dictionary
            comp_traces[n][m][comp] = tensor
    comp_traces = dict(comp_traces)
    torch.save(comp_traces,"comp_traces.pt")
    







if False:

    # Get the feaure pairwise cos sim for encoder and decoder
    enc_dict = {}
    dec_dict = {}
    for feat in feats:
        enc_dict[feat] = tc.W_enc.detach()[:,feat].unsqueeze(0)
        dec_dict[feat] = tc.W_dec.detach()[feat].unsqueeze(1)
    enc_sim_mat = torch.zeros((len(feats),len(feats)))
    for i,(feat1,enc1) in enumerate(enc_dict.items()):
        for j,(feat2,enc2) in enumerate(enc_dict.items()):
            enc_sim = torch.nn.functional.cosine_similarity(enc1,enc2,dim = 1)
            enc_sim_mat[i][j] = enc_sim

    dec_sim_mat = torch.zeros((len(feats),len(feats)))
    for i,(feat1,dec1) in enumerate(dec_dict.items()):
        for j,(feat2,dec2) in enumerate(dec_dict.items()):
            dec_sim = torch.nn.functional.cosine_similarity(dec1.T,dec2.T,dim = 1)
            dec_sim_mat[i][j] = dec_sim
    feat_sims = {"enc":enc_sim_mat,"dec":dec_sim_mat,"feats":feats}
    torch.save(feat_sims,"feat_sims.pt")



if False:
    import torch
    from collections import defaultdict

    dist_comp = defaultdict(dict)

    for comp in components:
        for feat, file_names in feats_file_dict.items():
            indices = []  # List to store the indices of non-zero values
            values = []   # List to store the corresponding cosine similarity values
            n = len(file_names)
            
            file_traces = {}
            
            # Load each file once and store it in a dictionary
            for i, file in enumerate(file_names):
                file_traces[file] = torch.load(f"app_data/{file}", map_location="cpu")["Mean trace"][comp]

            # Now compute the pairwise cosine similarity
            for i, file1 in enumerate(file_names):
                trace1 = file_traces[file1]
                for j, file2 in enumerate(file_names):
                    if i <= j:  # Only calculate for the upper triangular matrix
                        trace2 = file_traces[file2]
                        dist = torch.nn.functional.cosine_similarity(trace1, trace2, dim=0)
                        
                        # Only store non-zero values to save memory
                        if dist != 0:  
                            # Add both the upper and lower triangular entries (symmetry)
                            indices.append([i, j])
                            values.append(dist)
                            
                            if i != j:
                                indices.append([j, i])
                                values.append(dist)

            # Convert the indices and values into a sparse tensor
            if values:  # Check if there are any non-zero values
                indices_tensor = torch.tensor(indices, dtype=torch.long).t()  # Transpose to match sparse tensor format
                values_tensor = torch.tensor(values, dtype=torch.float)

                mat_sparse = torch.sparse_coo_tensor(indices_tensor, values_tensor, size=(n, n))
            else:
                # If there are no non-zero values, create an empty sparse matrix
                mat_sparse = torch.sparse_coo_tensor(size=(n, n), dtype=torch.float)

            dist_comp[feat][comp] = mat_sparse

# Convert the sparse tensors to dense format
        for feat,feat_dict in dist_comp.items():
            for comp,tensor in feat_dict.items():
                dist_comp[feat][comp] = tensor.to_dense()

# Save the entire dictionary in a .pt file
        torch.save(dist_comp, "dist_comp.pt")

if False:

    import torch
    from collections import defaultdict

# Step 1: Computing top components
    print("Computing top components")
    comp_aggregations = defaultdict(dict)

# First pass: aggregate components by layer and feature type
    for feats, file_list in feats_file_dict.items():
        aggregation_dict = defaultdict(list)  # Use list directly instead of dict
        for file_name in file_list:
            trace = torch.load("app_data/" + file_name, map_location="cpu")
            for comp in components:
                aggregation_dict[comp].append(trace["Mean trace"][comp])
        
        # Compute the mean for each component
        for c in aggregation_dict.keys():
            stacked_traces = torch.stack(aggregation_dict[c])
            comp_mean = stacked_traces.mean(dim=0)
            
            # Optional: Consider converting the mean tensor to sparse if it's sparse
            if comp_mean.count_nonzero() / comp_mean.numel() < 0.5:  # Assuming sparsity threshold
                comp_mean_sparse = comp_mean.to_sparse()
                comp_aggregations[feats][c] = comp_mean_sparse
            else:
                comp_aggregations[feats][c] = comp_mean  # Store as dense tensor if not sparse

# Step 2: Extract top components
    top_components = defaultdict(dict)

# Select the top-k components with more than zero attribution
    for feat, comp_dict in comp_aggregations.items():
        for comp, comp_trace in comp_dict.items():
            # Convert sparse tensor to dense for processing (necessary for .max() and .topk())
            if comp_trace.is_sparse:
                comp_trace_dense = comp_trace.to_dense()
            else:
                comp_trace_dense = comp_trace
            
            # Apply masking for positive values
            positive_mask = comp_trace_dense > 0
            positive_values = comp_trace_dense[positive_mask]

            # If there are no positive values, create a mask of -1
            if positive_values.numel() == 0:
                masked_indices = torch.full((10,), -1, dtype=torch.long)
            else:
                # Get the top k values, up to 10, where values are positive
                top_k = min(10, positive_values.numel())
                top_k_indices = positive_values.topk(top_k).indices
                
                # Map the indices back to their original positions
                valid_indices = torch.nonzero(positive_mask).squeeze()
                if valid_indices.dim()==0:
                    valid_indices = valid_indices.unsqueeze(0)
                    original_indices = torch.full((10,), -1, dtype=torch.long)
                else:
                    original_indices = valid_indices[top_k_indices]

                # If fewer than 10, mask the remaining positions with -1
                if original_indices.numel() < 10:
                    mask = torch.full((10,), -1, dtype=torch.long)
                    mask[:original_indices.numel()] = original_indices
                    masked_indices = mask
                else:
                    masked_indices = original_indices

            top_components[feat][comp] = masked_indices
# Save the final dictionary as a .pt file
    torch.save(top_components, "top_components.pt")



if False:
    import torch
    import os

# Function to compute in-place aggregated mean
    def accumulate_trace_mean(agg_trace, new_trace, idx):
        if idx == 0:
            agg_trace.copy_(new_trace)  # First element initialization
        else:
            agg_trace.add_(new_trace)  # In-place addition to accumulate
        return agg_trace

# Memory-efficient version: pairwise cosine similarity with lazy loading and sparse tensors
    dist_comp = {}
    for comp in components:
        # Create the matrix to store pairwise cosine similarities
        mat = torch.zeros((len(feats_file_dict), len(feats_file_dict)))

        # Cache aggregated traces to avoid redundant file operations
        agg_traces = {}

        for i, (feat1, file_list1) in enumerate(feats_file_dict.items()):
            if feat1 not in agg_traces:
                # Instead of stacking and mean, use in-place accumulation to reduce memory
                agg_trace1 = None
                for idx, file_name1 in enumerate(file_list1):
                    trace1 = torch.load(os.path.join("app_data", file_name1), map_location="cpu")
                    trace1_component = trace1["Mean trace"][comp].float()  # Convert to float32
                    if idx == 0:
                        agg_trace1 = torch.zeros_like(trace1_component)
                    agg_trace1 = accumulate_trace_mean(agg_trace1, trace1_component, idx)
                agg_trace1.div_(len(file_list1))  # Final mean (in-place division)
                
                # Convert to sparse if beneficial
                if agg_trace1.count_nonzero() / agg_trace1.numel() < 0.5:  # Example sparsity threshold
                    agg_trace1_sparse = agg_trace1.to_sparse()
                else:
                    agg_trace1_sparse = agg_trace1
                agg_traces[feat1] = agg_trace1_sparse  # Cache the result
            
            for j, (feat2, file_list2) in enumerate(feats_file_dict.items()):
                if feat2 not in agg_traces:
                    # Same aggregation logic for feat2
                    agg_trace2 = None
                    for idx, file_name2 in enumerate(file_list2):
                        trace2 = torch.load(os.path.join("app_data", file_name2), map_location="cpu")
                        trace2_component = trace2["Mean trace"][comp].float()  # Convert to float32
                        if idx == 0:
                            agg_trace2 = torch.zeros_like(trace2_component)
                        agg_trace2 = accumulate_trace_mean(agg_trace2, trace2_component, idx)
                    agg_trace2.div_(len(file_list2))  # Final mean (in-place division)

                    # Convert to sparse if beneficial
                    if agg_trace2.count_nonzero() / agg_trace2.numel() < 0.5:
                        agg_trace2_sparse = agg_trace2.to_sparse()
                    else:
                        agg_trace2_sparse = agg_trace2
                    agg_traces[feat2] = agg_trace2_sparse  # Cache the result

                # Use sparse or dense representations for cosine similarity calculation
                agg_trace1 = agg_traces[feat1]
                agg_trace2 = agg_traces[feat2]

                # Ensure dense format for cosine similarity computation
                if agg_trace1.is_sparse:
                    agg_trace1_dense = agg_trace1.to_dense()
                else:
                    agg_trace1_dense = agg_trace1

                if agg_trace2.is_sparse:
                    agg_trace2_dense = agg_trace2.to_dense()
                else:
                    agg_trace2_dense = agg_trace2

                # Compute cosine similarity (reduced precision)
                dist = torch.nn.functional.cosine_similarity(agg_trace1_dense, agg_trace2_dense, dim=0).float()
                mat[i][j] = dist  # Store the distance in the matrix

        dist_comp[comp] = mat

# Save the pairwise distance matrix for each component
    print("Saving the pairwise distance matrix for each component")
    torch.save(dist_comp, "feat_pairwise_dist_comp.pt")

