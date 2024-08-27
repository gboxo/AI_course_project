# AI Safety Course Project



import json
from sae_lens import SAE, SAEConfig, HookedSAETransformer
from matplotlib import pyplot as plt
import os
import torch
import numpy as np
from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset
from torch.utils.data import DataLoader, dataset
from tqdm import tqdm
from collections import defaultdict


from filter_tokens import filter_pred, get_correct_sequences
from collect_activations import ActivationsColector
from cluster_activations import SpectralClusteringAnalyzer









"""
TODO List:
    - Add the option to get the activations from the last n tokens before the predictions
    - Add the option to store attributions


"""

# Function to get the model activations for each sequence









if __name__ == "__main__":
    model = HookedSAETransformer.from_pretrained("gpt2")
    data = load_dataset("/home/gerard/MI/pile-10k/", split = "train")

    tokens = tokenize_and_concatenate(data,tokenizer = model.tokenizer, max_length = 128)
    #create_visualization(tokens, "final_dict.json", 4)
    #filter_pred(model,tokens, 4)
    #get_correct_sequences("checkpoints", 3)
    with open("final_dict.json", "r") as f:
        location_dict = json.load(f)
    acts = ActivationsColector(model, tokens, 4, ["blocks.4.hook_attn_out","blocks.5.hook_attn_out"],"Activations",location_dict, cat_activations=False, quantize = True ,average = True, load = True)
    clusters = SpectralClusteringAnalyzer(acts.activations)
    clusters.perform_clustering(3)
    clusters.save_cluster_labels("cluster_labels.h5")








    


