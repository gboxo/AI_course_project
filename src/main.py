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


from filter_tokens import PredictionFilter
from collect_activations import ActivationsColector
from cluster_activations import SpectralClusteringAnalyzer









"""
TODO List:
    - Add the option to get the activations from the last n tokens before the predictions
    - Add the option to store attributions


"""

# Function to get the model activations for each sequence








if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedSAETransformer.from_pretrained("gpt2", device = device)
    data = load_dataset("/home/gerard/MI/pile-10k/", split = "train")
    tokens = tokenize_and_concatenate(data,tokenizer = model.tokenizer, max_length = 128)


    # ======== Get the predictions and the tokens =========


    pred_filt = PredictionFilter(model, batch_size = 4, checkpoint_dir = "../checkpoints",final_dicts_dir = "../final_dicts")
    pred_filt.filter_predictions(tokens,save = True,strict = False,threshold = 0.1)
    pred_filt.get_correct_sequences(3)# The sequences of contiguous correct predictions must be at least 3 tokens long
    final_dicts_dir = pred_filt.final_dicts_dir_versioned


    acts = ActivationsColector(model, tokens, ["blocks.4.hook_attn_out","blocks.5.hook_attn_out"],"Features","../activations/",final_dicts_dir, cat_activations=False, quantize = True ,average = True, load = False)
    clusters = SpectralClusteringAnalyzer(acts.activations, "../clusters/")
    clusters.perform_clustering(3)
    clusters.save_cluster_labels()












    


