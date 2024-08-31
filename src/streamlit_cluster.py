
from h5py._hl.base import default_lapl
import streamlit as st
import torch
import json
from tqdm import tqdm
from torch.utils.data import dataloader
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import defaultdict
import ast

import h5py



def create_streamlit_visualization(token_dataset, model, cluster_keys, cluster_labels, batch_size=256):
    # Load final_dict from json file
    with open("../final_dicts/version_2/final_dict.json", "r") as f:
        final_dict = json.load(f)

    # Create a dataloader for the token dataset
    dataset = DataLoader(token_dataset, batch_size=batch_size)

    # Streamlit title
    st.title("Token Prediction Visualization")

    # Define CSS styles for correct predictions
    st.markdown(
        """
        <style>
        .correct-token {
            background-color: #d1ecf1;
            border-radius: 5px;
            padding: 2px 4px;
            margin: 2px;
        }
        .cluster-token {
            background-color: #ffeeba;
            border-radius: 5px;
            padding: 2px 4px;
            margin: 2px;
        }
        .token {
            display: inline-block;
            margin: 2px;
        }
        .header {
            font-size: 24px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    cluster_keys_dict = defaultdict(list)
    for i in range(len(cluster_keys)):
        cluster_keys_dict[cluster_labels[i]].append(cluster_keys[i])
    # Sidebar for navigation
    st.sidebar.header("Navigation")
    cluster_options = sorted(set(cluster_labels))
    selected_cluster = st.sidebar.selectbox("Select a Cluster", cluster_options)
    if selected_cluster not in cluster_keys_dict:
        st.warning("The cluster has not been found")

    # Important in the current implmentatio the same document can be repeated in the cluster

    seq_pos_list = []
    tuple_list = []

    # Get positions
    for key in cluster_keys_dict[selected_cluster]:
        batch, doc, hook, pos = key.split("/")
        batch = int(batch.split(" ")[1])
        doc = int(doc)

        seq_pos_list.append(batch*batch_size + doc)
        tuple_list.append(tuple(ast.literal_eval(pos)))
    input_ids = torch.stack([dataset.dataset[i]["tokens"] for i in seq_pos_list], dim = 0)
    print(tuple_list)




    






    for doc, seq in enumerate(input_ids):# keep in mind the doc position if global (not inside a batch)
        
        indices = torch.zeros(len(seq))
        str_tokens = [model.to_string(ind) for ind in seq]
        

        if seq.numel() == 0:
            st.warning(f"Document {doc} is empty.")
            continue  # Skip if seq is empty

        # Mark the indices according to final_dict and the selected cluster
        
        tup = tuple_list[doc]
        for j in range(tup[0], tup[1] + 1):
            indices[j] = 1

        # Highlight only tokens that belong to the selected cluster
        
        # Streamlit document header
        st.subheader(f"Document {doc}")

        # Display the tokens with correct/incorrect annotations using columns
        token_display = []
        for token_idx, (token, correct) in enumerate(zip(str_tokens[1:], indices[1:])):
            token_text = html.escape(token)
            if correct == 1:
                token_display.append(f'<span class="correct-token token">{token_text}</span>')
            else:
                token_display.append(f'<span class="token">{token_text}</span>')

        st.markdown(" ".join(token_display), unsafe_allow_html=True)

        # Display the original text below the tokens
        original_text = model.to_string(seq[1:])
        st.markdown(f"**Original text:** `{original_text}`")

# Load model and dataset
model = HookedSAETransformer.from_pretrained("gpt2")
data = load_dataset("/home/gerard/MI/pile-10k/", split="train")
tokens = tokenize_and_concatenate(data, tokenizer=model.tokenizer, max_length=128)

# Load cluster labels (assuming you have saved them previously)

cluster_keys = []
cluster_labels = []
with h5py.File("../clusters/version_2/cluster_labels.h5", "r") as f:
    for batch in f.keys():
        batch_group = f[batch]
        for doc in batch_group.keys():
            doc_group = batch_group[doc]
            for hook in doc_group.keys():
                hook_group = doc_group[hook]
                for pos in hook_group.keys():
                    key = f"{batch}/{doc}/{hook}/{pos}"
                    clust = hook_group[pos][()]
                    cluster_labels.append(clust)
                    cluster_keys.append(key)




# Create visualization
create_streamlit_visualization(tokens, model, cluster_keys,cluster_labels, batch_size=4)
