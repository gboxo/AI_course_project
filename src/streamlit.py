import streamlit as st
import torch
import json
from tqdm import tqdm
from torch.utils.data import dataloader
from typing import List,Optional
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


import h5py


def create_streamlit_visualization(token_dataset, model, batch_size=256):
    # load final_dict from json file
    with open("final_dict.json", "r") as f:
        final_dict = json.load(f)

    # create a dataloader for the token dataset
    dataset = DataLoader(token_dataset, batch_size=batch_size)

    # streamlit title
    st.title("token prediction visualization")

    # define css styles for correct predictions
    st.markdown(
        """
        <style>
        .correct-token {
            background-color: #d1ecf1;
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

    # sidebar for navigation
    st.sidebar.header("navigation")
    batch_options = list(final_dict.keys())
    selected_batch = st.sidebar.selectbox("select a batch", batch_options)

    # determine selected batch index
    batch_index = int(selected_batch.split(" ")[1])

    # check if the current batch is in final_dict
    if f"Batch {batch_index}" not in final_dict:
        st.warning("this batch does not exist in the final dictionary.")
        return

    batch_dict = final_dict[f"Batch {batch_index}"]
    input_ids = dataset.dataset[batch_index*batch_size:(batch_index+1)*batch_size]['tokens']

    st.header(f"Batch {batch_index}")
    for doc, seq in enumerate(input_ids):
        doc_list = batch_dict.get(str(doc), [])
        if not doc_list:
            continue
        indices = torch.zeros(len(seq))
        str_tokens = [model.to_string(ind) for ind in seq]
        if isinstance(seq, torch.Tensor) and seq.dim() == 0:
            st.warning(f"document {doc} has no tokens.")
            continue  # skip this document if seq is a scalar tensor

        if seq.numel() == 0:
            st.warning(f"document {doc} is empty.")
            continue  # skip if seq is empty


        # mark the indices according to final_dict
        for tup in doc_list:
            for j in range(tup[0], tup[1] + 1):
                indices[j] = 1

        # streamlit document header
        st.subheader(f"document {doc}")




        # display the tokens with correct/incorrect annotations using columns
        token_display = []
        for token, correct in zip(str_tokens[1:], indices[1:]):
            token_text = html.escape(token)
            if correct == 1:
                token_display.append(f'<span class="correct-token token">{token_text}</span>')
            else:
                token_display.append(f'<span class="token">{token_text}</span>')

        st.markdown(" ".join(token_display), unsafe_allow_html=True)

        # display the original text below the tokens
        original_text = model.to_string(seq[1:])
        st.markdown(f"**original text:** `{original_text}`")











  



model = HookedSAETransformer.from_pretrained("gpt2")
data = load_dataset("/home/gerard/MI/pile-10k/", split = "train")

tokens = tokenize_and_concatenate(data,tokenizer = model.tokenizer, max_length = 128)
create_streamlit_visualization(tokens, model, batch_size=256)




