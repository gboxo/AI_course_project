import streamlit as st
import torch
import json
from torch.utils.data import DataLoader
from collections import defaultdict
import ast
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset
from sae_lens import HookedSAETransformer
import html

def sanitize_text(text: str) -> str:
    """Decode HTML special characters and then escape the text."""
    decoded_text = html.unescape(text)
    return html.escape(decoded_text)

def load_data():
    with open("../final_dicts/version_2/final_dict.json", "r") as f:
        final_dict = json.load(f)
    
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
    
    return final_dict, cluster_keys, cluster_labels

def create_cluster_statistics(cluster_keys_dict, selected_cluster):
    num_docs = len(cluster_keys_dict[selected_cluster])
    num_tokens = sum([len(key.split("/")) for key in cluster_keys_dict[selected_cluster]])
    token_counts = [len(key.split("/")) for key in cluster_keys_dict[selected_cluster]]
    
    return num_docs, num_tokens, token_counts

def plot_token_distribution(token_counts):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(token_counts, bins=range(1, max(token_counts) + 1), kde=True, ax=ax)
    ax.set_title("Tokens per Document in Cluster")
    ax.set_xlabel("Number of Tokens")
    ax.set_ylabel("Number of Documents")
    return fig

def display_document(doc, seq, indices, str_tokens, model):
    st.subheader(f"Document {doc + 1}")

    token_display = []
    for token_idx, (token, correct) in enumerate(zip(str_tokens[1:], indices[1:])):
        token_text = sanitize_text(token)
        if correct == 1:
            token_display.append(f'<span class="correct-token token">{token_text}</span>')
        else:
            token_display.append(f'<span class="token">{token_text}</span>')

    st.markdown(" ".join(token_display), unsafe_allow_html=True)

    original_text = model.to_string(seq[1:])
    st.markdown(f"**Original text:** `{sanitize_text(original_text)}`")

    with st.expander(f"Token details for Document {doc + 1}"):
        token_details = []
        for token_idx, (token, correct) in enumerate(zip(str_tokens[1:], indices[1:])):
            token_text = sanitize_text(token)
            status = "Correct" if correct == 1 else "Incorrect"
            token_details.append(f"Token {token_idx + 1}: {token_text} ({status})")
        st.write("\n".join(token_details))

def create_streamlit_visualization(token_dataset, model, cluster_keys, cluster_labels, batch_size=256):
    st.set_page_config(layout="wide", page_title="Token Prediction Visualization")
    
    st.title("Token Prediction Visualization")

    st.markdown(
        """
        <style>
        .correct-token { background-color: #d1ecf1; border-radius: 5px; padding: 2px 4px; margin: 2px; }
        .cluster-token { background-color: #ffeeba; border-radius: 5px; padding: 2px 4px; margin: 2px; }
        .token { display: inline-block; margin: 2px; }
        .header { font-size: 24px; margin-bottom: 10px; }
        .stExpander { border: 1px solid #e0e0e0; border-radius: 5px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    dataset = DataLoader(token_dataset, batch_size=batch_size)

    cluster_keys_dict = defaultdict(list)
    for i in range(len(cluster_keys)):
        cluster_keys_dict[cluster_labels[i]].append(cluster_keys[i])

    st.sidebar.header("Navigation")
    cluster_options = sorted(set(cluster_labels))
    
    search_query = st.sidebar.text_input("Search clusters", "")
    filtered_clusters = [cluster for cluster in cluster_options if search_query.lower() in cluster.lower()]
    
    selected_cluster = st.sidebar.selectbox("Select a Cluster", filtered_clusters)

    if selected_cluster not in cluster_keys_dict:
        st.warning("The selected cluster has not been found.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header(f"Cluster: '{sanitize_text(selected_cluster)}'")
        num_docs, num_tokens, token_counts = create_cluster_statistics(cluster_keys_dict, selected_cluster)
        st.metric("Number of Documents", num_docs)
        st.metric("Total Tokens", num_tokens)
        st.metric("Average Tokens per Document", round(num_tokens / num_docs, 2))

    with col2:
        st.subheader("Token Distribution")
        fig = plot_token_distribution(token_counts)
        st.pyplot(fig)

    seq_pos_list = []
    tuple_list = []

    for key in cluster_keys_dict[selected_cluster]:
        batch, doc, hook, pos = key.split("/")
        batch = int(batch.split(" ")[1])
        doc = int(doc)
        seq_pos_list.append(batch * batch_size + doc)
        tuple_list.append(tuple(ast.literal_eval(pos)))

    input_ids = torch.stack([dataset.dataset[i]["tokens"] for i in seq_pos_list], dim=0)

    doc_search_query = st.text_input("Search documents", "")
    
    for doc, seq in enumerate(input_ids):
        indices = torch.zeros(len(seq))
        str_tokens = [model.to_string(ind) for ind in seq]

        if seq.numel() == 0:
            continue

        tup = tuple_list[doc]
        for j in range(tup[0], tup[1] + 1):
            indices[j] = 1

        original_text = model.to_string(seq[1:])
        
        if doc_search_query.lower() not in original_text.lower():
            continue

        display_document(doc, seq, indices, str_tokens, model)

def main():
    model = HookedSAETransformer.from_pretrained("gpt2")
    data = load_dataset("/home/gerard/MI/pile-10k/", split="train")
    tokens = tokenize_and_concatenate(data, tokenizer=model.tokenizer, max_length=128)

    final_dict, cluster_keys, cluster_labels = load_data()

    create_streamlit_visualization(tokens, model, cluster_keys, cluster_labels, batch_size=4)

if __name__ == "__main__":
    main()

