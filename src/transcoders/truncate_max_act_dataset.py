# Get the max activating dataset examples for each feature for 5.Attn.SAE
# Right truncate each prompt from the right (get till the max act token)
# Left truncate each prompt to achieve 80% of the original max activation

from tc_utils import apply_sae, detach_at
from tc_modules import run_with_ref_cache
from sae_training.sparse_autoencoder import SparseAutoencoder
import json
from sae_lens import SAE, SAEConfig, HookedSAETransformer
import torch
import numpy as np
import torch.nn as nn
from jaxtyping import Int, Float 
from datasets import load_dataset
from torch.utils.data import DataLoader, dataset
from tqdm import tqdm
from collections import defaultdict
from sae_utils import get_attention_sae_dict
from transformer_lens.ActivationCache import ActivationCache
from sae_utils import get_attention_sae_dict
import os

"""
The process is simple select a feature from a layer, sae and perform attribution wrt to that feature
"""

def get_max_act_dataset(model,tc, feature_id,path):

    with open(path, "r") as f:
        data = json.load(f)
    tok_list = ["".join([tok if model.to_tokens(tok,prepend_bos = False).shape[1]==1 else "<|endoftext|>" for tok in elem["tokens"] ]) for i,elem in enumerate(data["activations"]) if i%2==0]
    tokens = {i:model.to_tokens(tok, prepend_bos = False) for i,tok in enumerate(tok_list)}
    max_positions = []

    candidates =   ["blocks.5.hook_mlp_out.sae.hook_hidden_post"]

    all_saes = [tc]
    max_positions = []
    for i,toks in tokens.items():

        with apply_sae(model, all_saes):
            name_filter = lambda x: x in candidates
            _, cache = run_with_ref_cache(model,toks = toks, names_filter=name_filter)
            max_pos = cache["blocks.5.hook_mlp_out.sae.hook_hidden_post"][0][:,feature_id].argmax().item()
            max_positions.append(max_pos)


    right_truncated = {i:(idx,seq[0,:idx].tolist()) for idx,(i,seq) in zip(max_positions,tokens.items()) if idx>0}
    
    return right_truncated


if __name__ == "__main__":
    model = HookedSAETransformer.from_pretrained("gpt2", device = "cpu")
    transcoder_template  = "/media/workspace/gpt-2-small-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"
    threshold = 0
    i = 5
    tc = SparseAutoencoder.load_from_pretrained(f"{transcoder_template.format(i)}.pt").eval()
    all_file_names = os.listdir("../../dataset/")
    r_truncated_seqs = defaultdict(list) 
    
    for file_name in tqdm(all_file_names):
        feature_id = int(file_name.split(".")[0])
        r_truncated_seqs[feature_id] = get_max_act_dataset(model,tc,feature_id,"../../dataset/"+file_name)

    with open("full_dataset.json","w") as f:
        json.dump(r_truncated_seqs,f)


    with open("full_dataset.json","r") as f:
        d = json.load(f)

