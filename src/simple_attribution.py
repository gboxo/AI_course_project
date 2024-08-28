


import h5py
import os
from sae_lens import SAE, SAEConfig, HookedSAETransformer
from matplotlib import pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from jaxtyping import Int, Float 
from transformer_lens.utils import tokenize_and_concatenate, get_act_name
from datasets import load_dataset
from torch.utils.data import DataLoader, dataset
from tqdm import tqdm
from collections import defaultdict
from sae_utils import get_attention_sae_dict
from transformer_lens.ActivationCache import ActivationCache
from typing import List,Dict, Tuple,Any, Optional, Literal
from jaxtyping import Int, Float 
from torch import Tensor
import h5py
from sae_utils import get_attention_sae_dict



# Function to get the model activations for each sequence
torch.set_grad_enabled(True)
filter_sae_acts = lambda name: ("hook_sae_acts_post" in name)
def get_cache_fwd_and_bwd(model, tokens, metric,pos:Tensor,answer_tokens:Tensor):
    model.reset_hooks()
    cache = {}
    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()
    model.add_hook(filter_sae_acts, forward_cache_hook, "fwd")

    grad_cache = {}
    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()
    model.add_hook(filter_sae_acts, backward_cache_hook, "bwd")
    # Set gradients at the specified positions to 1
    

    
    logits = model(tokens)

    gradients = torch.zeros_like(logits[:,1:,:])
    for i in range(pos.size(0)):  # Iterate over the batch size
        for j in (pos[i,:]-1).tolist():  # Iterate over the positions
            gradients[i, j, answer_tokens[i, j]] = 1



    value = metric(model,logits,answer_tokens)
    value.backward(gradients)
    model.reset_hooks()
    return value, ActivationCache(cache, model), ActivationCache(grad_cache, model)


def neg_log_prob(model,logits,answer_tokens):
    assert len(answer_tokens.shape) == 2, "The answer tokens must be a 2 dim tensor (batch seq_len-1)"
    assert len(logits.shape) == 3, "The logits must be a 3 dim tensor (batch position vocab)"
    log_probs = logits.log_softmax(dim=-1)
    return -log_probs[:,:-1,:] # Shape (batch, seq_len-1)


def attr_patch_sae_acts(
        act_cache: ActivationCache,
        grad_cache: ActivationCache,
        site:str,
        layer: int,
        ):
    sae_acts = act_cache[get_act_name(site,layer)+".hook_sae_acts_post"]
    sae_grads = grad_cache[get_act_name(site,layer)+".hook_sae_acts_post"]
    sae_act_attr = sae_acts * sae_grads
    return sae_act_attr


if __name__ == "__main__":
    model = HookedSAETransformer.from_pretrained("gpt2", device = "cpu")
    data = load_dataset("/home/gerard/MI/pile-10k/", split = "train")
    token_dataset = tokenize_and_concatenate(data, model.tokenizer, max_length=128)
    dataset = DataLoader(token_dataset, batch_size = 4)
    tokens = dataset.dataset[1:2]["tokens"]
    pos = torch.tensor([list(range(97,100))])
    answer_tokens = tokens[:,1:]
    sae_dict = get_attention_sae_dict(layers = [0,2,3,4])
    # Add the SAEs to the model
    for name, sae in sae_dict.items():
        sae.use_error_term = True
        model.add_sae(sae)
    value, fwd_cache, bwd_cache = get_cache_fwd_and_bwd(model, tokens, neg_log_prob,pos,answer_tokens)
    sae_act_attr = attr_patch_sae_acts(fwd_cache, bwd_cache, "z", 4)











