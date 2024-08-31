


# perform attribution without ussing the logits as the metric, usethe activations of the model instead



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


"""


The process is simple select a feature from a layer, sae and perform attribution wrt to that feature
"""




# Function to get the model activations for each sequence
filter_sae_acts = lambda name: "hook_sae_acts_post" in name 
torch.set_grad_enabled(True)
def get_cache_fwd_and_bwd(model, tokens, metric,feature_index):
    model.reset_hooks()
    cache = {}
    def forward_cache_hook(act, hook):
        if hook.name == "blocks.6.attn.hook_z.hook_sae_acts_post":
            cache[hook.name] = act
        else:
            cache[hook.name] = act.detach()
    model.add_hook(filter_sae_acts, forward_cache_hook, "fwd")

    grad_cache = {}
    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()
    model.add_hook(filter_sae_acts, backward_cache_hook, "bwd")
    

    
    logits = model(tokens)
    downstream = cache["blocks.6.attn.hook_z.hook_sae_acts_post"]

    gradients = torch.zeros_like(downstream)
    gradients[:,:,feature_index] = 1.0
    value = metric(downstream)
    value.backward(gradients)
    model.reset_hooks()
    return value, ActivationCache(cache, model), ActivationCache(grad_cache, model)





def metric_fn(dowstream):
    return dowstream




if __name__ == "__main__":
    model = HookedSAETransformer.from_pretrained("gpt2", device = "cpu")
    data = load_dataset("/home/gerard/MI/pile-10k/", split = "train")
    token_dataset = tokenize_and_concatenate(data, model.tokenizer, max_length=128)
    dataset = DataLoader(token_dataset, batch_size = 4)
    tokens = dataset.dataset[1:2]["tokens"]
    pos = torch.tensor([list(range(97,100))])
    answer_tokens = tokens[:,1:]
    sae_dict = get_attention_sae_dict(layers = [2,6])
    # Add the SAEs to the model
    for name, sae in sae_dict.items():
        sae.use_error_term = True
        model.add_sae(sae)
    value, fwd_cache, bwd_cache = get_cache_fwd_and_bwd(model, tokens, metric_fn,feature_index = 11842 )


