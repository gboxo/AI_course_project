# %%    
from threading import active_count
from sae_lens import SAE, SAEConfig, HookedSAETransformer
import torch
from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset
import json 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sae_utils import get_attention_sae_dict
from html_visualization import create_visualization_loss, create_visualization_plotly

"""
Mullti Token Circuits

What are multlitoken circuits in the context of max  activating dataset examples for a given attention features.


It's very usual that a given feature (attention or otherwise) is active in several tokens in a prompt, and that the (DFA destination token) is outside those tokens in which the feature is active.

A common approach is to either average the activations over the token positions or just use the first token in which the feature is activates.


In this work this apporach changes and the focus is draw to all the multiple tokens in which the feature is active not just the first.


To do so  we have several approaches, but from first principles the best approach seems to iterate over the token positions and for each positions search for shallow circuits, as the position index progresses inside the fewe tokens where the feature is active, we will reference circuit formations from the previous tokens.


This introduces a tradeoff because a very similar thing can be naively done by just attributing from rhs tokens, the problem is that this attribution is not faithful at all with the causal attention mask and AR nature of language models.




"""

# %%
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Load the tokens
with open("5-att-kk-148.json", "r") as f:
    feat_dict = json.load(f)


strings = ["".join(elem["tokens"]) for elem in feat_dict["activations"]]

model = HookedSAETransformer.from_pretrained("gpt2", device = device)


tokens = [model.to_tokens(string, prepend_bos=False) for string in strings]



sae_dict = get_attention_sae_dict([5],device = device)

# %%

acts = []
filter_sae_acts = lambda name: "hook_sae_acts_post" in name 
with  torch.no_grad():
    for tok in tokens:
         _,cache = model.run_with_cache_with_saes(tok,saes = [sae  for _,sae in sae_dict.items() ], names_filter = filter_sae_acts)
         acts.append(cache["blocks.5.attn.hook_z.hook_sae_acts_post"][0,:,148])

# filter the prompts with no activations

tok_is_active = [tok for act,tok in zip(acts,tokens) if act.sum() != 0]
active_tensor = [(act != 0) for act in acts ]


# get the per token loss for the prompts in which the feature is acive
all_losses = []
with torch.no_grad():
    for tok in tok_is_active:
        per_token_loss = model(tok, loss_per_token = True, return_type = "loss")
        all_losses.append(per_token_loss)


# %%

create_visualization_plotly(model,tok_is_active,all_losses, active_tensor)










