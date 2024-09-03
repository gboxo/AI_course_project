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
import einops


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
"""
all_losses = []
with torch.no_grad():
    for tok in tok_is_active:
        per_token_loss = model(tok, loss_per_token = True, return_type = "loss")
        all_losses.append(per_token_loss)
"""

# %%

#create_visualization_plotly(model,tok_is_active,all_losses, active_tensor)




# %%

# For every one of the prompts in which the feature fires get the first position in which the fature fires and perform logit lens

all_toks = model.to_tokens(strings, prepend_bos = False)
with  torch.no_grad():
    _,cache = model.run_with_cache_with_saes(all_toks,saes = [sae  for _,sae in sae_dict.items() ], stop_at_layer = 6)


# Get the first position in which the feature is active for each prompt




accumulated_residual, labels = cache.accumulated_resid(
        layer = 5,incl_mid=False,  return_labels=True,apply_ln = False)


first_pos = torch.argmax((cache["blocks.5.attn.hook_z.hook_sae_acts_post"][:,:,148]!=0).to(torch.int), dim =  -1)

results = []
for i,pos in enumerate(first_pos):
    x = accumulated_residual[:,i,pos,:]
    results.append(x)

result = torch.stack(results)[:,:5,:]
feat_dir = sae_dict["blocks.5.attn.hook_z"].W_enc.detach()[:,148]

feature_attribution = einops.einsum(result,feat_dir, "batch layer d_model, d_model -> batch layer " )



sns.boxplot(feature_attribution)
plt.ylim(0,10)
plt.show()










