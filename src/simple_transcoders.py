# simple implementation of transcoder hooks




# %%
from sae_lens import SAE, SAEConfig, HookedSAETransformer
from matplotlib import pyplot as plt
import pandas as pd
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
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory 
from transformer_lens import utils



from huggingface_hub import hf_hub_download
from sae_lens import training
import sae_lens.config  as config 
import sys
sys.modules['sae_training'] = training
sys.modules['sae_training.config'] = config 




# %%

model = HookedSAETransformer.from_pretrained("gpt2")

# %%
device = "cpu"
hf_repo_tc = "pchlenski/gpt2-transcoders"
l = 5
auto_encoder_run = f"final_sparse_autoencoder_gpt2-small_blocks.{l}.ln2.hook_normalized_24576"

tc_sae_cfg = utils.download_file_from_hf(hf_repo_tc, f"{auto_encoder_run}_cfg.json")

state_dict = utils.download_file_from_hf(hf_repo_tc, f"{auto_encoder_run}.pt", force_is_torch=True)

hooked_sae = SAEConfig.from_dict(tc_sae_cfg)
hooked_sae = SAE(hooked_sae)
    hooked_sae.load_state_dict(state_dict)
    
    hook_name_to_sae_attn[cfg['hook_name']] = hooked_sae
saes_dict = {}
for l in range(5,7):

    sae, cfg_dict, sparsity = SAE.from_pretrained(
            release = "gpt2-small-mlp-tm",
            sae_id = f"blocks.{l}.hook_mlp_out",
            device = device
            )
    saes_dict[f"blocks.{l}.hook_resid_post"] = sae
    if l == 0:
        cfg = cfg_dict






# %%

# Define the hooks



