
import h5py

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
from torch.utils.data import DataLoader, dataset
from tqdm import tqdm
from collections import defaultdict


import h5py

# Function to load attention SAEs

from sae_lens import SAE, SAEConfig
import transformer_lens.utils as utils
def attn_sae_cfg_to_hooked_sae_cfg(attn_sae_cfg,device):
    new_cfg = {
            "model_name":attn_sae_cfg['model_name'],
            "hook_head_index":attn_sae_cfg['act_name'],
            "device": device,
        "architecture":"standard",
        "d_sae": attn_sae_cfg["dict_size"],
        "d_in": attn_sae_cfg["act_size"],
        "activation_fn_str": "relu",
        "apply_b_dec_to_input":True,
        "finetuning_scaling_factor":None,
        "context_size":attn_sae_cfg['seq_len'],
        "hook_name": attn_sae_cfg["act_name"],
        "hook_layer":attn_sae_cfg['layer'],
        "prepend_bos":True,
        "dataset_path":None,
        "dataset_trust_remote_code":None,
        "normalize_activations":None,
        "dtype":"float32",
        "sae_lens_training_version":None



    }
    return new_cfg


def get_attention_sae_dict(layers: Optional[List[Int]],
                           all_layers: Optional[bool] =  False,
                           device = "cpu") -> Dict[str, SAE]:


    if all_layers:
        assert layers is None, "If all_layers is True, layers must be None"
    else:
        assert layers is not None, "If all_layers is False, layers must be provided"

    auto_encoder_runs = [
        "gpt2-small_L0_Hcat_z_lr1.20e-03_l11.80e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
        "gpt2-small_L1_Hcat_z_lr1.20e-03_l18.00e-01_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v5",
        "gpt2-small_L2_Hcat_z_lr1.20e-03_l11.00e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v4",
        "gpt2-small_L3_Hcat_z_lr1.20e-03_l19.00e-01_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
        "gpt2-small_L4_Hcat_z_lr1.20e-03_l11.10e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v7",
        "gpt2-small_L5_Hcat_z_lr1.20e-03_l11.00e+00_ds49152_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
        "gpt2-small_L6_Hcat_z_lr1.20e-03_l11.10e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
        "gpt2-small_L7_Hcat_z_lr1.20e-03_l11.10e+00_ds49152_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
        "gpt2-small_L8_Hcat_z_lr1.20e-03_l11.30e+00_ds24576_bs4096_dc1.00e-05_rsanthropic_rie25000_nr4_v6",
        "gpt2-small_L9_Hcat_z_lr1.20e-03_l11.20e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
        "gpt2-small_L10_Hcat_z_lr1.20e-03_l11.30e+00_ds24576_bs4096_dc1.00e-05_rsanthropic_rie25000_nr4_v9",
        "gpt2-small_L11_Hcat_z_lr1.20e-03_l13.00e+00_ds24576_bs4096_dc3.16e-06_rsanthropic_rie25000_nr4_v9",
    ]

    hf_repo_attn = "ckkissane/attn-saes-gpt2-small-all-layers"

    hook_name_to_sae_attn = {}
    if all_layers:
        for auto_encoder_run in auto_encoder_runs:
            attn_sae_cfg = utils.download_file_from_hf(hf_repo_attn, f"{auto_encoder_run}_cfg.json")
            cfg = attn_sae_cfg_to_hooked_sae_cfg(attn_sae_cfg,device)
            
            state_dict = utils.download_file_from_hf(hf_repo_attn, f"{auto_encoder_run}.pt", force_is_torch=True)
        
            hooked_sae = SAEConfig.from_dict(cfg)
            hooked_sae = SAE(hooked_sae)
            hooked_sae.load_state_dict(state_dict)
            
            hook_name_to_sae_attn[cfg['hook_name']] = hooked_sae
    else:
        for layer in layers:
            auto_encoder_run = auto_encoder_runs[layer]
            attn_sae_cfg = utils.download_file_from_hf(hf_repo_attn, f"{auto_encoder_run}_cfg.json")
            cfg = attn_sae_cfg_to_hooked_sae_cfg(attn_sae_cfg,device)
            
            state_dict = utils.download_file_from_hf(hf_repo_attn, f"{auto_encoder_run}.pt", force_is_torch=True)
        
            hooked_sae = SAEConfig.from_dict(cfg)
            hooked_sae = SAE(hooked_sae)
            hooked_sae.load_state_dict(state_dict)
            
            hook_name_to_sae_attn[cfg['hook_name']] = hooked_sae


    return hook_name_to_sae_attn
