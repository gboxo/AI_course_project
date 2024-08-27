# AI Safety Course Project



# %%
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
from transformers.pipelines.base import get_framework

# %%







def filter_pred(token_dataset, batch_size, strict = False, threshold = 0.1):

    if not strict:
        assert threshold is not None, "Threshold must be provided if strict is False"
    dataset = DataLoader(token_dataset, batch_size = batch_size)
    check_point_dict = defaultdict(dict) 


    for i,d in tqdm(enumerate(dataset)):
        if i == 10:
            break
        input_ids = d['tokens']
        with torch.no_grad():
            logits = model(input_ids)


        logits = logits[:,:-1].contiguous()
        target_ids = input_ids[:,1:].contiguous()
        if strict:
            argmax_pred = torch.argmax(logits, dim = -1)
            pred = (argmax_pred == target_ids).float()
            pred_pos = torch.where(pred)
        else:
            loss = nn.CrossEntropyLoss(reduction = "none")
            loss_val = loss(logits.view(-1,logits.size(-1)), target_ids.view(-1))
            loss_val = loss_val.view(logits.size(0),logits.size(1))
            pred = (loss_val < threshold).float()
            pred_pos = torch.where(pred)




        dict_doc_pos = defaultdict(list)
        for doc,ind in zip(pred_pos[0], pred_pos[1]):
            dict_doc_pos[doc.item()].append(ind.item())

        check_point_dict[f"Batch {i}"] = dict_doc_pos 
        if strict:
            with open(f"checkpoints/pred_checkpoint_{i}.pt", "wb") as f:
                torch.save(check_point_dict, f)
        else:
            with open(f"checkpoints/pred_checkpoint_{i}_threshold_{threshold}.pt", "wb") as f:
                torch.save(check_point_dict, f)








# %%

# Get sequences of correct predictions

def get_correct_sequences(checkpoint_folder:"str", seq_leng:int):


    all_checkpoints = os.listdir(checkpoint_folder)
    all_contiguous_positions = {}


    for checkpoint in all_checkpoints:

        with open(checkpoint_folder+"/"+checkpoint, "rb") as f:
            checkpoint_dict = torch.load(f)



        for batch,pred_dict in checkpoint_dict.items():
            all_contiguous_positions[batch] = defaultdict(list)
            for doc, pos_list in pred_dict.items():
                # check for contiguous positions (eg 15 16)
                if len(pos_list) == 0:
                    continue
                contiguous_positions = []
                start = pos_list[0]
                end = pos_list[0]
                for i in range(1,len(pos_list)):
                    if pos_list[i] == end + 1:
                        end = pos_list[i]
                    else:
                        if end - start >= seq_leng:
                            contiguous_positions.append((start,end))
                        start = pos_list[i]
                        end = pos_list[i]

                all_contiguous_positions[batch][doc] = contiguous_positions

        
    with open("final_dict.json", "w") as f:
        json.dump(all_contiguous_positions,f)

     
# %%
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











 
    

# %%
"""
TODO List:
    - Add the option to get the activations from the last n tokens before the predictions
    - Add the option to store attributions


"""

# Function to get the model activations for each sequence


from transformer_lens.hook_points import HookPoint
class ActivationsColector:

    def __init__(self,
                 model:HookedSAETransformer,
                 dataset,
                 ctx_len: int,
                 modules: List[str],
                 type_activations:  Literal["Activations", "Features", "Activations DE","Features DE"],
                 location_dictionary: dict,
                 cat_activations: bool = False,
                 quantize: bool = True,
                 average: bool = True,
                 load: bool = True

                 ):



        self.dataset = DataLoader(dataset, batch_size = 4)
        self.model = model  
        self.ctx_len = ctx_len
        self.modules = modules
        self.modules = modules 
        self.cat_activations = cat_activations
        self.location_dictionary = location_dictionary
        self.quantize = quantize
        self.average = average
        self.type_activations = type_activations
        


        self.type_checking()


        if load:
            self.load_activations()
        else:
            self.activations = self.collect_activations()
            self.save_activations()
        if self.cat_activations:
            assert self.average == True, "If cat_activations they must be averaged"
            self.get_cat_modules()



    def type_checking(self):
        self.act_shapes = []
        if self.type_activations == "Activations":
            with torch.no_grad():
                _,cache = model.run_with_cache(" ")
            for hook in self.modules:
                assert isinstance(cache[hook], torch.Tensor), "The module must return a torch.Tensor"
                if self.average:
                    self.act_shapes.append(cache[hook].mean(dim = 1).shape)
                
                else:
                    self.act_shapes.append(cache[hook].shape)

                if self.cat_activations:
                    size = sum([x[1] for x in self.act_shapes])
                    self.act_shapes = [torch.empty( 1,size).shape]
                        

        if self.type_activations == "Features":
            for module in self.modules:
                assert "attn" in module, "For now only Attention SAEs are supported"

            layers = []
            for module in self.modules:
                layers.append(int(module.split(".")[1].replace("L","")))


            self.saes_dict = get_attention_sae_dict(layers,device = "cpu")
            

            with torch.no_grad():
                _,cache = model.run_with_saes(" ",saes = [self.saes_dict.values()])
            for hook in self.modules:
                assert isinstance(cache[hook+".hook_sae_acts_post"], torch.Tensor), "The module must return a torch.Tensor"
                if self.average:
                    self.act_shapes.append(cache[hook+".hook_sae_acts_post"].mean(1).shape)
                else:
                    self.act_shapes.append(cache[hook+".hook_sae_acts_post"].shape)






            
    def get_cat_modules(self):
        assert self.cat_activations, "Only works if cat_activations is True"
        if self.type_activations == "Activations":
            self.simp_modules = "/".join([".".join([e.replace("blocks","").replace("hook","") for e in x.split(".")]) for x in self.modules])
        elif self.type_activations == "Features":
            self.simp_modules = "/".join([".".join([e.replace("blocks","").replace("hook","") for e in x.split(".")]+["hook_sae_acts_post"]) for x in self.modules])
            




    def collect_activations(self):

        activations = {}
        postfix = ""
        if self.type_activations == "Features":
            postfix = ".hook_sae_acts_post"

        for i,d in tqdm(enumerate(self.dataset)):# Select the batch

            batch_activations_dict = {}
            input_ids = d['tokens']
            if f"Batch {i}" not in self.location_dictionary:
                break
            batch_dict = self.location_dictionary[f"Batch {i}"]
            for doc, seq in enumerate(input_ids):# select the document
                doc_list = batch_dict[str(doc)]
                with torch.no_grad():
                    _,cache = model.run_with_cache(seq)# Get all the activations (this needs to be changed)
                    act_seq = {}
                    for hook in self.modules:
                        act = cache[hook+postfix]
                        act_hook_pos = {}
                        for tup in doc_list:
                            x = act[:,tup[0]:tup[1]+1]
                            if self.average: 
                                x = x.mean(dim = 1)
                            act_hook_pos[str(tup)]= x.to("cpu").numpy()# Add the activations for the correct predictions positions
                        act_seq[hook] = act_hook_pos # Add the activations for the module
                if self.cat_activations:
                    cat_act = torch.cat([v for v in act_seq.values()], dim = 1)
                    
                    act_seq[self.simp_modules] = cat_act.to("cpu").numpy()
                batch_activations_dict[doc] = act_seq # Add the activations for the document
            activations[f"Batch {i}"] = batch_activations_dict

        return activations

    def save_activations(self, compression = "gzip", chunks_size = None):

        if self.quantize:
           self.quantize_activations()
        def flatten_activations(activations_dict):
            """Helper function to flatten the nested activations dictionary."""
            for batch, batch_dict in activations_dict.items():  # Get the batch
                for doc, doc_list in batch_dict.items():  # Get the document
                    for hook, hook_dict in doc_list.items():  # Get the hook
                        for pos, act in hook_dict.items():
                            key = f"{batch}/{doc}/{hook}/{pos}"
                            yield key, act

        with h5py.File(f"activations_{self.type_activations}_{self.cat_activations}.h5", "w") as f:
            for key, act in flatten_activations(self.activations):
                f.create_dataset(key, data=act, compression=compression, chunks=chunks_size)


    def quantize_activations(self):
        quantized_activations = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for batch, batch_dict in self.activations.items():# Get the batch
            for doc, doc_list in batch_dict.items():# Get the document
                for hook, hook_dict in doc_list.items():# Get the hook
                    for pos, act in hook_dict.items():
                        quantized_activations[batch][doc][hook][pos]= act.astype("float16")    
        self.activations = quantized_activations


    def load_activations(self, filename="activations.h5"):
        def nested_dict():
            """Helper function to create a nested defaultdict."""
            return defaultdict(nested_dict)

        # Initialize an empty nested dictionary
        activations = nested_dict()

        with h5py.File(filename, "r") as f:
            for batch in f.keys():
                print(batch)
                batch_group = f[batch]
                for doc in batch_group.keys():
                    doc_group = batch_group[doc]
                    for hook in doc_group.keys():
                        hook_group = doc_group[hook]
                        for pos in hook_group.keys():
                            key = f"{batch}/{doc}/{hook}/{pos}"
                            act = hook_group[pos][()]
                            # Store the data in the nested dictionary
                            activations[batch][doc][hook][pos] = act

        # Convert nested defaultdict to normal dicts for easier access
        def convert_to_dict(d):
            if isinstance(d, defaultdict):
                d = {k: convert_to_dict(v) for k, v in d.items()}
            return d

        self.activations = convert_to_dict(activations)







# %%


    

from sklearn.cluster import SpectralClustering

class SpectralClusteringAnalyzer:
    def __init__(self, activations: Dict):
        self.activations = activations
        self.cluster_labels = None

    def flatten_activations(self) -> np.ndarray:
        """Flatten the nested activations dictionary into a 2D array."""
        all_activations = []
        for batch, batch_dict in self.activations.items():
            for doc, doc_list in batch_dict.items():
                for hook, hook_dict in doc_list.items():
                    for pos, act in hook_dict.items():
                        all_activations.append(act.flatten())
        return np.array(all_activations)

    def perform_clustering(self, n_clusters: int, affinity: str = 'nearest_neighbors'):
        """Perform spectral clustering on the activations."""
        # Flatten the activations to a 2D array for clustering
        flattened_activations = self.flatten_activations()
        
        # Perform spectral clustering
        clustering = SpectralClustering(n_clusters=n_clusters, affinity=affinity)
        self.cluster_labels = clustering.fit_predict(flattened_activations)

    def get_cluster_labels(self) -> np.ndarray:
        """Get the cluster labels assigned to the activations."""
        if self.cluster_labels is None:
            raise ValueError("Clustering has not been performed yet. Call 'perform_clustering' first.")
        return self.cluster_labels

    def save_cluster_labels(self, filename: str = "cluster_labels.h5"):
        """Save the cluster labels to an HDF5 file with the same structure as activations."""
        if self.cluster_labels is None:
            raise ValueError("Clustering has not been performed yet. Call 'perform_clustering' first.")
        
        # Create a mapping from the flattened structure to the original structure
        index = 0
        with h5py.File(filename, "w") as f:
            for batch, batch_dict in self.activations.items():
                batch_group = f.create_group(batch)
                for doc, doc_list in batch_dict.items():
                    doc_group = batch_group.create_group(str(doc))
                    for hook, hook_dict in doc_list.items():
                        hook_group = doc_group.create_group(hook)
                        for pos, act in hook_dict.items():
                            # Here we save the cluster label corresponding to the current position
                            hook_group.create_dataset(str(pos), data=self.cluster_labels[index])
                            index += 1









if __name__ == "__main__":
    model = HookedSAETransformer.from_pretrained("gpt2")
    data = load_dataset("/home/gerard/MI/pile-10k/", split = "train")

    tokens = tokenize_and_concatenate(data,tokenizer = model.tokenizer, max_length = 128)
#create_visualization(tokens, "final_dict.json", 4)
    #filter_pred(tokens, 4)
    #get_correct_sequences("checkpoints", 3)
    with open("final_dict.json", "r") as f:
        location_dict = json.load(f)
    acts = ActivationsColector(model, tokens, 4, ["blocks.4.hook_attn_out","blocks.5.hook_attn_out"],"Activations",location_dict, cat_activations=False, quantize = True ,average = True, load = True)
    clusters = SpectralClusteringAnalyzer(acts.activations)
    clusters.perform_clustering(3)
    clusters.save_cluster_labels("cluster_labels.h5")








    


