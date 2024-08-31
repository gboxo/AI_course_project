

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
from sae_utils import get_attention_sae_dict
from transformer_lens.ActivationCache import ActivationCache
from typing import List,Dict, Tuple,Any, Optional, Literal
from jaxtyping import Int, Float 
from torch import Tensor
import h5py



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
    gradients = torch.zeros_like(tokens)
    gradients[0,pos] = 1
    logits = model(tokens)

    value = metric(model,logits, pos,answer_tokens)
    value.backward(gradients)
    model.reset_hooks()
    return value.item(), ActivationCache(cache, model), ActivationCache(grad_cache, model)


def neg_log_pron(model,logits, pos,answer_tokens):
    assert len(pos.shape) == 2, "The positions must be a 2 dim tensor (batch position)"
    assert len(answer_tokens.shape) == 2, "The answer tokens must be a 2 dim tensor (batch correct tokens)"
    assert len(logits.shape) == 3, "The logits must be a 3 dim tensor (batch position vocab)"
    
    pos = pos.unsqueeze(-1).expand(-1, -1, model.cfg.d_vocab)
    log_probs = logits.log_softmax(logits, dim=-1)
    gather_rows  = log_probs.gather(1, pos)
    log_probs = gather_rows.squeeze(0).gather(-1,answer_tokens)
    return -log_probs # Shape (batch, correct tokens)




from transformer_lens.hook_points import HookPoint
class ActivationsColector:

    def __init__(self,
                 model:HookedSAETransformer,
                 dataset,
                 modules: List[str],
                 type_activations:  Literal["Activations", "Features", "Activations DE","Features DE"],
                 activation_dir: str,
                 location_dictionary_path: str,
                 cat_activations: bool = False,
                 quantize: bool = True,
                 average: bool = True,
                 load: bool = True,
                 version: Optional[int] = None,
                 filename: Optional[str] = None,
                 device: str = "cpu",
                 batch_size: int = 256,
                 

                 ):



        self.dataset = DataLoader(dataset, batch_size = batch_size)
        self.model = model  
        self.modules = modules
        self.modules = modules 
        self.activation_dir = activation_dir
        self.device = device
        self.cat_activations = cat_activations
        with open(location_dictionary_path+"/final_dict.json") as f:
            self.location_dictionary = json.load(f)
        self.quantize = quantize
        self.average = average
        self.type_activations = type_activations
        


        self.type_checking()


        if self.cat_activations:
            assert self.average == True, "If cat_activations they must be averaged"
            self.get_cat_modules()
        if load:
            assert filename is not None, "The filename must be provided"
            assert version is not None, "The version must be provided"
            self.filename = filename
            self.version = version
            self.versioned_dir = os.path.join(self.activation_dir,f"version_{self.version}")
            self.load_activations()
        else:
            self.activations = self.collect_activations()
            self.save_activations()



    def type_checking(self):
        self.act_shapes = []
        if self.type_activations == "Activations":
            with torch.no_grad():
                _,cache = self.model.run_with_cache(" ")
            for hook in self.modules:
                assert isinstance(cache[hook], torch.Tensor), "The module must return a torch.Tensor"
                if self.average:
                    self.act_shapes.append(cache[hook].mean(dim = 1).shape)
                
                else:
                    self.act_shapes.append(cache[hook].shape)

                if self.cat_activations:
                    size = sum([x[1] for x in self.act_shapes])
                    self.act_shapes = [torch.empty( 1,size).shape]
                        

        elif self.type_activations == "Features":
            for module in self.modules:
                assert "attn" in module, "For now only Attention SAEs are supported"

            layers = []
            for module in self.modules:
                layers.append(int(module.split(".")[1].replace("L","")))


            self.saes_dict = get_attention_sae_dict(layers,device = self.device)
            

            filter_sae_acts = lambda name: ("hook_sae_acts_post" in name)
            with torch.no_grad():
                _,cache = self.model.run_with_cache_with_saes(" ",saes = [sae  for _,sae in self.saes_dict.items() ], names_filter = filter_sae_acts)
            for hook in self.modules:
                assert isinstance(cache[hook+".hook_sae_acts_post"], torch.Tensor), "The module must return a torch.Tensor"
                if self.average:
                    self.act_shapes.append(cache[hook+".hook_sae_acts_post"].mean(1).shape)
                else:
                    self.act_shapes.append(cache[hook+".hook_sae_acts_post"].shape)
        elif self.type_activations == "Features DE":
            pass


        





    
            
    def get_cat_modules(self):
        assert self.cat_activations, "Only works if cat_activations is True"
        if self.type_activations == "Activations":
            self.simp_modules = "-".join(["".join([e.replace("blocks","").replace("hook","") for e in x.split(".")]) for x in self.modules])
        elif self.type_activations == "Features":
            self.simp_modules = "-".join(["".join([e.replace("blocks","").replace("hook","") for e in x.split(".")]+["sae"]) for x in self.modules])
            





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
            for doc, seq in zip(batch_dict.keys(),input_ids):# select the document
                doc_list = batch_dict[str(doc)]
                if self.type_activations == "Activations":
                    with torch.no_grad():
                        _,cache = self.model.run_with_cache(seq)# Get all the activations (this needs to be changed)
                elif self.type_activations == "Features":
                    filter_sae_acts = lambda name: ("hook_sae_acts_post" in name)
                    with torch.no_grad():
                        _,cache = self.model.run_with_cache_with_saes(seq,saes = [sae  for _,sae in self.saes_dict.items() ], names_filter = filter_sae_acts)
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
                    cat_act = defaultdict(list) 
                    for hook,act_dict in act_seq.items():
                        for pos,act in act_dict.items():
                            cat_act[pos] += [act]
                    cat_act = {pos:np.concatenate(act_list, axis = 1) for pos,act_list in cat_act.items()}
                    act_seq = {}
                    
                    act_seq[self.simp_modules] = cat_act
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

        self.filename = f"activations_{self.type_activations}_{self.cat_activations}.h5"
        self.versioned_dir = self._create_versioned_dir(self.activation_dir)
        full_dir = os.path.join(self.versioned_dir,self.filename)

        with h5py.File(full_dir, "w") as f:
            for key, act in flatten_activations(self.activations):
                group = f.require_group("/".join(key.split("/")[:-1]))
                dataset_name = key.split("/")[-1]
                group.create_dataset(dataset_name, data=act, compression=compression, chunks=chunks_size)



    def _create_versioned_dir(self,base_dir) -> str:
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        version = 0
        while True:
            versioned_dir = os.path.join(base_dir, f"version_{version}")
            if not os.path.exists(versioned_dir):
                os.makedirs(versioned_dir)
                return versioned_dir
            version += 1


    def quantize_activations(self):
        quantized_activations = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for batch, batch_dict in self.activations.items():# Get the batch
            for doc, doc_list in batch_dict.items():# Get the document
                for hook, hook_dict in doc_list.items():# Get the hook
                    for pos, act in hook_dict.items():
                        quantized_activations[batch][doc][hook][pos]= act.astype("float16")    
        self.activations = quantized_activations


    def load_activations(self):

        assert self.versioned_dir is not None, "The activations must be saved first"
        def nested_dict():
            """Helper function to create a nested defaultdict."""
            return defaultdict(nested_dict)

        # Initialize an empty nested dictionary
        activations = nested_dict()
        full_dir = os.path.join(self.versioned_dir,self.filename)

        with h5py.File(full_dir, "r") as f:
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
