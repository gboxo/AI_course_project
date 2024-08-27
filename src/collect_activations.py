

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


import h5py



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
