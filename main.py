# AI Safety Course Project



# %%
from typing import List,Dict, Tuple,Any, Optional
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

# %%

model = HookedSAETransformer.from_pretrained("gpt2")
data = load_dataset("/home/gerard/MI/pile-10k/", split = "train")

tokens = tokenize_and_concatenate(data,tokenizer = model.tokenizer, max_length = 128)

# %%


def filter_pred(token_dataset, batch_size):

    dataset = DataLoader(token_dataset, batch_size = batch_size)
    check_point_dict = defaultdict(dict) 


    for i,d in tqdm(enumerate(dataset)):
        input_ids = d['tokens']
        with torch.no_grad():
            logits = model(input_ids)

        logits = logits[:,:-1].contiguous()
        target_ids = input_ids[:,1:].contiguous()
        argmax_pred = torch.argmax(logits, dim = -1)
        pred = (argmax_pred == target_ids).float()
        pred_pos = torch.where(pred)
        dict_doc_pos = defaultdict(list)
        for doc,ind in zip(pred_pos[0], pred_pos[1]):
            dict_doc_pos[doc.item()].append(ind.item())

        check_point_dict[f"Batch {i}"] = dict_doc_pos 
        with open(f"checkpoints/pred_checkpoint_{i}.pt", "wb") as f:
            torch.save(check_point_dict, f)




#filter_pred(tokens, 4)



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

        

get_correct_sequences("checkpoints", 3)
# %%

# Function to display the sequenes of correct predictions




def create_visualization(token_dataset, final_dict,batch_size = 4):
    with open("final_dict.json", "r") as f:
        final_dict = json.load(f)

    dataset = DataLoader(token_dataset, batch_size = batch_size)


    html_content = "<html><head><style>\
        .token {font-size: 14px; margin: 2px; padding: 2px;}\
        .correct {background-color: #d4edda; border: 1px solid #c3e6cb;}\
        .incorrect {background-color: #FFFFFF; border: 1px solid #f5c6cb;}\
        .prompt-group {margin-bottom: 20px;}\
        .prompt-title {font-weight: bold; margin-bottom: 10px;}\
        </style></head><body>\
        <h1>Token Prediction Visualization</h1>"
    

    for i,d in tqdm(enumerate(dataset)):# Select the batch
        input_ids = d['tokens']
        if f"Batch {i}" not in final_dict:
            break
        batch_dict = final_dict[f"Batch {i}"]
        html_content += f'<div class="prompt-title">Batch {i}</div>'
        for doc, seq in enumerate(input_ids):
            doc_list = batch_dict[str(doc)]
            if len(doc_list) == 0:
                continue
            indices = torch.zeros(len(seq))
            str_tokens = [model.to_string(ind) for ind in seq]
            for tup in doc_list:
                for i in range(tup[0],tup[1]+1):
                    indices[i] = 1

            html_content += f'<div class="prompt-title">Document {doc}</div>'
            for token, correct in zip(str_tokens[1:], indices[1:]):
                token_text = html.escape(token)
                token_class = 'correct' if correct==1 else 'incorrect'
                html_content += f'<span class="token {token_class}">{token_text}</span> '
            html_content += "Original Text:"
            html_content += html.escape(model.to_string(seq[1:])).replace(' ', '&nbsp;')
            html_content += '</div><br>'
        html_content += '</div>'
        html_content += '</div>'
    
    html_content += '</body></html>'

    with open("vis0.html","w") as f:
        f.write(html_content)



# create_visualization(tokens, "final_dict.json", 4)
 
    

# %%


# Function to get the model activations for each sequence


from transformer_lens.hook_points import HookPoint
class ActivationsColector:

    def __init__(self,
                 model:HookedSAETransformer,
                 dataset,
                 ctx_len: int,
                 modules: List[str],
                 location_dictionary: dict,
                 cat_activations: bool = False,
                 quantize: bool = True

                 ):



        self.dataset = DataLoader(dataset, batch_size = 4)
        self.model = model  
        self.ctx_len = ctx_len
        self.modules = modules
        self.modules = modules 
        self.cat_activations = cat_activations
        self.location_dictionary = location_dictionary
        self.quantize = quantize


        self.activations = self.collect_activations()
        #self.quantize_activations()



    def collect_activations(self):
        activations = {}

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
                        act = cache[hook]
                        act_hook_pos = {}
                        for tup in doc_list:
                            x = act[:,tup[0]:tup[1]+1]
                            act_hook_pos[str(tup)]= x.to("cpu").numpy()# Add the activations for the correct predictions positions
                        act_seq[hook] = act_hook_pos # Add the activations for the module
                batch_activations_dict[doc] = act_seq # Add the activations for the document
            activations[f"Batch {i}"] = batch_activations_dict

        return activations

    def save_activations(self, compression = "gzip", chunks_size = None):

        if self.quantize:
            self.activations = self.quantize_activations()
        with h5py.File("activations.h5","w") as f:

            for batch, batch_dict in self.activations.items():# Get the batch
                for doc, doc_list in batch_dict.items():# Get the document
                    for hook, hook_dict in doc_list.items():# Get the hook
                        for pos, act in hook_dict.items():
                            key = f"{batch}/{doc}/{hook}/{pos}"
                            f.create_dataset(key, data = act, compression = compression, chunks = chunks_size)


    def quantize_activations(self):
        quantized_activations = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for batch, batch_dict in self.activations.items():# Get the batch
            for doc, doc_list in batch_dict.items():# Get the document
                for hook, hook_dict in doc_list.items():# Get the hook
                    for pos, act in hook_dict.items():
                        quantized_activations[batch][doc][hook][pos]= act.astype("float16")    
        self.activations = quantized_activations



    def load_activations(file_name, batch_index, doc_index, hook_name, pos):
        activations = {}
        with h5py.File(file_name,"r") as f:
            group = f"{batch_index}/{doc_index}/{hook_name}/{pos}"
            for tensor_





# %%

with open("final_dict.json", "r") as f:
    location_dict = json.load(f)
acts = ActivationsColector(model, tokens, 3, ["blocks.10.hook_attn_out","blocks.10.hook_resid_pre"],location_dict )












    


