
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



def filter_pred(model,token_dataset, batch_size, strict = False, threshold = 0.1):

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
