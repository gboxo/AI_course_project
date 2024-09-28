"""
For each Feature

1) Get the computational trace (80% if the attr by pos)
2) Get the explanations
3) Compute per comp distance between the traces of the examples of the same feature
4) Get the top components by layer and type


"""



"""
For all features

1) Get the pairwise cos sim between component traces
2) Get the feaure pairwise cos sim for encoder and decoder
3) Compute a grpah of the accuulated distance
4) Create a gif of the evolution of the graph as we add components

"""




"""
comparison_dict = {}
for i in range(len(feats)):
    for j in range(i+1,len(feats)):
        feat1 = feats[i]
        feat2 = feats[j]
        ex_comp_dict = defaultdict(dict) 
        for ex_idx,ex_trace in comp_traces[feat1].items():
            for ex_idx2,ex_trace2 in comp_traces[feat2].items():
                for comp in ex_trace.keys():
                    dist = torch.nn.functional.cosine_similarity(ex_trace[comp],ex_trace2[comp],dim = 0)
                    ex_comp_dict[comp][(ex_idx,ex_idx2)] = dist
        comparison_dict[(feat1,feat2)] = ex_comp_dict

"""


import json
from sae_lens import SAE, SAEConfig, HookedSAETransformer
import torch
import numpy as np
import torch.nn as nn
from jaxtyping import Int, Float 
from datasets import load_dataset
from torch.utils.data import DataLoader, dataset
from tqdm import tqdm
from collections import defaultdict
from sae_utils import get_attention_sae_dict
from transformer_lens.ActivationCache import ActivationCache
from sae_utils import get_attention_sae_out_dict
import os



def get_explanation(path):
    with open(path, "r") as f:
        data = json.load(f)

    if len(data["explanations"]) == 0:
        return "No explanation"
    explanations = data["explanations"][0]["description"]
    
    if explanations is None:
        return "No explanation"
    return explanations


def get_computational_trace(path):
    # Modigy to get top p
    with open(path, "rb") as f:
        data = torch.load(f)
    all_traces = {} 
    for key,val in data.items():
        all_traces[key] = val.sum(dim = 1)
    return all_traces

    

def return_wenc_w_dec(sae,feature_id):
    W_enc = sae.W_enc[feature_id]
    W_dec = sae.W_dec[feature_id]
    








if __name__ == "__main__":
    model = HookedSAETransformer.from_pretrained("gpt2", device = "cpu")
    sae_dict = get_attention_sae_out_dict(layers = [5])


    all_file_names_dataset = os.listdir("../../dataset/")
    with open("full_dataset.json","r") as f:
        full_dataset = json.load(f)
    all_file_names_act = os.listdir("../../dataset_acts/")   
    feats_with_acts = [int(file_name.split("_")[0]) for file_name in all_file_names_act]
    feats_with_data = [int(file_name.split(".")[0]) for file_name in all_file_names_dataset]
    explanations = [get_explanation("../../dataset/"+file_name) for file_name in all_file_names_dataset]
    # Convert to set and    
    feats = list(set(feats_with_acts).intersection(set(feats_with_data)))




    # Get the feaure pairwise cos sim for encoder and decoder
    enc_dict = {}
    dec_dict = {}
    for feat in feats:
        sae = list(sae_dict.values())[0]
        enc_dict[feat] = sae.W_enc.detach()[:,feat].unsqueeze(0)
        dec_dict[feat] = sae.W_dec.detach()[feat].unsqueeze(1)
    enc_sim_mat = torch.zeros((len(feats),len(feats)))
    for i,(feat1,enc1) in enumerate(enc_dict.items()):
        for j,(feat2,enc2) in enumerate(enc_dict.items()):
            enc_sim = torch.nn.functional.cosine_similarity(enc1,enc2,dim = 1)
            print(enc_sim)
            enc_sim_mat[i][j] = enc_sim

    dec_sim_mat = torch.zeros((len(feats),len(feats)))
    for i,(feat1,dec1) in enumerate(dec_dict.items()):
        for j,(feat2,dec2) in enumerate(dec_dict.items()):
            dec_sim = torch.nn.functional.cosine_similarity(dec1,dec2,dim = 1)
            print(dec_sim)
            dec_sim_mat[i][j] = dec_sim
    feat_sims = {"enc":enc_sim_mat,"dec":dec_sim_mat,"feats":feats}
    torch.save(feat_sims,"app_data/feat_sims.pt")

if False:


    computational_traces = defaultdict(dict) 
    # Get the computational trace
    for feat in feats:
        feat_acts_paths = ["../../dataset_acts/"+file_name for file_name in all_file_names_act if int(file_name.split("_")[0]) == feat]
        example_trace = {}
        for path in feat_acts_paths:
            example_id = int(path.split("_")[2])
            trace = get_computational_trace(path)
            total_attr = sum([val.sum() for val in trace.values()])
            print(total_attr)
            if total_attr == 0:
                continue
            computational_traces[feat][example_id] = trace
    torch.save(computational_traces,"app_data/computational_traces.pt")


    # Get the per comp distance
    computational_traces = torch.load("app_data/computational_traces.pt")
    components = list(list(list(computational_traces.values())[0].values())[0].keys())
    dist_comp = defaultdict(dict)
    for feat,traces in computational_traces.items():
        for comp in components:
            mat = torch.zeros((len(traces),len(traces)))
            for i,(ex_idx, ex_trace) in enumerate(traces.items()):
                for j,(ex_idx2, ex_trace2) in enumerate(traces.items()):
                    dist = torch.nn.functional.cosine_similarity(ex_trace[comp],ex_trace2[comp],dim = 1)
                    mat[i][j] = dist
            dist_comp[feat][comp] = mat
    torch.save(dist_comp,"app_data/dist_comp.pt")



    # Get the top components by layer and type  

    computational_traces = torch.load("app_data/computational_traces.pt")
    components = list(list(list(computational_traces.values())[0].values())[0].keys())
    comp_aggregations = defaultdict(dict)
    for feat,comp_dict in computational_traces.items():
        for comp in components:
            comp_aggregations[feat][comp] = torch.stack([val[comp] for val in comp_dict.values()]).mean(dim = 0)

    top_components = defaultdict(dict)
    for feat,comp_dict in comp_aggregations.items():
        for comp,comp_trace in comp_dict.items():
            # Modify to only get components with more than 0 total attribution
            top_components[feat][comp] = comp_trace.topk(10).indices
    torch.save(top_components,"app_data/top_components.pt")



    # Get the pairwise cos sim between component traces
    computational_traces = torch.load("app_data/computational_traces.pt")
    components = list(list(list(computational_traces.values())[0].values())[0].keys())
    dist_comp = {}
    for comp in components:
        mat = torch.zeros((len(computational_traces),len(computational_traces)))
        for i,(feat1,traces1) in enumerate(computational_traces.items()):
            agg_trace1 = torch.stack([val[comp] for val in traces1.values()]).mean(dim = 0)
            for j,(feat2,traces2) in enumerate(computational_traces.items()):
                agg_trace2 = torch.stack([val[comp] for val in traces2.values()]).mean(dim = 0)
                dist = torch.nn.functional.cosine_similarity(agg_trace1,agg_trace2,dim = 1)
                mat[i][j] = dist
        dist_comp[comp] = mat
    torch.save(dist_comp,"app_data/feat_pairwise_dist_comp.pt")

    # Get the feaure pairwise cos sim for encoder and decoder
    enc_dict = {}
    dec_dict = {}
    for feat in feats:
        sae = list(sae_dict.values())[0]
        enc_dict[feat] = sae.W_enc.detach()[:,feat].unsqueeze(0)
        dec_dict[feat] = sae.W_dec.detach()[feat].unsqueeze(1)
    enc_sim_mat = torch.zeros((len(feats),len(feats)))
    for i,(feat1,enc1) in enumerate(enc_dict.items()):
        for j,(feat2,enc2) in enumerate(enc_dict.items()):
            enc_sim = torch.nn.functional.cosine_similarity(enc1,enc2,dim = 1)
            enc_sim_mat[i][j] = enc_sim

    dec_sim_mat = torch.zeros((len(feats),len(feats)))
    for i,(feat1,dec1) in enumerate(dec_dict.items()):
        for j,(feat2,dec2) in enumerate(dec_dict.items()):
            dec_sim = torch.nn.functional.cosine_similarity(dec1,dec2,dim = 0)
            dec_sim_mat[i][j] = dec_sim
    feat_sims = {"enc":enc_sim_mat,"dec":dec_sim_mat,"feats":feats}
    torch.save(feat_sims,"app_data/feat_sims.pt")




