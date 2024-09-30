from torch.autograd import backward
import numpy as np
from tc_modules import *
import tqdm
from typing import NamedTuple, Literal
from dataclasses import dataclass
from functools import partial
from transformer_lens.hook_points import HookPoint
import seaborn as sns
import matplotlib.pyplot as plt

def generate_attribution_score_filter_hook():
    v = None
    def fwd_hook(tensor: torch.Tensor, hook: HookPoint):
        nonlocal v
        v = tensor
        return tensor
    def attribution_score_filter_hook(grad: torch.Tensor, hook: HookPoint):
        assert v is not None, "fwd_hook must be called before attribution_score_filter_hook."
        return (torch.where(v * grad > threshold, grad, torch.zeros_like(grad)),)
        #return (grad,)
    return fwd_hook, attribution_score_filter_hook

def return_attrb_dict(model, toks,pos,threshold,feat):


    candidates = [f"blocks.{i}.attn.hook_attn_scores" for i in range(6)]
    candidates +=   [f"{sae.cfg.out_hook_point}.sae.hook_hidden_post" for sae in tcs] + [f"{sae.cfg.hook_name}.sae.hook_sae_acts_post" for sae in saes]



    #with join_contexts(model,tcs,saes,candidates):
    all_saes = tcs + saes
    with apply_sae(model, all_saes):
        with detach_at(model, candidates):

            name_filter = [f"{sae.cfg.hook_name}.sae.hook_sae_acts_post" for sae in saes] + [f"{sae.cfg.hook_name}.sae.hook_sae_acts_post.pre" for sae in saes] + [f"{sae.cfg.hook_name}.sae.hook_sae_acts_post.post" for sae in saes]
            name_filter += [f"{tc.cfg.out_hook_point}.sae.hook_hidden_post" for tc in tcs] + [f"{tc.cfg.out_hook_point}.sae.hook_hidden_post.pre" for tc in tcs] + [f"{tc.cfg.out_hook_point}.sae.hook_hidden_post.post" for tc in tcs]
            if True:
                name_filter +=  [f"blocks.{i}.attn.hook_attn_scores.pre" for i in range(model.cfg.n_layers)] + [f"blocks.{i}.attn.hook_attn_scores.post" for i in range(6)]
                name_filter +=  [f"blocks.{i}.attn.hook_pattern" for i in range(6)]             
            output, cache = run_with_ref_cache(model,toks = toks, names_filter=name_filter)
            target_act = cache["blocks.5.hook_mlp_out.sae.hook_hidden_post.post"][0][pos,feat].item()



            cache["blocks.5.hook_mlp_out.sae.hook_hidden_post.pre"][0][pos,feat].backward(retain_graph = True)

            attr_cache = {}

            for candidate in candidates:
                if "attn_scores" in candidate:
                    continue
                    
                feature_acts = cache[candidate+".post"]
                if feature_acts.grad is None:
                    continue
                grad = feature_acts.grad[:,:pos,:]
                attrb = feature_acts[:,:pos,:] * grad
                attrb[attrb<threshold] = 0
                attr_cache[candidate] = attrb 
                feature_acts.grad.zero_()




            for candidate in candidates:
                if not "attn_scores" in candidate:
                    continue
                    
                feature_acts = cache[candidate+".post"]
                if feature_acts.grad is None:
                    continue
                grad = feature_acts.grad[:,:,:pos,:pos]
                attrb = feature_acts[:,:,:pos,:pos] * grad
                attrb[attrb.isnan()] = 0
                attrb[attrb<threshold] = 0
                attr_cache[candidate] = attrb
                feature_acts.grad.zero_()




        model.reset_hooks()

    return attr_cache, target_act


import gc
def get_attribution_fraction(model, toks, pos, thresholds,feat):
    # Get the attribution with threhsold 0
    attrb_threshold_dict = {}
    for threshold in thresholds+[0]:
        attrb_dict,_ = return_attrb_dict(model, toks, pos, threshold,feat)
        total_attrb = 0
        for key,val in attrb_dict.items():
            if "scores" in key:
                continue
            total_attrb += val.sum()
        attrb_threshold_dict[threshold] = total_attrb
        del attrb_dict
        gc.collect()
    return attrb_threshold_dict

def get_max_act(model,tokens):
    with apply_sae(model, saes):
        with detach_at(model, [f"blocks.{i}.attn.hook_attn_scores" for i in range(6)]):
            output, cache = run_with_ref_cache(model,toks = tokens)
            max_feat = cache["blocks.5.hook_attn_out.sae.hook_sae_acts_post"][0][-1].argmax().item()
    return max_feat




def get_attribution_fraction_dataet(model,dataset):
    all_tuples = []
    all_fractions = []
    for tokens in tqdm.tqdm(dataset["tokens"]):
        #max_feat = get_max_act(model,tokens)
        attrb_dict,target_act = return_attrb_dict(model, tokens, 31, 0.05,max_feat)
        total_attrb = 0
        for key,val in attrb_dict.items():
            if "scores" in key:
                continue
            total_attrb += val.sum()    
        all_fractions.append(total_attrb/(target_act+1e-6))
        all_tuples.append((target_act,total_attrb))
    return all_fractions, all_tuples


def get_trace(attrb_dict):
    trace = {}
    for key,val in attrb_dict.items():
        if "scores" in key:
            continue
        trace[key] = val.mean(dim = 0).mean(dim = 0)
    return trace



from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset
# Attribution fraction
if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("gpt2")

    sae_dict = get_attention_sae_out_dict(layers = [0,1,2,3,4,5])
    saes = [value for value in sae_dict.values()]

    transcoder_template  = "/media/workspace/gpt-2-small-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"
    threshold = 0

    tcs_dict = {}
    for i in range(6):
        tc = SparseAutoencoder.load_from_pretrained(f"{transcoder_template.format(i)}.pt").eval()
        tcs_dict[tc.cfg.hook_point] = tc
    tcs = list(tcs_dict.values())


    #dataset = load_dataset("NeelNanda/pile-10k", split = "train")
    #tokens = tokenize_and_concatenate(dataset, model.tokenizer, max_length=32)
    #tokens = tokens[:100]
    #all_fractions,all_tuples  = get_attribution_fraction_dataet(model,tokens)
    #all_fractions_dict = {key: val for key,val in zip(tokens["tokens"],all_fractions)}
    #torch.save(all_fractions_dict,"attribution_fractions.pt")
    #torch.save(all_tuples,"all_tuples.pt")
# =======
    #all_tuples = torch.load("all_tuples.pt")
    #all_fractions = torch.load("attribution_fractions.pt")


    #sns.scatterplot(x = [x[0] for x in all_tuples],y = [x[1].detach().item() for x in all_tuples])
    #x_vals = np.array([x[0] for x in all_tuples])
    #plt.plot(x_vals, x_vals, color='red', linestyle='--')  # Red dashed line for y=x
    #plt.plot(x_vals, 1.5*x_vals, color='red', linestyle='--')  # Red dashed line for y=x
    #plt.plot(x_vals, 0.5*x_vals, color='red', linestyle='--')  # Red dashed line for y=x
    #plt.show()

    #attrb_dict,_ = return_attrb_dict(model, toks, 9, threshold,8506)
    #thresholds = [0.2,0.1,0.05,0.025,0.01,0.001]
    #attrb_threshold_dict = get_attribution_fraction(model, toks, 9, thresholds,8506)
    #print(attrb_threshold_dict)

    feat_sims = torch.load("feat_sims.pt")
    feats = feat_sims["feats"]
    del feat_sims
    with open("full_dataset.json","r") as f:
        full_dataset = json.load(f)
    full_dataset = {key:val for key,val in full_dataset.items() if int(key) in feats}
    dataset = {key:[v for v in val.values()][0] for key,val in full_dataset.items()} 
    all_fractions_dicts = []
    for key,val in tqdm.tqdm(dataset.items()):
        pos = val[0]
        toks = val[1]
        toks = torch.tensor(toks).unsqueeze(0)
        frction_dict = get_attribution_fraction(model, toks, pos, [0.05,0.025,0.0125,0.00625,0.001],int(key))
        all_fractions_dicts.append(frction_dict)

        
    
    for i,val_dict in enumerate(all_fractions_dicts):
        for threshold, val in val_dict.items():
            all_fractions_dicts[i][threshold] = val.detach().item()

    for i,val_dict in enumerate(all_fractions_dicts):
        total_attrb = val_dict[0]
        for threshold, val in val_dict.items():
            all_fractions_dicts[i][threshold] = val/total_attrb

    all_dicts = defaultdict(dict) 
    for i,val_dict in enumerate(all_fractions_dicts):
        feat = list(dataset.keys())[i]
        for threshold, val in val_dict.items():
            all_dicts[feat][threshold] = val
    
    torch.save(all_dicts,"attribution_fractions.pt")
    # plot a lineplot for each feature

    all_dicts = torch.load("attribution_fractions.pt")
    plt.figure(figsize=(10, 6))

    for key, val in all_dicts.items():
        plt.plot(list(val.keys()), list(val.values()), label=key)

    plt.xlabel("Threshold")  # Replace with an appropriate label
    plt.ylabel("Fraction")  # Replace with an appropriate label
    plt.title("Attribution Fractions Plot")  # Replace with a suitable title

    plt.grid(True)

    plt.gca().invert_xaxis()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, ncol=2)  # ncol=2 splits legend into 2 columns
    plt.tight_layout()

    plt.savefig("attribution_fractions_plot.png")

    #i = 0
    #for feat,feat_dict in tqdm.tqdm(full_dataset.items()):
    #    for eg_id,elem_list in feat_dict.items():
    #        pos = elem_list[0]
    #        toks = elem_list[1]
    #        toks = torch.tensor(toks).unsqueeze(0)
    #        attrb_dict,target_act = return_attrb_dict(model, toks, pos, 0,int(feat))
    #        for key,val in attrb_dict.items():
    #            if "scores" in key:
    #                continue
    #        mean_trace = get_trace(attrb_dict)
    #        comp_trace = {"target_act":target_act,"Mean trace":mean_trace}

            # Save the trace and the target act
     #       torch.save(comp_trace,f"app_data/mean_trace_{feat}_{eg_id}.pt")



