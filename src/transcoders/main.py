from torch.autograd import backward
from tc_modules import *
import tqdm
from typing import NamedTuple, Literal
from dataclasses import dataclass
from functools import partial
from transformer_lens.hook_points import HookPoint


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
            name_filter += [f"{tc.cfg.out_hook_point}.sae.hook_hidden_post" for tc in tcs] + [f"{tc.cfg.out_hook_point}.sae.hook_feature_acts.pre" for tc in tcs] + [f"{tc.cfg.out_hook_point}.sae.hook_hidden_post.post" for tc in tcs]
            if True:
                name_filter +=  [f"blocks.{i}.attn.hook_attn_scores.pre" for i in range(model.cfg.n_layers)] + [f"blocks.{i}.attn.hook_attn_scores.post" for i in range(6)]
                name_filter +=  [f"blocks.{i}.attn.hook_pattern" for i in range(6)]             
            output, cache = run_with_ref_cache(model,toks = toks, names_filter=name_filter)



            cache["blocks.5.hook_attn_out.sae.hook_sae_acts_post.pre"][0][pos,feat].backward(retain_graph = True)

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

    return attr_cache


import gc
def get_attribution_fraction(model, toks, pos, thresholds,feat):
    # Get the attribution with threhsold 0
    attrb_threshold_dict = {}
    for threshold in [0]+thresholds:
        attrb_dict = return_attrb_dict(model, toks, pos, threshold,feat)
        total_attrb = 0
        for key,val in attrb_dict.items():
            if "scores" in key:
                continue
            total_attrb += val.sum()
        attrb_threshold_dict[threshold] = total_attrb
        del attrb_dict
        gc.collect()
    return attrb_threshold_dict









# Attribution fraction
if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("gpt2")

    sae_dict = get_attention_sae_out_dict(layers = [0,1,2,3,4,5])
    saes = [value for value in sae_dict.values()]
    with open("full_dataset.json", "r") as f:
        dataset = json.load(f)

    transcoder_template  = "/media/workspace/gpt-2-small-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"
    threshold = 0

    tcs_dict = {}
    for i in range(5):
        tc = SparseAutoencoder.load_from_pretrained(f"{transcoder_template.format(i)}.pt").eval()
        tcs_dict[tc.cfg.hook_point] = tc
    tcs = list(tcs_dict.values())
    with open("5-att-kk-148.json", "r") as f:
        feat_dict = json.load(f)

    strings = ["".join(elem["tokens"]) for elem in feat_dict["activations"]]
    toks = strings[0]

    #attrb_dict = return_attrb_dict(model, toks, 9, threshold,8506)
    thresholds = [0.2,0.1,0.05,0.025,0.01,0.001]
    attrb_threshold_dict = get_attribution_fraction(model, toks, 9, thresholds,8506)
    print(attrb_threshold_dict)



