from tc_modules import *
import tqdm
from typing import NamedTuple, Literal
from dataclasses import dataclass
from functools import partial
from transformer_lens.hook_points import HookPoint


threshold = 0.01
def generate_attribution_score_filter_hook():
    v = None
    def fwd_hook(tensor: torch.Tensor, hook: HookPoint):
        nonlocal v
        v = tensor
        return tensor
    def attribution_score_filter_hook(grad: torch.Tensor, hook: HookPoint):
        assert v is not None, "fwd_hook must be called before attribution_score_filter_hook."
        return (torch.where(v * grad > threshold, grad, torch.zeros_like(grad)),)
    return fwd_hook, attribution_score_filter_hook

def return_attrb_dict(model, toks,pos,threshold,feat):
    pos -= 1

    fwd_hooks = []
    bwd_hooks = []

    with join_contexts(model,tcs,saes):
        target = Node("blocks.5.attn.hook_z.sae.hook_sae_acts_post",reduction=f"0.{pos}.{feat}")
        candidates = [Node(f"{sae.cfg.hook_name}.sae.hook_sae_acts_post") for sae in saes[:-1]] + [Node(f"{sae.cfg.out_hook_point}.sae.hook_hidden_post") for sae in tcs] + [Node(f"blocks.{i}.attn.hook_attn_scores") for i in range(6)]
        nodes = candidates + [target]
        
        attribution_score_filter_hooks = {candidate: generate_attribution_score_filter_hook() for candidate in candidates}
        fwd_hooks = [(candidate.hook_point, compose_hooks(attribution_score_filter_hooks[candidate][0], retain_grad_hook)) for candidate in candidates]
        with model.hooks(
            fwd_hooks=fwd_hooks,
            bwd_hooks=[(candidate.hook_point, attribution_score_filter_hooks[candidate][1]) for candidate in candidates]
        ):
            output, cache = run_with_ref_cache(model,toks = toks, names_filter=[node.hook_point for node in nodes])
            cache = Cache(output,cache) 

            cache[target].backward()
            attr_cache = {}
            with torch.no_grad():
                for candidate in candidates:
                    if "attn_scores" in candidate.hook_point:
                        continue
                    grad = cache[candidate].grad
                    attrb = cache[candidate] * grad
                    attrb[attrb<threshold] = 0

                    attr_cache[candidate.hook_point] = attrb 

        model.reset_hooks()
        for key,val in attr_cache.items():
            print(val.sum())
    return attr_cache




if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("gpt2")
    sae_dict = get_attention_sae_dict(layers = [0,1,2,3,4,5])
    saes = [value for value in sae_dict.values()]

    with open("full_dataset.json", "r") as f:
        dataset = json.load(f)

    transcoder_template  = "/media/workspace/gpt-2-small-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"
    threshold = 0.01

    tcs_dict = {}
    for i in range(5):
        tc = SparseAutoencoder.load_from_pretrained(f"{transcoder_template.format(i)}.pt").eval()
        tcs_dict[tc.cfg.hook_point] = tc
    tcs = list(tcs_dict.values())

    for feature_id, data in tqdm.tqdm(dataset.items()):
        for idx, (pos, toks) in data.items():
            toks = torch.tensor(toks).unsqueeze(0)
            attrb_dict = return_attrb_dict(model, toks, pos, threshold,int(feature_id))
            print("Saving attributions...")
            # Save the attributions to a .pt file 
            torch.save(attrb_dict, f"../../dataset_acts/{feature_id}_{idx}_{threshold}.pt")











