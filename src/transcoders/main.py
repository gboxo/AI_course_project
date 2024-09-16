from tc_modules import *
from typing import NamedTuple, Literal
from dataclasses import dataclass
from functools import partial
from transformer_lens.hook_points import HookPoint


threshold = 0.1
def generate_attribution_score_filter_hook():
    v = None
    def fwd_hook(tensor: torch.Tensor, hook: HookPoint):
        nonlocal v
        v = tensor
        return tensor
    def attribution_score_filter_hook(grad: torch.Tensor, hook: HookPoint):
        print(hook.name)
        assert v is not None, "fwd_hook must be called before attribution_score_filter_hook."
        return (torch.where(v * grad > threshold, grad, torch.zeros_like(grad)),)
    return fwd_hook, attribution_score_filter_hook


if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("gpt2")
    sae_dict = get_attention_sae_dict(layers = [0,1,2,3,4,5])
    saes = [value for value in sae_dict.values()]


    with open("5-att-kk-148.json", "r") as f:
        feat_dict = json.load(f)

    strings = ["".join(elem["tokens"]) for elem in feat_dict["activations"]]
    toks = strings[0]
    first_pos = 10


    transcoder_template  = "/media/workspace/gpt-2-small-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"

    tcs_dict = {}
    for i in range(5):
        tc = SparseAutoencoder.load_from_pretrained(f"{transcoder_template.format(i)}.pt").eval()
        tcs_dict[tc.cfg.hook_point] = tc
    tcs = list(tcs_dict.values())
    fwd_hooks = []
    bwd_hooks = []

    with join_contexts(model,tcs,saes):
        target = Node("blocks.5.attn.hook_z.sae.hook_sae_acts_post",reduction="0.12.148")
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









