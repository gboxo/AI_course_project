from sae_lens import HookedSAETransformer
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule, HookPoint
import os
from sae_training.sparse_autoencoder import SparseAutoencoder
from dataclasses import dataclass
from functools import partial
from typing import Any,Literal, NamedTuple, Callable
from transformer_lens.hook_points import HookPoint




if __name__ == "__main__":

    from transformer_lens import HookedTransformer, utils
    model = HookedTransformer.from_pretrained('gpt2')


    transcoder_template = "/mnt/myssd/gpt-2-small-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"
    transcoders = []
    sparsities = []
    for i in range(12):
        transcoders.append(SparseAutoencoder.load_from_pretrained(f"{transcoder_template.format(i)}.pt").eval())
        sparsities.append(torch.load(f"{transcoder_template.format(i)}_log_feature_sparsity.pt"))

    tcs_dict = {tc.cfg.out_hook_point: tc for tc in transcoders}


    # Run with transcoders


    cache_in = {}
    def caching_hook_in(acts, hook):
        cache_in[hook.name] = acts


    def reconstruct_hook(acts_out,hook:HookPoint, hook_in):
       tc = tcs_dict[hook.name]
       tc_in = cache_in[hook_in]
       tc_out, feature_acts, _,_,_,_ = tc(tc_in)

       tc_error = (acts_out - tc_out).detach().clone()
        
       return tc_out
       #return tc_out+tc_error

    fwd_hooks = []

    for hook_point,tc in tcs_dict.items():
        hook_point_in = tc.cfg.hook_point
        print(hook_point)
        print(hook_point_in)
        fwd_hooks.append( (hook_point_in, caching_hook_in))
        fwd_hooks.append((hook_point, partial(reconstruct_hook, hook_in = hook_point_in)))

    with model.hooks(fwd_hooks = fwd_hooks):
        input = "Hello, my name is"
        model_output = model(input)












