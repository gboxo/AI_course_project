from contextlib import contextmanager
from sae_lens import HookedSAETransformer
import torch
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule, HookPoint
from dataclasses import dataclass
from functools import partial
from typing import Any,Literal, NamedTuple, Callable
from transformer_lens.hook_points import HookPoint
from typing import Tuple, Union
from sae_training.sparse_autoencoder import SparseAutoencoder



def compose_hooks(*hooks):
    """
    Compose multiple hooks into a single hook by executing them in order.
    """
    def composed_hook(tensor: torch.Tensor, hook: HookPoint):
        for hook_fn in hooks:
            tensor = hook_fn(tensor, hook)
        return tensor
    return composed_hook

def retain_grad_hook(tensor: torch.Tensor, hook: HookPoint):
    """
    Retain the gradient of the tensor at the given hook point.
    """
    tensor.retain_grad()
    return tensor

def detach_hook(tensor: torch.Tensor, hook: HookPoint):
    """
    Detach the tensor at the given hook point.
    """
    return tensor.detach().requires_grad_(True)
def generate_attribution_score_filter_hook():
    v = None
    def fwd_hook(tensor: torch.Tensor, hook: HookPoint):
        print(hook.name)
        nonlocal v
        v = tensor
        return tensor
    def attribution_score_filter_hook(grad: torch.Tensor, hook: HookPoint):
        print(hook.name)
        assert v is not None, "fwd_hook must be called before attribution_score_filter_hook."
        return (torch.where(v * grad*0 > threshold, grad, torch.zeros_like(grad)),)
    return fwd_hook, attribution_score_filter_hook



def get_fwd_hooks(sae: SparseAutoencoder) -> list[Tuple[Union[str, Callable], Callable]]:
    x = None
    def hook_in(tensor: torch.Tensor, hook: HookPoint):
        nonlocal x
        x = tensor
        return tensor
    def hook_out(tensor: torch.Tensor, hook: HookPoint):
        nonlocal x
        assert x is not None, "hook_in must be called before hook_out."
        reconstructed,_,_,_,_,_ = sae.forward(x)
        x = None
        return reconstructed + (tensor - reconstructed).detach()
    return [(sae.cfg.hook_point, hook_in), (sae.cfg.out_hook_point, hook_out)]


if __name__ == "__main__":

    from transformer_lens import HookedTransformer, utils
    model = HookedSAETransformer.from_pretrained('gpt2')


    transcoder_template = "/media/workspace/gpt-2-small-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"
    transcoders = []
    sparsities = []
    for i in range(4,7):
        transcoders.append(SparseAutoencoder.load_from_pretrained(f"{transcoder_template.format(i)}.pt").eval())
        sparsities.append(torch.load(f"{transcoder_template.format(i)}_log_feature_sparsity.pt"))

    tcs_dict = {tc.cfg.out_hook_point: tc for tc in transcoders}


    # Run with transcoders

    threshold = 0.1

    fwd_hooks: list[Tuple[Union[str, Callable], Callable]] = []
    retain_grad = True

    cache = {}
    def save_hook(tensor: torch.Tensor, hook: HookPoint):
        if retain_grad:
            tensor.retain_grad()
        cache[hook.name] = tensor

    fwd_hooks = []
    names_filter = lambda name: True
    for name, hp in model.hook_dict.items():
        if names_filter(name):
            fwd_hooks.append((name, save_hook))
    for sae in tcs_dict.values():
        hooks = get_fwd_hooks(sae)
        fwd_hooks.extend(hooks)

    candidates = list(tcs_dict.keys())
        
    attribution_score_filter_hooks = {candidate: generate_attribution_score_filter_hook() for candidate in candidates}
    fwd_hooks += [(candidate, compose_hooks(attribution_score_filter_hooks[candidate][0], retain_grad_hook)) for candidate in candidates]
    bwd_hooks = [(candidate, attribution_score_filter_hooks[candidate][1]) for candidate in candidates]


    with model.hooks(
        fwd_hooks=fwd_hooks,
        bwd_hooks=bwd_hooks    ):
        print(model.hooks)
        input = "Hello, my name is"
        model_output = model(input)
        metric = model_output.sum()
        metric.backward()
        print(metric.grad)





        

