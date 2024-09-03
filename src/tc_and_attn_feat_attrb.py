
from sae_lens import SAE, SAEConfig, HookedSAETransformer
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, dataset
from tqdm import tqdm
from collections import defaultdict
from transformer_lens.ActivationCache import ActivationCache
from jaxtyping import Int, Float 
from sae_utils import get_attention_sae_dict


import os
from struct import calcsize
import plotly.express as px
import pandas as pd

from sae_lens import HookedSAETransformer
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from transformer_lens.hook_points import HookedRootModule, HookPoint
import os
from sae_training.sparse_autoencoder import SparseAutoencoder
from typing import Any,Literal, NamedTuple, Callable
from transformer_lens.hook_points import HookPoint

torch.set_grad_enabled(False)

device = "cuda:0" if torch.cuda.is_available() else "cpu"


from datasets import load_dataset

from transformer_lens.utils import tokenize_and_concatenate



# Feature Attribution


from dataclasses import dataclass
from functools import partial
from transformer_lens.hook_points import HookPoint



class TcReconstructionCache(NamedTuple):
    tc_in: torch.Tensor
    feature_acts: torch.Tensor
    tc_out: torch.Tensor
    tc_error: torch.Tensor


class SaeReconstructionCache(NamedTuple):
    sae_in: torch.Tensor
    feature_acts: torch.Tensor
    sae_out: torch.Tensor
    sae_error: torch.Tensor

def track_grad(tensor: torch.Tensor) -> None:
    tensor.requires_grad_(True)
    tensor.retain_grad()





@dataclass
class ApplyAndRunOutput:
    model_output: torch.Tensor
    model_activations: dict[str,torch.Tensor]
    tc_activations: dict[str, TcReconstructionCache]
    sae_activations: dict[str, SaeReconstructionCache]

    def zero_grad(self) -> None:
        """helper to zero grad all tensors in this object"""
        self.model_output.grad = None
        for act in self.model_activations.values():
            act.grad = None
        for cache in self.tc_activations.values():
            cache.tc_in.grad = None
            cache.feature_acts.grad = None
            cache.tc_out.grad = None
            cache.tc_error.grad = None
        for cache_sae in self.sae_activations.values():
            cache_sae.sae_in.grad = None
            cache_sae.feature_acts.grad = None
            cache_sae.sae_out.grad = None
            cache_sae.sae_error.grad = None

def apply_and_run(
        model: HookedSAETransformer,
        tcs_dict: dict[str,SparseAutoencoder],
        saes_dict: dict[str,SAE],
        input:Any,
        include_error_term: bool = True,
        track_model_hooks: list[str] | None = None,
        return_type: Literal["logits","loss"] = "logits",
        track_grads: bool = False

        ) -> ApplyAndRunOutput:
    fwd_hooks = []
    bwd_hooks = []
    tc_activations: dict[str, TcReconstructionCache] = {}
    sae_activations: dict[str, SaeReconstructionCache] = {}
    model_activations:  dict[str, torch.Tensor] = {}

    cache_in = {}
    def caching_hook_in(acts, hook):
        cache_in[hook.name] = acts.detach().clone()

    def reconstruct_hook(acts_out,hook:HookPoint, hook_in):
       tc = tcs_dict[hook.name]
       tc_in = cache_in[hook_in]
       tc_out, feature_acts, _,_,_,_ = tc(tc_in)

       tc_error = (acts_out - tc_out).detach().clone()
        
       if track_grads:
            track_grad(tc_error)
            track_grad(tc_out)
            track_grad(feature_acts)
            track_grad(tc_in)

       tc_activations[hook.name] = TcReconstructionCache(
                tc_in = tc_in,
                feature_acts = feature_acts,
                tc_out = tc_out,
                tc_error = tc_error

                )
       if include_error_term:
           return tc_out + tc_error
       else:
           return tc_out



    def reconstruct_sae_hook(sae_in,hook:HookPoint, hook_point: str):
        sae = saes_dict[hook_point]
        feature_acts = sae.encode(sae_in)
        sae_out = sae.decode(feature_acts)

        sae_error = (sae_in - sae_out).detach().clone()
        if track_grads:
            track_grad(sae_error)
            track_grad(sae_out)
            track_grad(feature_acts)
            track_grad(sae_in)
        sae_activations[hook_point] = SaeReconstructionCache(
                sae_in = sae_in,
                feature_acts = feature_acts,
                sae_out = sae_out,
                sae_error = sae_error

                )
        if include_error_term:
            return sae_out + sae_error
        else:
            return sae_out





    def sae_bwd_hook(output_grads: torch.Tensor, hook:HookPoint):
        return (output_grads,)
    def tracking_hook(hook_input: torch.Tensor, hook:HookPoint, hook_point: str):
        model_activations[hook_point] = hook_input
        if track_grads:
            track_grad(hook_input)
        return hook_input

    for hook_point in saes_dict.keys():
        fwd_hooks.append(
                (hook_point,partial(reconstruct_sae_hook, hook_point= hook_point))
                )
        bwd_hooks.append((hook_point,sae_bwd_hook))
    for hook_point,tc in tcs_dict.items():
        hook_point_in = tc.cfg.hook_point
        fwd_hooks.append( (hook_point_in, caching_hook_in))
        fwd_hooks.append((hook_point, partial(reconstruct_hook, hook_in = hook_point_in)))
        bwd_hooks.append((hook_point,sae_bwd_hook))
    for hook_point in track_model_hooks or []:
        fwd_hooks.append((hook_point,partial(tracking_hook, hook_point=hook_point)))
    # run the model while applying the hooks

    with model.hooks(fwd_hooks = fwd_hooks, bwd_hooks = bwd_hooks):
        model_output = model(input, return_type = return_type)

    print(tc_activations.keys())


    return ApplyAndRunOutput(
            model_output= model_output,
            model_activations = model_activations,
            tc_activations = tc_activations,
            sae_activations = sae_activations
            ) 




EPS = 1e-8

torch.set_grad_enabled(True)
@dataclass
class AttributionGrads:
    metric: torch.Tensor
    model_output: torch.Tensor
    model_activations: dict[str, torch.Tensor]
    tc_activations: dict[str, TcReconstructionCache]
    sae_activations: dict[str, SaeReconstructionCache]
    






@dataclass
class Attribution:
    model_attributions: dict[str, torch.Tensor]
    model_activations: dict[str, torch.Tensor]
    model_grads: dict[str, torch.Tensor]
    tc_feature_attributions: dict[str, torch.Tensor]
    tc_feature_activations: dict[str,torch.Tensor]
    tc_feature_grads: dict[str, torch.Tensor]
    tc_errors_attribution_proportion: dict[str, float]
    sae_feature_attributions: dict[str, torch.Tensor]
    sae_feature_activations: dict[str,torch.Tensor]
    sae_feature_grads: dict[str, torch.Tensor]
    sae_errors_attribution_proportion: dict[str, float]
    




def calculate_attribution_grads(
    model: HookedSAETransformer,
    prompt: str,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    track_hook_points: list[str] | None = None,
    include_tcs: dict[str, SparseAutoencoder] | None = None,
    saes_dict: dict[str, SAE] | None = None,
    return_logits: bool = True,
    include_error_term: bool = True,
) -> AttributionGrads:
    """
    Wrapper around apply_saes_and_run that calculates gradients wrt to the metric_fn.
    Tracks grads for both SAE feature and model neurons, and returns them in a structured format.
    """
    output = apply_and_run(
        model,
        tcs_dict=include_tcs or {},
        saes_dict = saes_dict or {},
        input=prompt,
        return_type="logits" if return_logits else "loss",
        track_model_hooks=track_hook_points,
        include_error_term=include_error_term,
        track_grads=True,
    )
    metric = metric_fn(output.model_output)
    gradients = torch.ones_like(metric)
    output.zero_grad()
    metric.backward(gradients)
    return AttributionGrads(
        metric=metric,
        model_output=output.model_output,
        model_activations=output.model_activations,
        tc_activations=output.tc_activations,
        sae_activations = output.sae_activations
    )


def calculate_feature_attribution(
    model: HookedSAETransformer,
    input: Any,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    track_hook_points: list[str] | None = None,
    include_tcs: dict[str, SparseAutoencoder] | None = None,
    include_saes: dict[str, SAE] | None = None,
    return_logits: bool = True,
    include_error_term: bool = True,
) -> Attribution:

    outputs_with_grads = calculate_attribution_grads(
        model,
        input,
        metric_fn,
        track_hook_points,
        include_tcs=include_tcs,
        saes_dict = include_saes,
        return_logits=return_logits,
        include_error_term=include_error_term,
    )
    model_attributions = {}
    model_activations = {}
    model_grads = {}
    tc_feature_attributions = {}
    tc_feature_activations = {}
    tc_feature_grads = {}
    tc_error_proportions = {}
    sae_feature_attributions = {}
    sae_feature_activations = {}
    sae_feature_grads = {}
    sae_errors_proportions = {}






    # this code is long, but all it's doing is multiplying the grads by the activations
    # and recording grads, acts, and attributions in dictionaries to return to the user
    with torch.no_grad():
        for name, act in outputs_with_grads.model_activations.items():
            assert act.grad is not None
            raw_activation = act.detach().clone()
            model_attributions[name] = (act.grad * raw_activation).detach().clone()
            model_activations[name] = raw_activation
            model_grads[name] = act.grad.detach().clone()
        for name, act in outputs_with_grads.tc_activations.items():
            assert act.feature_acts.grad is not None
            assert act.tc_out.grad is not None
            raw_activation = act.feature_acts.detach().clone()
            tc_feature_attributions[name] = (
                (act.feature_acts.grad * raw_activation).detach().clone()
            )
            tc_feature_activations[name] = raw_activation
            tc_feature_grads[name] = act.feature_acts.grad.detach().clone()
            if include_error_term:
                assert act.tc_error.grad is not None
                error_grad_norm = act.tc_error.grad.norm().item()
            else:
                error_grad_norm = 0
            tc_out_norm = act.tc_out.grad.norm().item()
            tc_error_proportions[name] = error_grad_norm / (
                tc_out_norm + error_grad_norm + EPS
            )
        for name, act in outputs_with_grads.sae_activations.items():
            assert act.feature_acts.grad is not None
            assert act.sae_out.grad is not None
            raw_activation = act.feature_acts.detach().clone()
            sae_feature_attributions[name] = (
                (act.feature_acts.grad * raw_activation).detach().clone()
            )
            sae_feature_activations[name] = raw_activation
            sae_feature_grads[name] = act.feature_acts.grad.detach().clone()
            if include_error_term:
                assert act.sae_error.grad is not None
                error_grad_norm = act.sae_error.grad.norm().item()
            else:
                error_grad_norm = 0
            sae_out_norm = act.sae_out.grad.norm().item()
            sae_errors_proportions[name] = error_grad_norm / (
                sae_out_norm + error_grad_norm + EPS
            )
        return Attribution(
            model_attributions=model_attributions,
            model_activations=model_activations,
            model_grads=model_grads,
            tc_feature_attributions=tc_feature_attributions,
            tc_feature_activations=tc_feature_activations,
            tc_feature_grads=tc_feature_grads,
            tc_errors_attribution_proportion=tc_error_proportions,
            sae_feature_attributions=sae_feature_attributions,
            sae_feature_activations=sae_feature_activations,
            sae_feature_grads=sae_feature_grads,
            sae_errors_attribution_proportion=sae_errors_proportions,
        )




if __name__ == "__main__":

    from transformer_lens import HookedTransformer, utils
    model = HookedSAETransformer.from_pretrained('gpt2')


    transcoder_template = "/media/workspace//gpt-2-small-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"
    transcoders = []
    sparsities = []
    for i in range(5):
        transcoders.append(SparseAutoencoder.load_from_pretrained(f"{transcoder_template.format(i)}.pt").eval())
        sparsities.append(torch.load(f"{transcoder_template.format(i)}_log_feature_sparsity.pt"))

    tcs_dict = {tc.cfg.out_hook_point: tc for tc in transcoders}
    saes_dict = get_attention_sae_dict(list(range(5)),device = device)


    attr = calculate_feature_attribution(model, "Hello, my name is", lambda x: x.sum(), include_tcs=tcs_dict,include_saes = saes_dict)





"""
The process is simple select a feature from a layer, sae and perform attribution wrt to that feature
"""



