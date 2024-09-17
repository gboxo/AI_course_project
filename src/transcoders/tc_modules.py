
from transformer_lens.hook_points import NamesFilter
from sae_utils import get_attention_sae_dict
import json
import random
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from typing import List, Tuple, Union, Callable, Dict, Any, Optional
import torch
from functools import partial
from sae_training.sparse_autoencoder import SparseAutoencoder
from contextlib import contextmanager
from dataclasses import dataclass
from abc import ABC

from tc_utils import (
    compose_hooks,
    retain_grad_hook,
    detach_hook,
    mount_hooked_modules,
    apply_tc,
    apply_sae,
    Node,
    Cache
        )
from plot_circuit import visualize_circuit
from h_attr_utils import (
        get_attention_score_nodes,
        sort_nodes_by_attribution,
        get_upstream_nodes_pos,
        get_decoder,
        compute_prods,
        compute_QK_edges,
        compute_all_edges
        )



def get_ref_caching_hooks(
    model,
    names_filter,
    retain_grad: bool = False,
    cache: Optional[dict] = None,
) -> Tuple[dict, list]:
    """Creates hooks to keep references to activations. Note: It does not add the hooks to the model.

    Args:
        names_filter (NamesFilter, optional): Which activations to cache. Can be a list of strings (hook names) or a filter function mapping hook names to booleans. Defaults to lambda name: True.
        retain_grad (bool, optional): Whether to retain gradients for the activations. Defaults to False.
        cache (Optional[dict], optional): The cache to store activations in, a new dict is created by default. Defaults to None.

    Returns:
        cache (dict): The cache where activations will be stored.
        fwd_hooks (list): The forward hooks.
    """
    if cache is None:
        cache = {}

    if names_filter is None:
        names_filter = lambda name: True
    elif isinstance(names_filter, str):
        filter_str = names_filter
        names_filter = lambda name: name == filter_str
    elif isinstance(names_filter, list):
        filter_list = names_filter
        names_filter = lambda name: name in filter_list
    elif callable(names_filter):
        names_filter = names_filter
    else:
        raise ValueError("names_filter must be a string, list of strings, or function")
    assert callable(names_filter)  # Callable[[str], bool]

    def save_hook(tensor: torch.Tensor, hook: HookPoint):
        if retain_grad:
            tensor.retain_grad()
        cache[hook.name] = tensor

    fwd_hooks = []
    for name, hp in model.hook_dict.items():
        if names_filter(name):
            fwd_hooks.append((name, save_hook))

    return cache, fwd_hooks


def run_with_ref_cache(
    model,
    toks,
    names_filter: NamesFilter = None,
    retain_grad: bool = False,
    reset_hooks_end=True,
    clear_contexts=False,
    
):
    """
    Runs the model and returns the model output and a reference cache.

    Args:
        *model_args: Positional arguments for the model.
        names_filter (NamesFilter, optional): A filter for which activations to cache. Accepts None, str,
            list of str, or a function that takes a string and returns a bool. Defaults to None, which
            means cache everything.
        retain_grad (bool, optional): Whether to retain gradients for the activations. Defaults to False.
        **model_kwargs: Keyword arguments for the model.

    Returns:
        tuple: A tuple containing the model output and the reference cache.

    """

    cache_dict, fwd = get_ref_caching_hooks(
            model,
        names_filter,
        retain_grad=retain_grad,
    )

    with model.hooks(
        fwd_hooks=fwd,
        reset_hooks_end=reset_hooks_end,
        clear_contexts=clear_contexts,
    ):
        model_out = model(toks)

    return model_out, cache_dict


class Attributor(ABC):
    def __init__(
        self,
        model: HookedTransformer
    ):
        self.model = model

    def attribute(
        self,
        input: Any,
        target: Node,
        candidates: list[Node],
        **kwargs
    ):
        """
        Attribute the target hook point of the model to given candidates in the model, w.r.t. the given input.

        Args:
            input (Any): The input to the model.
            target: The target node to attribute.
            candidates: The intermediate nodes to attribute to.
            **kwargs: Additional keyword arguments.

        Returns:
            nx.MultiDiGraph: The attributed graph, i.e. the circuit. Each node and edge should have an attribute "attribution",
                showing its "importance" w.r.t. the target.
        """
        raise NotImplementedError

    def cache_nodes(
        self,
        toks: Any,
        nodes: list[Node],
    ):
        """
        Cache the activation of  in the model forward pass.

        Args:
            toks (Any): The input to the model.
            nodes (list[Node]): The nodes to cache.
        """
        output, cache = run_with_ref_cache(self.model,toks = toks, names_filter=[node.hook_point for node in nodes])
        return Cache(output, cache)


class HierachicalAttributor(Attributor):
    def attribute(
        self, 
        toks: Any,
        target: Node,
        candidates: list[Node],
        **kwargs
    ) -> Dict:
        """
        Attribute the target node of the model to given candidates in the model, w.r.t. the given input.

        Args:
            toks (Any): The input to the model.
            target: The target node to attribute.
            candidates: The intermediate nodes to attribute to.
            **kwargs: Additional keyword arguments.

        Returns:
            nx.MultiDiGraph: The attributed graph, i.e. the circuit. Each node and edge should have an attribute "attribution",
                showing its "importance" w.r.t. the target.
        """

        threshold: int = kwargs.get("threshold", 0.1)
    
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
        attribution_score_filter_hooks = {candidate: generate_attribution_score_filter_hook() for candidate in candidates}
        fwd_hooks = [(candidate.hook_point, compose_hooks(attribution_score_filter_hooks[candidate][0], retain_grad_hook)) for candidate in candidates]
        with self.model.hooks(
            fwd_hooks=fwd_hooks,
            bwd_hooks=[(candidate.hook_point, attribution_score_filter_hooks[candidate][1]) for candidate in candidates]
        ):
            cache = self.cache_nodes(toks, candidates + [target])
            cache[target].backward()

            # Construct the circuit
            circuit = {"nodes": {}, "edges": []}
            circuit["nodes"][target] = {"attribution": cache[target].sum().item(), "activation": cache[target].sum().item()}


            for candidate in candidates:
                if candidate.hook_point == target.hook_point:
                    continue
                grad = cache.grad(candidate)
                if grad is None:
                    continue
                attributions = grad * cache[candidate]
                if len(attributions.shape) == 0:
                    if attributions > threshold:
                        circuit["nodes"][candidate] = {"attribution": attributions.item(), "activation": cache[candidate].item()}
                else:
                    for index in (attributions > threshold).nonzero():
                        index = tuple(index.tolist())
                        node_name = candidate.append_reduction(*index)
                        # Add node to circuit
                        circuit["nodes"][node_name] = {"attribution": attributions[index].item(), "activation": cache[candidate][index].item()}
                        # Add edge to circuit
                        circuit["edges"].append((node_name, target, {"attribution": attributions[index].item()}))                        


        return circuit




@contextmanager
def join_contexts(model,tcs,saes):
    with apply_tc(model, tcs):
        with apply_sae(model, saes):
            with model.hooks([(f"blocks.{i}.attn.hook_attn_scores", detach_hook) for i in range(12)]):
                yield



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

    candidates = None

    with apply_tc(model, tcs):
        with apply_sae(model, saes):
            with model.hooks([(f"blocks.{i}.attn.hook_attn_scores", detach_hook) for i in range(12)]):
                attributor = HierachicalAttributor(model = model)

                target = Node("blocks.5.attn.hook_z.sae.hook_sae_acts_post",reduction="0.12.148")
                if candidates is None:
                    candidates = [Node(f"{sae.cfg.hook_name}.sae.hook_sae_acts_post") for sae in saes[:-1]] + [Node(f"{sae.cfg.out_hook_point}.sae.hook_hidden_post") for sae in tcs] + [Node(f"blocks.{i}.attn.hook_attn_scores") for i in range(12)]
                circuit = attributor.attribute(toks=toks, target=target, candidates=candidates, threshold=0.06)
    visualize_circuit(circuit)

# Check the sums


# %%

    total_attribution = circuit["nodes"][target]["attribution"]
    attribution_by_comp = {"attn_score":0,
                           "TC":0,
                           "SAE":0}

    for node in circuit["nodes"]:
        if node == target:
            continue

        if "attn_scores" in node.hook_point:
            attribution_by_comp["attn_score"] += circuit["nodes"][node]["attribution"]
        elif "hook_hidden_post" in node.hook_point:
            attribution_by_comp["TC"] += circuit["nodes"][node]["attribution"]
        elif "hook_sae_acts_post" in node.hook_point:
            attribution_by_comp["SAE"] += circuit["nodes"][node]["attribution"]

    print(attribution_by_comp)




