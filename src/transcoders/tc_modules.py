from multiprocess import context
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

from transformer_lens.hook_points import NamesFilter
from sae_utils import get_attention_sae_dict




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

import networkx as nx
import matplotlib.pyplot as plt
import re

def extract_vertical_position(node_name):
    match = re.search(r'blocks\.(\d+)', node_name)
    return int(match.group(1)) if match else 0

def extract_horizontal_position(node_name):
    match = re.search(r'\[.*?(\d+)', node_name)
    return int(match.group(1)) if match else 0
def visualize_circuit(circuit):
    # Create a new directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    for node, data in circuit["nodes"].items():
        G.add_node(node, **data)

    # Add edges to the graph
    for edge in circuit["edges"]:
        source, target, data = edge
        G.add_edge(source, target, **data)

    # Set up the plot
    plt.figure(figsize=(20, 16))
    
    # Calculate node sizes based on attribution
    node_sizes = [data['attribution'] * 1000 for node, data in G.nodes(data=True)]
    
    # Calculate edge widths based on attribution
    edge_widths = [data['attribution'] * 5 for (u, v, data) in G.edges(data=True)]

    # Create custom positions for nodes with jitter
    pos = {}
    for node in G.nodes():
        x = extract_horizontal_position(node.reduction) + random.uniform(-0.3, 0.3)
        y = extract_vertical_position(node.hook_point) + random.uniform(-0.6, 0.6)  # Not negated now
        pos[node] = (x, y)

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', arrows=True)
    
    # Adjust label positions slightly above nodes
    label_pos = {node: (x, y + 0.1) for node, (x, y) in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=8)

    # Add a title
    plt.title("Circuit Visualization (Sorted with Jitter)")
    
    # Show the plot
    plt.axis('off')
    plt.tight_layout()
    plt.show()




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




# How to fint the edges between nodes

# To find the edges between attention sccore nodes and features in lower layers we:
# 1. Select the most important attention score node, get the WQ and wK matrices for that head.
# 2. Get the decoder of the features in the lower layers present in that position.
# 3. Multiply the WDi@WQ@WK@WDj
# 4. Get the total attribution and normalize it to add up  to the total attribution of the attention score node.

# %%

# Get the  the attention score nodes of the first layer (starting from 1)

def get_attention_score_nodes(circuit,layer):
    return [node for node in circuit["nodes"] if f"blocks.{layer}" in node.hook_point and "attn_scores" in node.hook_point]


# Sort the attention score nodes by attribution
def sort_nodes_by_attribution(circuit,nodes):
    return sorted(nodes,key = lambda node: circuit["nodes"][node]["attribution"],reverse=True)

# Get the WQ and WK matrices of the attention score node
def attn_matrices(model,attn_score_node):
    layer = int(attn_score_node.hook_point.split(".")[1])
    head = int(attn_score_node.reduction.split(".")[1])
    WQ = model.blocks[layer].attn.W_Q.detach()[head]
    WK = model.blocks[layer].attn.W_K.detach()[head]
    return WQ,WK

# Get the upstream nodes in a certain position
def get_upstream_nodes_pos(circuit,pos,layer):
    upstream_nodes = []
    for node in circuit["nodes"].keys():
        node_layer = int(node.hook_point.split(".")[1])
        node_pos = node.reduction.split(".")[1]

        if "sae" in node.hook_point and node_layer < layer and node_pos == pos:
            upstream_nodes.append(node)
    return upstream_nodes

# Get the decoder of a SAE node

def get_decoder(sae_node):
    assert "attn_scores" not in sae_node.hook_point
    if "hook_sae_acts_post" in sae_node.hook_point:
        layer = int(sae_node.hook_point.split(".")[1])
        feature = int(sae_node.reduction.split(".")[-1])
        return saes[layer].W_dec.detach()[feature]
    elif "hook_hidden_post" in sae_node.hook_point:
        layer = int(sae_node.hook_point.split(".")[1])
        feature = int(sae_node.reduction.split(".")[-1])
        return tcs[layer].W_dec.detach()[feature]



# Compute the multiplication of the matrices


def compute_prods(WQ,WK,decoder1,decoder2):
    return decoder1 @ WQ @ WK.T @ decoder2.T
# Compute the attributions matrix multiplication

def compute_QK_edges(circuit,layer):
    attn_score_nodes = get_attention_score_nodes(circuit,layer)
    attn_score_nodes = sort_nodes_by_attribution(circuit,attn_score_nodes)
    for attn_score_node in attn_score_nodes:
        attrb = circuit["nodes"][attn_score_node]["attribution"]
        WQ,WK = attn_matrices(model,attn_score_node)
        pos1 = attn_score_node.reduction.split(".")[2]
        pos2 = attn_score_node.reduction.split(".")[3]
        upstream_nodes_pos1 = get_upstream_nodes_pos(circuit,pos1,layer)
        upstream_nodes_pos2 = get_upstream_nodes_pos(circuit,pos2,layer)
        prods = []
        for upstream_node1 in upstream_nodes_pos1:
            decoder1 = get_decoder(upstream_node1)
            for upstream_node2 in upstream_nodes_pos2:
                decoder2 = get_decoder(upstream_node2)
                prod = compute_prods(WQ,WK,decoder1,decoder2)
                prods.append(prod)
        for i,upstream_node1 in enumerate(upstream_nodes_pos1): 
            for j,upstream_node2 in enumerate(upstream_nodes_pos2):
                prod = prods[i*len(upstream_nodes_pos2)+j]
                circuit["edges"].append((upstream_node1,upstream_node2,{"attribution":(attrb*(prod/sum(prods))).item()}))
    return circuit




def compute_all_edges(circuit):
    for layer in range(1,6):
        circuit = compute_QK_edges(circuit,layer)
    return circuit




