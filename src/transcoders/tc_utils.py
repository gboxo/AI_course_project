from sae_lens import SAE
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from typing import List, Tuple, Union, Callable, Dict, Any
import torch
from functools import partial
from sae_training.sparse_autoencoder import SparseAutoencoder
from contextlib import contextmanager
from dataclasses import dataclass
from abc import ABC

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

@contextmanager
def mount_hooked_modules(
    model: HookedTransformer,
    hooked_modules: List[Tuple[str, str, SparseAutoencoder]]
):
    """
    Mount the hooked modules to the model.
    """
    for name, module_name, sae in hooked_modules:
        hook_point = model.mod_dict[name]
        hook_point._modules[module_name] = sae

    model.setup()
    yield model

@contextmanager
def apply_sae(model:HookedTransformer,
            saes: list[SAE]) :

    fwd_hooks: list[Tuple[Union[str, Callable], Callable]] = []
    def get_fwd_hooks(sae: SAE) -> list[Tuple[Union[str, Callable], Callable]]:
        def hook(tensor: torch.Tensor, hook: HookPoint):
            reconstructed = sae.forward(tensor)
            return reconstructed + (tensor - reconstructed).detach()
        return [(sae.cfg.hook_name, hook)]

    for sae in saes:
        hooks = get_fwd_hooks(sae)
        fwd_hooks.extend(hooks)
    with mount_hooked_modules(model,[(sae.cfg.hook_name, "sae", sae) for sae in saes]):
        with model.hooks(fwd_hooks):
            yield model

def set_deep_attr(obj: Any, path: str, value: Any):
    """Helper function to change the value of a nested attribute from a object.
    In practice used to swap HookedTransformer HookPoints (eg model.blocks[0].attn.hook_z) with HookedSAEs and vice versa

    Args:
        obj: Any object. In practice, this is a HookedTransformer (or subclass)
        path: str. The path to the attribute you want to access. (eg "blocks.0.attn.hook_z")
        value: Any. The value you want to set the attribute to (eg a HookedSAE object)
    """
    parts = path.split(".")
    # Navigate to the last component in the path
    for part in parts[:-1]:
        if part.isdigit():  # This is a list index
            obj = obj[int(part)]
        else:  # This is an attribute
            obj = getattr(obj, part)
    # Set the value on the final attribute
    setattr(obj, parts[-1], value)

@contextmanager
def apply_tc(
    model: HookedTransformer,
    saes: list[SparseAutoencoder]
):
    """
    Apply the sparse autoencoders to the model.
    """
    fwd_hooks: list[Tuple[Union[str, Callable], Callable]] = []
    def get_fwd_hooks(sae: SparseAutoencoder) -> list[Tuple[Union[str, Callable], Callable]]:
        x = None
        def hook_in(tensor: torch.Tensor, hook: HookPoint):
            nonlocal x
            x = tensor
            return tensor
        def hook_out(tensor: torch.Tensor, hook: HookPoint):
            nonlocal x
            assert x is not None, "hook_in must be called before hook_out."
            reconstructed,_,_,_,_,_, = sae.forward(x)
            x = None

            return reconstructed + (tensor - reconstructed).detach()
        return [(sae.cfg.hook_point, hook_in), (sae.cfg.out_hook_point, hook_out)]
    for sae in saes:
        hooks = get_fwd_hooks(sae)
        fwd_hooks.extend(hooks)
    with mount_hooked_modules(model,[(sae.cfg.out_hook_point, "sae", sae) for sae in saes]):
        with model.hooks(fwd_hooks):
            yield model


@dataclass(frozen=True)
class Node:
    """
    A node in the circuit.
    """

    hook_point: str | None
    """ The hook point of the node. None means the node is the output of the model. """
    reduction: str | None = None
    """ The reduction function to apply to the node. """

    def reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        reductions = self.reduction.split(".") if self.reduction is not None else []
        for reduction in reductions:
            if reduction == "max":
                tensor = tensor.max()
            elif reduction == "mean":
                tensor = tensor.mean()
            elif reduction == "sum":
                tensor = tensor.sum()
            else:
                try:
                    index = int(reduction)
                    tensor = tensor[index]
                except ValueError:
                    raise ValueError(f"Unknown reduction function: {reduction} in {self.reduction}.")
        return tensor
    
    def append_reduction(self, *reduction: list[str | int]) -> "Node":
        reduction: str = ".".join(map(str, reduction))
        return Node(self.hook_point, f"{self.reduction}.{reduction}" if self.reduction is not None else reduction)
    
    def __hash__(self):
        return hash((self.hook_point, self.reduction))
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.hook_point == other.hook_point and self.reduction == other.reduction
    
    def __str__(self) -> str:
        hook_point = self.hook_point if self.hook_point is not None else "output"
        return f"{hook_point}.{self.reduction}" if self.reduction is not None else hook_point

class Cache:
    def __init__(self, output, cache: dict[str, torch.Tensor]):
        self.cache = cache
        self.output = output

    def tensor(self, node: Node) -> torch.Tensor:
        return node.reduce(self[node.hook_point])
    
    def grad(self, node: Node) -> torch.Tensor | None:
        grad = self[node.hook_point].grad
        return node.reduce(grad) if grad is not None else None
    
    def __getitem__(self, key: Node | str | None) -> torch.Tensor:
        if isinstance(key, Node):
            return self.tensor(key)
        return self.cache[key] if key is not None else self.output
