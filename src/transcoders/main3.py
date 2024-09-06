# Try to get transcoders as transformer lens modules


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






if __name__ == "__main__":

    from transformer_lens import HookedTransformer, utils
    model = HookedTransformer.from_pretrained('gpt2')


    transcoder_template = "/media/workspace/gpt-2-small-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"
    transcoders = []
    for i in range(4,7):
        transcoders.append(SparseAutoencoder.load_from_pretrained(f"{transcoder_template.format(i)}.pt").eval())

    tcs_dict = {tc.cfg.out_hook_point: tc for tc in transcoders}


    # Run with transcoders


    fwd_hooks: list[Tuple[Union[str, Callable], Callable]] = []
    retain_grad = True
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

    def setup(model):
        """
        Sets up model.

        This function must be called in the model's `__init__` method AFTER defining all layers. It
        adds a parameter to each module containing its name, and builds a dictionary mapping module
        names to the module instances. It also initializes a hook dictionary for modules of type
        "HookPoint".
        """
        model.mod_dict = {}
        model.hook_dict = {}
        for name, module in model.named_modules():
            if name == "":
                continue
            module.name = name
            model.mod_dict[name] = module
            # TODO: is the bottom line the same as "if "HookPoint" in str(type(module)):"
            if isinstance(module, HookPoint):
                model.hook_dict[name] = module

    # Add the TC in a smilar fashion as the SAELens
    for tc in tcs_dict.values():
        set_deep_attr(model, tc.cfg.hook_point, tc)
        setup(model)

    cache, logits = model.run_with_cache("the")







        

