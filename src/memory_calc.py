
# How much memory will storing activations take



import numpy as np
import h5py
import os


# 5000 prompts for 1 SAE gzip compression and an average of 2 positions is 1.2 GB (without quantization)

# 5000 prompts for 1 SAE gzip compression and an average of 2 positions is 0.35 GB (without quantization)

# 5000 prompts for 1 SAE gzip compression (opts = 9) and an average of 2 positions is 0.31 GB (without quantization)

def test_save_activations(compression = "gzip", chunks_size = None):
    activations_dict = {}
    # 5000 prompts

    for batch in range(50):
        activations_dict[batch] = {}
        for doc in range(100):
            activations_dict[batch][doc] = {}
            for hook in range(1):
                activations_dict[batch][doc][hook] = {}
                for pos in range(2):
                    activations_dict[batch][doc][hook][pos] = np.random.rand(1, 16000).astype(np.float16)

    def flatten_activations(activations_dict):
        """Helper function to flatten the nested activations dictionary."""
        for batch, batch_dict in activations_dict.items():  # Get the batch
            for doc, doc_list in batch_dict.items():  # Get the document
                for hook, hook_dict in doc_list.items():  # Get the hook
                    for pos, act in hook_dict.items():
                        key = f"{batch}/{doc}/{hook}/{pos}"
                        yield key, act

    filename = f"activations_Features_False.h5"
    versioned_dir = "../activations/test_quant"
    if not os.path.exists(versioned_dir):
        os.makedirs(versioned_dir)

    full_dir = os.path.join(versioned_dir,filename)

    with h5py.File(full_dir, "w") as f:
        for key, act in flatten_activations(activations_dict):
            group = f.require_group("/".join(key.split("/")[:-1]))
            dataset_name = key.split("/")[-1]
            group.create_dataset(dataset_name, data=act, compression=compression,compression_opts = 9, chunks=chunks_size)


test_save_activations(compression = "gzip", chunks_size = None)

