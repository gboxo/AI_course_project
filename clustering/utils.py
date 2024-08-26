import os
import torch
from transformer_lens import HookedTransformer
def compute_average_loss_every_n_tokens(loss_tensor, n):
    """
    Computes the average loss every n tokens for each sequence in the batch.

    Args:
    - loss_tensor (torch.Tensor): A tensor of shape [batch, seq_len] containing the losses.
    - n (int): The number of tokens over which to compute the average loss.

    Returns:
    - avg_losses (torch.Tensor): A tensor of shape [batch, seq_len - n + 1] containing the average losses.
    """
    batch_size, seq_len = loss_tensor.shape
    avg_losses = torch.zeros(batch_size, seq_len - n + 1)
    
    for i in range(seq_len - n + 1):
        avg_losses[:, i] = loss_tensor[:, i:i+n].mean(dim=1)
    
    return avg_losses





model = HookedTransformer.from_pretrained("gpt2-small")

# %%

import os

def join_all_files():

# Specify the directory containing the .pt files and the output file name
    directory = "/mnt/myssd/tutorial/clustering/checkpoints/"
    output_file = 'joined_file.pt'

# Initialize lists to store tensors from each file
    loss_tensors = []
    tokens_tensors = []

# Load each .pt file and append the tensors to the lists
    for filename in os.listdir(directory):
        if filename.endswith('.pt'):
            file_path = os.path.join(directory, filename)
            data = torch.load(file_path)
            loss_tensors.append(data['Loss'])
            tokens_tensors.append(data['Tokens'])

# Concatenate the tensors from the lists
    concatenated_loss = torch.cat(loss_tensors)
    concatenated_tokens = torch.cat(tokens_tensors)

# Create a new dictionary with the concatenated tensors and save it to a new .pt file
    concatenated_data = {
        'loss': concatenated_loss,
        'tokens': concatenated_tokens
    }
    torch.save(concatenated_data, output_file)

    print(f'All .pt files have been successfully joined into {output_file}')
