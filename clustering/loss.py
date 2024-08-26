# %%

from plotly.graph_objs import Data
from torch import return_types
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from tqdm import tqdm

from torch.nn import CrossEntropyLoss 
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
torch.enable_grad(False)

# %%







def compute_loss(token_dataset, batch_size):

    data = DataLoader(token_dataset, batch_size = batch_size)

    for i,d in tqdm(enumerate(data)):
        input_ids = d['tokens']
        with torch.no_grad():
            logits = model(input_ids)

        logits = logits[:,:-1].contiguous()
        target_ids = input_ids[:,1:].contiguous()

        loss_fn = CrossEntropyLoss(reduction = "none")
        loss = loss_fn(logits.view(-1, logits.size(-1)),target_ids.view(-1))
        loss = loss.view(input_ids.size(0),-1)
        check_point_dict = {"Batch":i,"Tokens":input_ids, "Loss":loss}

        torch.save(check_point_dict, f"checkpoints/loss_checkpoint_{i}.pt")

if __name__ == "__main__":

    model = HookedTransformer.from_pretrained("gpt2-small")
    dataset = load_dataset("/home/gerard/MI/pile-10k/", split = "train")
    token_dataset = tokenize_and_concatenate(dataset = dataset,tokenizer =  model.tokenizer, max_length=32)
    compute_loss(token_dataset, 32)





