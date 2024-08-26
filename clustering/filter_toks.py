
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

# We want to filter the tokens that the model predicts correctly,
# We will say that a model correctly predict a token if its argmax logit if the correct tokne




def correct_pred(token_dataset, batch_size):

    data = DataLoader(token_dataset, batch_size = batch_size)

    for i,d in tqdm(enumerate(data)):
        input_ids = d['tokens']
        with torch.no_grad():
            logits = model(input_ids)

        logits = logits[:,:-1].contiguous()
        target_ids = input_ids[:,1:].contiguous()
        max_val, max_idx = logits.max(dim = -1)
        corr_pred = max_idx == target_ids

        corr_pred = corr_pred.view(input_ids.size(0),-1)
        check_point_dict = {"Batch":i,"Tokens":input_ids, "Corr Pred":corr_pred}

        torch.save(check_point_dict, f"checkpoints/corr_pred_{i}.pt")

if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("gpt2-small")
    dataset = load_dataset("/home/gerard/MI/pile-10k/", split = "train")
    token_dataset = tokenize_and_concatenate(dataset = dataset,tokenizer =  model.tokenizer, max_length=32)
    correct_pred(token_dataset, 32)






