


from sae_lens import SAE, SAEConfig, HookedSAETransformer
import torch
from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset


from filter_tokens import PredictionFilter
from collect_activations import ActivationsColector
from cluster_activations import  ClusteringAnalyzer







if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = HookedSAETransformer.from_pretrained("gpt2", device = device)
    data = load_dataset("/home/gerard/MI/pile-10k/", split = "train")
    tokens = tokenize_and_concatenate(data, model.tokenizer, max_length=128)
    # ======== Get the predictions and the tokens =========


    pred_filt = PredictionFilter(model, batch_size = 4, checkpoint_dir = "../checkpoints",final_dicts_dir = "../final_dicts",device = device, batches_to_process = 10)
    pred_filt.filter_predictions(tokens,save = True,strict = True,threshold = 0.1)
    pred_filt.get_correct_sequences(3)# The sequences of contiguous correct predictions must be at least 3 tokens long
    final_dicts_dir = pred_filt.final_dicts_dir_versioned


    acts = ActivationsColector(model, tokens, ["blocks.4.attn.hook_z","blocks.5.attn.hook_z"],"Features","../activations/",final_dicts_dir, cat_activations=True, quantize = True ,average = True, load = False, device = device)
    clusters = ClusteringAnalyzer(acts.activations, "../clusters/")
    clusters.perform_clustering(3, method = "kmeans")
    clusters.save_cluster_labels()



