
from typing import List, Dict, Tuple, Any, Optional, Literal
import json
import os
from sae_lens import SAE, SAEConfig, HookedSAETransformer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

class PredictionFilter:
    def __init__(self, model: HookedSAETransformer, batch_size: int, checkpoint_dir: str = "../checkpoints",final_dicts_dir: str = "../final_dicts"):
        self.model = model
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir_versioned = self._create_versioned_dir(self.checkpoint_dir)
        self.final_dicts_dir = final_dicts_dir
        self.final_dicts_dir_versioned = self._create_versioned_dir(self.final_dicts_dir)

    def _create_versioned_dir(self,base_dir) -> str:
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        version = 0
        while True:
            versioned_dir = os.path.join(base_dir, f"version_{version}")
            if not os.path.exists(versioned_dir):
                os.makedirs(versioned_dir)
                return versioned_dir
            version += 1


    def filter_predictions(self, token_dataset, save: bool = True, strict: bool = False, threshold: float = 0.1):
        if not strict:
            assert threshold is not None, "Threshold must be provided if strict is False"

        dataset = DataLoader(token_dataset, batch_size=self.batch_size)
        check_point_dict = defaultdict(dict)

        for i, d in tqdm(enumerate(dataset)):
            if i == 10:  # Limiting to 10 batches for demonstration
                break
            input_ids = d['tokens']
            with torch.no_grad():
                logits = self.model(input_ids)

            logits = logits[:, :-1].contiguous()
            target_ids = input_ids[:, 1:].contiguous()
            if strict:
                argmax_pred = torch.argmax(logits, dim=-1)
                pred = (argmax_pred == target_ids).float()
                pred_pos = torch.where(pred)
            else:
                loss = nn.CrossEntropyLoss(reduction="none")
                loss_val = loss(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                loss_val = loss_val.view(logits.size(0), logits.size(1))
                pred = (loss_val < threshold).float()
                pred_pos = torch.where(pred)

            dict_doc_pos = defaultdict(list)
            for doc, ind in zip(pred_pos[0], pred_pos[1]):
                dict_doc_pos[doc.item()].append(ind.item())

            check_point_dict[f"Batch {i}"] = dict_doc_pos
            if save:
                self._save_checkpoint(check_point_dict, i, strict, threshold)

    def _save_checkpoint(self, check_point_dict: Dict, batch_index: int, strict: bool, threshold: Optional[float]):
        if strict:
            file_path = os.path.join(self.checkpoint_dir_versioned, f"pred_checkpoint_{batch_index}.pt")
        else:
            file_path = os.path.join(self.checkpoint_dir_versioned, f"pred_checkpoint_{batch_index}_th_{threshold}.pt")
        
        with open(file_path, "wb") as f:
            torch.save(check_point_dict, f)

    def get_correct_sequences(self, seq_length: int) -> None:
        all_checkpoints = os.listdir(self.checkpoint_dir_versioned)
        all_contiguous_positions = {}

        for checkpoint in all_checkpoints:
            with open(os.path.join(self.checkpoint_dir_versioned, checkpoint), "rb") as f:
                checkpoint_dict = torch.load(f)

            for batch, pred_dict in checkpoint_dict.items():
                all_contiguous_positions[batch] = defaultdict(list)
                for doc, pos_list in pred_dict.items():
                    if len(pos_list) == 0:
                        continue
                    contiguous_positions = []
                    start = pos_list[0]
                    end = pos_list[0]
                    for i in range(1, len(pos_list)):
                        if pos_list[i] == end + 1:
                            end = pos_list[i]
                        else:
                            if end - start >= seq_length:
                                contiguous_positions.append((start, end))
                            start = pos_list[i]
                            end = pos_list[i]

                    all_contiguous_positions[batch][doc] = contiguous_positions
        self._save_final_dict(all_contiguous_positions)

    def _save_final_dict(self, final_dict: Dict):
        with open(os.path.join(self.final_dicts_dir_versioned, "final_dict.json"), "w") as f:
            json.dump(final_dict, f)

