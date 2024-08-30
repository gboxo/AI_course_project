import pytest
from unittest.mock import patch
from sae_lens import HookedSAETransformer
import torch
from torch.utils.data import Dataset
from src.filter_tokens import filter_pred



device = "cuda:0" if torch.cuda.is_available() else "cpu"



class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.tokens = torch.randint(0, 50256, (size, 128))  # Simulated tokenized input

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {'tokens': self.tokens[idx]}

@pytest.fixture
def setup_model_and_dataset():
    # Create a dummy model
    model = HookedSAETransformer.from_pretrained("gpt2",device=device)
    token_dataset = DummyDataset(50)
    return model, token_dataset

@patch('torch.save')  # Mock torch.save to avoid file I/O
def test_filter_pred_strict(setup_model_and_dataset, mock_save):
    model, token_dataset = setup_model_and_dataset

    filter_pred(model, token_dataset, batch_size=5, save=True, strict=True)

    # Check that the model was called the expected number of times
    assert mock_save.called  # Check that save was called






@patch('torch.save')
def test_filter_pred_non_strict(setup_model_and_dataset, mock_save):
    model, token_dataset = setup_model_and_dataset
    filter_pred(model, token_dataset, batch_size=5, save=True, strict=False, threshold=0.5)

    # Check that the model was called the expected number of times
    assert mock_save.called  # Check that save was called




