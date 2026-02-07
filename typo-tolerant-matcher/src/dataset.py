import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict


class CharTokenizer:
    """Simple character-level tokenizer"""
    def __init__(self, max_len=50):
        self.max_len = max_len
        # ASCII printable characters
        self.char_to_idx = {chr(i): i for i in range(128)}
        self.char_to_idx['<PAD>'] = 0
        self.char_to_idx['<UNK>'] = 1

    def encode(self, text: str) -> List[int]:
        """Encode text to character indices"""
        text = text.lower()[:self.max_len]
        indices = [self.char_to_idx.get(c, 1) for c in text]

        # Pad to max_len
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))

        return indices

    def decode(self, indices: List[int]) -> str:
        """Decode indices back to text"""
        idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        return ''.join([idx_to_char.get(idx, '') for idx in indices if idx != 0])


class FoodMatchDataset(Dataset):
    """Dataset for food query matching"""
    def __init__(self, data: List[Dict], tokenizer: CharTokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        query_ids = self.tokenizer.encode(item['query'])
        target_ids = self.tokenizer.encode(item['target'])
        label = float(item['label'])

        return {
            'query': torch.tensor(query_ids, dtype=torch.long),
            'target': torch.tensor(target_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    queries = torch.stack([item['query'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    return {
        'query': queries,
        'target': targets,
        'label': labels
    }
