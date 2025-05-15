import os
import json
import numpy as np 
import pandas as pd 
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Tuple, List


class ReasoningDataset(Dataset):
    def __init__(self, source_texts, target_labels):
        assert len(source_texts) == len(target_labels)
        self.source_texts = source_texts
        self.target_labels = target_labels

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        item = {'source_texts': self.source_texts[idx],
                'target_labels': self.target_labels[idx]}
        return item


def load_reasoning_dataset(
    dataset: str,
    split: str,
    base_path: str,
    max_size: Optional[int]
) -> Tuple[List[str]]:
    assert dataset in ['gsm8k', 'mawps']
    assert split in ['train', 'dev', 'test']

    filepath = f'{dataset}/{split}.json'
    full_filepath = os.path.join(base_path, filepath)
    with open(full_filepath) as f:
        instances = json.load(f)

    # Option to keep only certain number of examples
    if max_size is not None:
        instances = sentences[:max_size]
    
    source_texts = []
    target_labels = []
    if dataset == 'gsm8k':
        for instance in instances:
            source_texts.append(instance['instruction'])
            target_labels.append(instance['answer'])
    else:
        for instance in instances:
            source_texts.append(instance['original_text'])
            target_labels.append(instance['ans'])

    return source_texts, target_labels