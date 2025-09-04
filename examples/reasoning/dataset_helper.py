#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dataset_helper.py
# Author : Jianyu Wang
# Email  : jiw102@ucsd.edu
# Date   : 07/10/2025
#
# Distributed under terms of the MIT license.

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
) -> ReasoningDataset:
    assert dataset in ['gsm8k', 'mawps']
    assert split in ['train', 'dev', 'test']

    filepath = os.path.join(base_path, f'{dataset}/{split}.json')
    with open(filepath) as f:
        instances = json.load(f)

    # Option to keep only certain number of examples
    if max_size is not None:
        instances = instances[:max_size]
    
    source_field, label_field = {
        'gsm8k': ('instruction', 'answer'),
        'mawps': ('original_text', 'ans')
    }[dataset]

    source_texts = [inst[source_field] for inst in instances]
    target_labels = [inst[label_field] for inst in instances]

    return ReasoningDataset(source_texts, target_labels)

def make_reasoning_dataset(
    