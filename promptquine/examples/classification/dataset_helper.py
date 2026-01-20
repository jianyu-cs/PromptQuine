import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Tuple, List


class PromptedClassificationDataset(Dataset):
    def __init__(
        self, 
        source_texts: List[str], 
        class_labels: List[str]
    ):
        assert len(source_texts) == len(class_labels)
        self.source_texts = source_texts
        self.class_labels = class_labels

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        item = {'source_texts': self.source_texts[idx],
                'class_labels': self.class_labels[idx]}
        return item

class PromptedClassificationPairedDataset(Dataset):
    def __init__(
        self, 
        source_premises: List[str], 
        source_hypos: List[str],
        class_labels: List[str]
    ):
        print(len(source_premises), len(class_labels))
        #assert len(source_premises) == len(class_labels)
        self.source_premises = source_premises
        self.source_hypos = source_hypos
        self.class_labels = class_labels

    def __len__(self):
        return len(self.source_premises)

    def __getitem__(self, idx):
        item = {'source_premises': self.source_premises[idx],
                'source_hypos': self.source_hypos[idx],
                'class_labels': self.class_labels[idx]}
        return item


def make_few_shot_classification_dataset(
        config: "DictConfig"
): 
    data_dict = {}
    for split in ['train', 'dev', 'test']: 
        source_texts, class_labels, num_classes, verbalizers, template = \
            load_few_shot_classification_dataset(config.dataset, 
                                                 config.dataset_seed, 
                                                 split, config.base_path, 
                                                 config.num_shots)
        fsc_dataset = PromptedClassificationDataset(source_texts, 
                                                    class_labels)
        data_dict[split] = fsc_dataset

    return (data_dict['train'], data_dict['dev'], data_dict['test'],
            num_classes, verbalizers, template)

def make_classification_dataset(
        dataset: str,
        dataset_seed: int,
        base_path: str,
        task_lm: str,
        quine_fewshot: bool = False,
        num_shots: int = 0,
        # PromptQuine-Only
        data_split: bool = True, 
        data_split_seed: int = 0,
        # TAPruning-Only
        mode: str = "reduce"
) -> PromptedClassificationDataset:
    if quine_fewshot:
        # PromptQuine mode
        source_texts, class_labels = load_classification_dataset(
            dataset=dataset,
            dataset_seed=dataset_seed,
            base_path=base_path,
            num_shots=num_shots,
            mode='PromptQuine',
            data_split=data_split,
            data_split_seed=data_split_seed,
            task_lm=task_lm
        )
    else:
        # TAPruning mode
        source_texts, class_labels = load_classification_dataset(
            dataset=dataset,
            dataset_seed=dataset_seed,
            base_path=base_path,
            num_shots=16,
            mode='TAPruning',
            split=mode,  # reuse mode as split
            task_lm=task_lm
        )
    
        # ---------------- unify snli / MCQ structure ----------------
        if dataset in ['snli', 'piqa']:
            if dataset == 'snli':
                source_texts = [[x, y] for x, y in zip(source_texts[0], source_texts[1])]
            else:  # MCQ
                source_texts = [[x, y, z] for x, y, z in zip(source_texts[0], source_texts[1], source_texts[2])]

    fsc_dataset = PromptedClassificationDataset(source_texts, class_labels)
    return fsc_dataset

def load_classification_dataset(
    dataset: str,
    dataset_seed: int,
    base_path: str,
    num_shots: int,
    mode: str = 'TAPruning',  # 'TAPruning' or 'PromptQuine'
    split: Optional[str] = None,  # Only for TAPruning
    data_split: Optional[bool] = False,  # Only for PromptQuine
    data_split_seed: Optional[int] = 0,
    task_lm: Optional[str] = None
) -> Tuple:
    assert dataset in ['agnews', 'sst-2', 'subj',
                       'yelp-5', 'piqa', 'yahoo', 'snli']
    # ---------------- Path Handling ----------------
    if mode == 'TAPruning':
        seed_dict = {0:'16-100', 1:'16-13', 2:'16-21', 3:'16-42', 4:'16-87'}
        seed_path = seed_dict[dataset_seed]
        if split in ['reduce', 'reduce_test']:
            filepath = f'reduction_dataset/{dataset}/{ "dev" if split=="reduce" else "test"}.tsv'
        else:
            filepath = f'test/{dataset}/test.tsv' # Test set for both TAPruning and PromptQuine
    elif mode == 'PromptQuine':
        seed_dict = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}
        seed_path = seed_dict[dataset_seed]
        if data_split:
            sample_shots = num_shots // 2
            filepath = f'{num_shots}_shot/{dataset}/{seed_path}/{sample_shots}_shot/{sample_shots}_shot_{data_split_seed}.tsv'
        else:
            filepath = f'{num_shots}_shot/{dataset}/{seed_path}/{num_shots}_shot_0.tsv'
    else:
        raise ValueError(f"Unknown mode {mode}")
    
    full_filepath = os.path.join(base_path, filepath)
    print(full_filepath)
    
    # ---------------- Load Data ----------------
    df = pd.read_csv(full_filepath, sep='\t')
    
    if dataset not in ['snli', 'piqa']:
        text_col = 'text' if 'text' in df else 'sentence'
        source_texts = df[text_col].tolist()
    elif dataset == 'snli':
        premises = df.premise.tolist()
        hypos = df.hypothesis.tolist()
        if mode == 'TAPruning':
            source_texts = [premises, hypos]
        else:  # PromptQuine
            source_texts = [[p, h] for p, h in zip(premises, hypos)]
    else:  # MCQ
        questions = df.question.tolist()
        option1s = df.option1.tolist()
        option2s = df.option2.tolist()
        if mode == 'TAPruning':
            source_texts = [questions, option1s, option2s]
        else:
            source_texts = [[q, o1, o2] for q, o1, o2 in zip(questions, option1s, option2s)]
    
    # ---------------- Labels ----------------
    class_labels = df.label.tolist()
    
    if mode == 'TAPruning':
        return source_texts, class_labels
    else:  # PromptQuine
        if dataset == "subj":
            # Bug Fix
            class_labels = [1 - label for label in class_labels]
        return source_texts, class_labels
