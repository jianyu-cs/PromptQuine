from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Tuple, List

# Verbalizers
sst2_label_mapping = {0: 'terrible', 1: 'great'}
subj_label_mapping = {0: 'objective', 1: 'subjective'}
agnews_label_mapping = {0: 'World', 1: "Sports", 2: "Business", 3: "Tech"}
snli_label_mapping = {0: 'Yes', 1: 'Unknown', 2: 'No'}
yelp5_label_mapping = {0: 'terrible', 1: 'bad', 2: 'neutral', 3: 'good', 4: 'great'}
yahoo_label_mapping = {0: 'culture', 1: 'science', 2: 'health',
                      3: 'education', 4: 'computer', 5: 'sports',
                      6: 'business', 7: 'music', 8: 'family', 9: 'politics'}
mcq_label_mapping = {0: 'A', 1: 'B'}


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
        mode: str = "reduce",
) -> Tuple[PromptedClassificationDataset]: 
    source_texts, class_labels, num_classes, verbalizers, template = \
        load_few_shot_classification_dataset(dataset, 
                            dataset_seed, mode, 
                            base_path, 16,
                            task_lm)
    if dataset not in ["snli", 'piqa']:
        fsc_dataset = PromptedClassificationDataset(source_texts, 
                                                class_labels)
    elif dataset == 'snli':
        source_texts = [[x, y] for x, y in zip(source_texts[0], source_texts[1])]
        fsc_dataset = PromptedClassificationDataset(source_texts,
                                                class_labels)
    else:
        source_texts = [[x, y, z] for x, y, z in zip(source_texts[0], source_texts[1], source_texts[2])]
        fsc_dataset = PromptedClassificationDataset(source_texts,
                                                class_labels)

    return (fsc_dataset, num_classes, verbalizers, template)

def load_few_shot_classification_dataset(
    dataset: str,
    dataset_seed: Optional[int],
    split: str,
    base_path: str,
    num_shots: int,
    task_lm: str,
) -> Tuple[List[str]]:
    assert dataset in ['agnews', 'sst-2', 'subj',
                       'yelp-5', 'piqa', 'yahoo', 'snli']
    assert split in ['train', 'dev', 'test', 'reduce', 'reduce_test']

    seed_dict = {0:'16-100', 1:'16-13', 2:'16-21', 3:'16-42', 4:'16-87'}
    seed_path = seed_dict[dataset_seed]
    filepath = f'16-shot/{dataset}/{seed_path}/{split}.tsv'
    full_filepath = os.path.join(base_path, filepath)
    if split == 'reduce':
        filepath = f'reduction_dataset/{dataset}/dev.tsv'
        full_filepath = os.path.join(base_path, filepath)
    elif split =='reduce_test':
        filepath = f'reduction_dataset/{dataset}/test.tsv'
        full_filepath = os.path.join(base_path, filepath)
    print(full_filepath)
    df = pd.read_csv(full_filepath, sep='\t')
    if dataset not in ['snli', 'piqa']:
        if 'text' in df:
            source_texts = df.text.tolist()
        else: 
            source_texts = df.sentence.tolist()
    elif dataset == 'snli':
        source_premises = df.premise.tolist()
        source_hypos = df.hypothesis.tolist()
        source_texts = [source_premises, source_hypos]
    else:
        source_questions = df.question.tolist()
        source_option1s = df.option1.tolist()
        source_option2s = df.option2.tolist()
        
        source_texts = [source_questions, source_option1s, source_option2s]
        
    class_labels = df.label.tolist()

    verbalizers = get_dataset_verbalizers(dataset)
    num_classes = len(verbalizers)

    template = None
    
    return (source_texts, class_labels, 
            num_classes, verbalizers, template)

def get_dataset_verbalizers(dataset: str) -> List[str]: 
    if dataset in ['sst-2', 'yelp-2', 'mr', 'cr']:
        verbalizers = ['\u0120terrible', '\u0120great'] # num_classes
    elif dataset == 'agnews': 
        verbalizers = ['World', 'Sports', 'Business', 'Tech'] # num_classes
    elif dataset in ['sst-5', 'yelp-5']:
        verbalizers = ['\u0120terrible', '\u0120bad', '\u0120okay', 
                       '\u0120good', '\u0120great'] # num_classes
    elif dataset == 'subj':
        verbalizers = ['\u0120subjective', '\u0120objective']
    elif dataset == 'trec':
        verbalizers = ['\u0120Description', '\u0120Entity',
                    '\u0120Expression', '\u0120Human',
                    '\u0120Location', '\u0120Number']
    elif dataset == 'yahoo':
        verbalizers = ['culture', 'science',
                    'health', 'education',
                    'computer', 'sports',
                    'business', 'music',
                    'family', 'politics']
    elif dataset == 'dbpedia':
        verbalizers = ['\u0120Company', '\u0120Education',
                    '\u0120Artist', '\u0120Sports',
                    '\u0120Office', '\u0120Transportation',
                    '\u0120Building', '\u0120Natural',
                    '\u0120Village', '\u0120Animal',
                    '\u0120Plant', '\u0120Album',
                    '\u0120Film', '\u0120Written']
    return verbalizers

# PromptQuine
def make_balanced_classification_dataset(
        dataset: str, 
        dataset_seed: int,
        base_path: str,
        num_shots: int,  
        task_lm: str,
        data_split: bool,
        data_split_seed: int
):
    """Dataset Preparation for Prompt Pruning."""
    
    data_dict = {}
    source_texts, class_labels = load_balanced_classification_dataset(
        dataset, dataset_seed, base_path, num_shots,
        data_split, task_lm, data_split_seed
    )
    label_to_texts_idx = {}
    for i,label in enumerate(class_labels):
        if label not in label_to_texts_idx:
            label_to_texts_idx[label] = [i]
        else:
            label_to_texts_idx[label].append(i)
    
    source_texts_list = []
    labels_list = []
    for label in list(label_to_texts_idx.keys()):
        texts = []
        labels = [label] * len(label_to_texts_idx[label])
        text_indices = label_to_texts_idx[label]
        for index in text_indices:
            texts.append(source_texts[index])
        source_texts_list.append(texts)
        labels_list.append(labels)
            
    return source_texts_list, labels_list


def load_balanced_classification_dataset(
    dataset: str,
    dataset_seed: int,
    base_path: str,
    num_shots: int,
    data_split: bool,
    task_lm: str,
    data_split_seed: int
):
    assert dataset in ['agnews', 'sst-2', 'yelp-5', 'piqa', 'subj', 'yahoo', 'snli']

    seed_dict = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}
    seed_path = seed_dict[dataset_seed] 
    if data_split:
        sample_shots = num_shots//2
        filepath = f'{num_shots}_shot/{dataset}/{seed_path}/{sample_shots}_shot/{sample_shots}_shot_{data_split_seed}.tsv'
    else:
        filepath = f'{num_shots}_shot/{dataset}/{seed_path}/{num_shots}_shot_0.tsv'
        
    
    full_filepath = os.path.join(base_path, filepath)
    df = pd.read_csv(full_filepath, sep='\t')
    if dataset not in ['snli', 'piqa']:
        if 'text' in df:
            source_texts = df.text.tolist()
        else: 
            source_texts = df.sentence.tolist()
    elif dataset == 'snli':
        source_premises = df.premise.tolist()
        source_hypos = df.hypothesis.tolist()
        source_texts = [[source_premise, source_hypo] for source_premise, source_hypo in zip(source_premises, source_hypos)]
    else: # MCQ 
        source_questions = df.question.tolist()
        source_option1s = df.option1.tolist()
        source_option2s = df.option2.tolist()
        source_texts = [[source_question, source_option1, source_option2] for source_question, source_option1, source_option2 \
                        in zip(source_questions, source_option1s, source_option2s)]
        
    class_labels = df.label.tolist()
    class_labels_update = []
    
    for label in class_labels:
        if dataset == 'sst-2':
            class_labels_update.append(sst2_label_mapping[label])
        elif dataset == 'subj':
            class_labels_update.append(subj_label_mapping[label])
        elif dataset == 'agnews':
            class_labels_update.append(agnews_label_mapping[label])
        elif dataset == 'snli':
            class_labels_update.append(snli_label_mapping[label])
        elif dataset == 'yelp-5':
            class_labels_update.append(yelp5_label_mapping[label])
        elif dataset == 'yahoo':
            class_labels_update.append(yahoo_label_mapping[label])
        else:
            class_labels_update.append(mcq_label_mapping[label])       
    
    return (source_texts, class_labels_update)