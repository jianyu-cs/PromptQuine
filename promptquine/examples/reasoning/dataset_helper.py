import os
import json
import numpy as np 
import pandas as pd 
from transformers import AutoTokenizer
from typing import Optional, Tuple, List

from promptquine.utils.reasoning import read_reasoning_data


def load_reasoning_dataset(
    dataset: str,
    split: str,
    max_size: Optional[int] = None
):
    assert dataset in ['gsm8k', 'mawps']
    assert split in ['train', 'dev', 'test']

    filepath = os.path.join("./data", f'{dataset}/{split}.json')
    instances = read_reasoning_data(filepath)

    # Option to keep only certain number of examples
    if max_size is not None:
        instances = instances[:max_size]

    return instances
