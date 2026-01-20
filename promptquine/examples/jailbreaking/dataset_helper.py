import os
import json
import numpy as np 
import pandas as pd 
from transformers import AutoTokenizer
from typing import Optional, Tuple, List


def load_jailbreaking_dataset(
    dataset: str,
    split: str,
    base_path: str,
    max_size: Optional[int] = None
):
    assert dataset in ['Advbench'], "We only support Advbench"
    assert split in ['dev', 'test']

    filepath = os.path.join(base_path, f'{dataset}/{split}.csv')
    instances = pd.read_csv(filepath)

    # Option to keep only certain number of examples
    if max_size is not None:
        instances = instances[:max_size]

    return instances
