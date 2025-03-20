import hydra
import tiktoken
import os
import sys
sys.path.append("..")
import json
import copy
import sys
import time
import datetime
import pandas as pd
import random
from itertools import compress
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaTokenizer
from utils.task_wrappers import colorful_print, create_tabulist, PromptedTaskWrapperBase


class Pruner(object):
    """The Pruner base class for TAPruning and PromptQuine."""
    
    def __init__(self, data_dir: str) -> None:
        pass