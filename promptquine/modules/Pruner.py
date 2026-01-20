import hydra
import tiktoken
import os
import sys
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

from promptquine.utils.jailbreaking import format_chat_prompt

class Pruner(object):
    """The Pruner base class for TAPruning and PromptQuine."""
    
    def __init__(self, data_dir: str) -> None:
        pass

    def _parse_prompt_dict(self, prompt: dict) -> str:
        """Parse prompt dictionary and format as chat prompt.
        
        Args:
            prompt: Dictionary containing 'prompt', 'inputs', 'outputs'
            
        Returns:
            Formatted chat prompt string
        """
        return format_chat_prompt(
            user_prompt="{sentence_1}",
            demo_inputs=prompt['inputs'],
            demo_outputs=prompt['outputs'],
            model_name=self.task_lm,
        )