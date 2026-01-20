"""Utilities for prompt manipulation."""
import json
from typing import List, Dict, Tuple

def load_prompts(prompts_path: str) -> List[Dict]:
    """Load prompt templates from JSONL file"""
    prompt_dict_list = []
    with open(prompts_path, 'r') as f:
        for line in f:
            prompt_dict_list.append(json.loads(line))
    return prompt_dict_list