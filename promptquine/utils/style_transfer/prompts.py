"""Utilities for prompt manipulation."""
import json
import copy
from typing import List, Dict, Tuple, Any

import pandas as pd


fields = ["prompt", "reward", "length", "mask", "gm", "content", "style", "fluency"]
KEY_TO_INDEX = {k: i for i, k in enumerate(fields)}

METRICS = (
        "joint_score",
        "gm",
        "content",
        "style",
        "fluency",
    )

def get_style_transfer_field(lst: list, key: str, default=None, reward_driven=None):
    """
    Retrieve the value from a list corresponding to the given key.
    
    lst: the list object
    key: the field name, e.g., 'prompt', 'accuracy', etc.
    default: the value to return if the key does not exist or the index is out of range
    """
    if reward_driven is not None:
        return lst[KEY_TO_INDEX["reward"]]

    idx = KEY_TO_INDEX.get(key)
    if idx is None or idx >= len(lst):
        return default
    return lst[idx]

def should_be_evaluated_next_round_for_style_transfer(eval_result: Any, min_reward: float):
    if isinstance(eval_result, dict):
        # Dict form: {'reward': 0.90, ...}
        reward = eval_result.get('joint_score', 0.0)
    elif isinstance(eval_result, tuple) and len(eval_result) >= 1:
        # Tuple form: (reward, ...)
        reward = float(eval_result[0])
    else:
        raise ValueError(f"Unsupported eval_result: {type(eval_result)}")
    return reward >= min_reward

def extract_result_numbers(result: dict):
    """
    Extract evaluation metrics from eval result dict.

    Returns:
        joint_score, gm, content, style, fluency (floats)
    """
    if not isinstance(result, dict):
        raise TypeError(f"Expected dict, got {type(result).__name__}")

    return tuple(result.get(metric, 0.0) for metric in METRICS)

def aggregate_style_transfer_results(
    eval_results: List[Dict[str, float]],
) -> Dict[str, float]:
    assert len(eval_results) == 2, "Exactly two results are required for aggregation"

    r1, r2 = eval_results

    if not isinstance(r1, dict) or not isinstance(r2, dict):
        raise TypeError("Aggregation expects dict-based eval results")

    aggregated = {}

    for metric in METRICS:
        v1 = r1.get(metric, 0.0)
        v2 = r2.get(metric, 0.0)
        aggregated[metric] = (v1 + v2) / 2.0

    return aggregated

class StyleTransferPromptCandidate(dict):
    def __init__(self, 
                 prompt: str = "",
                 mask: List[bool] = None,
                 reward: float = 0.0,
                 length: int = 0,
                 gm: float = 0.0,
                 content: float = 0.0,
                 style: float = 0.0,
                 fluency: float = 0.0,
                 **kwargs):
        super().__init__(
            prompt=prompt,
            mask=copy.deepcopy(mask) if mask else [],
            reward=reward,
            length=length,
            gm=gm,
            content=content,
            style=style,
            fluency=fluency,
            **kwargs,
        )
    
    @classmethod
    def from_evaluation(cls,
                       prompt: str,
                       mask: List[bool],
                       eval_result: Any,
                       tokenizer: Any = None):
        """Create PromptCandidate from Direct Evaluation Results."""
        # Support multiple tester output formats
        if isinstance(eval_result, dict):
            # Dict form: {'reward': 0.90, ...}
            reward = eval_result.get('joint_score', 0.0)
            extra = {k: v for k, v in eval_result.items()
                    if k not in ['joint_score']}
        else:
            raise ValueError(f"Unsupported eval_result: {type(eval_result)}")
        
        # Calculate prompt length
        length = len(tokenizer.tokenize(prompt)) if tokenizer else len(prompt.split())
        
        return cls(
            prompt=prompt,
            mask=mask,
            reward=reward,
            length=length,
            **extra
        )
    
    def get_fitness(self, reward_driven: bool = False) -> float:
        return self.get('reward', 0.0)
    
    def to_list_copy(self) -> tuple:
        """
        Convert the candidate object to a list representation.

        Returns (Only Selective):
            list: [prompt, reward, length, mask_copy]
                - prompt (str)
                - reward (float)
                - length (int)
                - mask_copy (list[bool]): deep copy of mask
        """
        return [
            self.get('prompt'),
            self.get('reward'),
            self.get('length'),
            copy.deepcopy(self.get('mask')),
            self.get('gm'),
            self.get('content'),
            self.get('style'),
            self.get('fluency'),
        ]


def save_pruned_prompts(prompt_queues: List[Tuple], output_path: str) -> None:
    """Save pruned prompts to CSV file"""
    processed_data = [
        (   prompt,
            joint.item() if hasattr(joint, 'item') else joint,
            gm, content, style, fluency, length, mask
        )
        for (prompt, joint, length, mask, gm, content,
            style, fluency) in prompt_queues
    ]
    df = pd.DataFrame(prompt_queues, columns=['prompt', 'joint', 'gm', 
                                        'content', 'style', 'fluency', '#tokens', "mask"])
    df = df.drop(columns=["mask"])
    df.to_csv(output_path, index=False)