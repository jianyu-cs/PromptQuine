"""Utilities for prompt manipulation."""
import json
import copy
from typing import List, Dict, Tuple, Any, Union

import pandas as pd
from fastchat.model import get_conversation_template


fields = ["prompt", "reward", "length", "mask", "EM_score", "Guard_score", "SV_score"]
KEY_TO_INDEX = {k: i for i, k in enumerate(fields)}

# In-context Attack Only
def resolve_chat_template(model_name: str) -> str:
    if "vicuna" in model_name.lower():
        return "vicuna_1.5"
    if "llama-2" in model_name.lower():
        return "llama-2"
    raise ValueError(f"Unknown model name: {model_name}")

def format_chat_prompt(
    user_prompt: str,
    demo_inputs: List[str],
    demo_outputs: List[str],
    model_name: str,
    assistant_placeholder: bool = True,
) -> str:
    assert len(demo_inputs) == len(demo_outputs), "The number of demonstration inputs must equal the number of demonstration outputs."
    template_name = resolve_chat_template(model_name)
    conv = get_conversation_template(template_name)
    for i, demo_input in enumerate(demo_inputs):
        conv.append_message(conv.roles[0], demo_input)
        conv.append_message(conv.roles[1], demo_output)
    conv.append_message(conv.roles[0], user_prompt)
    if assistant_placeholder:
        conv.append_message(conv.roles[1], None)

    return conv.get_prompt()

# Others
def get_jailbreaking_field(lst: list, key: str, default=None, reward_driven=None):
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

def should_be_evaluated_next_round_for_jailbreaking(eval_result: Any, min_reward: float):
    if isinstance(eval_result, dict):
        # Dict form: {'reward': 0.90, ...}
        reward = eval_result.get('reward', 0.0)
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
        reward (floats), EM_score (int), Guard_score (int)
    """
    if not isinstance(result, dict):
        raise TypeError(f"Expected dict, got {type(result).__name__}")

    return result.get("reward", 0.0), result.get("EM_score", 0.0), result.get("Guard_score", 0.0)

def aggregate_jailbreaking_results(
    eval_results: List[Dict[str, float]],
) -> Dict[str, float]:
    assert len(eval_results) == 2, "Exactly two results are required for aggregation"

    r1, r2 = eval_results

    if not isinstance(r1, dict) or not isinstance(r2, dict):
        raise TypeError("Aggregation expects dict-based eval results")

    aggregated = {}

    for metric in ['reward', 'EM_score', 'Guard_score', 'SV_score']:
        v1 = r1.get(metric, 0.0)
        v2 = r2.get(metric, 0.0)
        aggregated[metric] = (v1 + v2) / 2.0

    return aggregated

class JailbreakingPromptCandidate(dict):
    def __init__(self, 
                 prompt: str = "",
                 mask: List[bool] = None,
                 reward: float = 0.0,
                 length: int = 0,
                 # ASR Scores
                 EM_score: Union[int, float] = 0,
                 Guard_score: Union[int, float] = 0,
                 SV_score: Union[int, float] = -1,
                 **kwargs):
        super().__init__(
            prompt=prompt,
            mask=copy.deepcopy(mask) if mask else [],
            reward=reward,
            length=length,
            EM_score=EM_score,
            Guard_score=Guard_score,
            SV_score=SV_score
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
            reward = eval_result.get('reward', 0.0)
            EM_score = eval_result.get('EM_score', 0.0)
            Guard_score = eval_result.get('Guard_score', 0.0)
            SV_score = eval_result.get('SV_score', 0.0)
            extra = {k: v for k, v in eval_result.items()
                    if k not in ['reward']}
        else:
            raise ValueError(f"Unsupported eval_result: {type(eval_result)}")
        
        # Calculate prompt length
        length = len(tokenizer.tokenize(prompt)) if tokenizer else len(prompt.split())
        
        return cls(
            prompt=prompt,
            mask=mask,
            reward=reward,
            length=length,
            EM_score=EM_score,
            Guard_score=Guard_score,
            SV_score=SV_score 
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
                - EM_score (float or int),
                - Guard_score (float or int),
                - SV_score (float or int)
        """
        return [
            self.get('prompt'),
            self.get('reward'),
            self.get('length'),
            copy.deepcopy(self.get('mask')),
            self.get('EM_score'),
            self.get('Guard_score'),
            self.get('SV_score')
        ]

def save_pruned_prompts(prompt_queues: List[Tuple], output_path: str) -> None:
    """Save pruned prompts to CSV file"""
    processed_data = [
        (prompt, reward, length, mask, EM_score, Guard_score, SV_score)
        for (prompt, reward, length, mask, EM_score, Guard_score, SV_score) in prompt_queues
    ]
    df = pd.DataFrame(
        prompt_queues,
        columns=[
            'prompt',
            'ASR-Fitness',
            '#tokens',
            'mask',
            'EM_score',
            'Guard_score',
            'SV_score'
        ]
    )
    df = df.drop(columns=["mask"])
    df.to_csv(output_path, index=False)