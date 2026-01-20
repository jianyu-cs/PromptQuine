"""Utilities for prompt manipulation."""
import json
import copy
from typing import List, Dict, Tuple, Any

import pandas as pd


fields = ["prompt", "accuracy", "reward", "length", "mask"]
KEY_TO_INDEX = {k: i for i, k in enumerate(fields)}

VERBALIZERS = {
    'sst-2': ['terrible', 'great'],
    'subj': ['subjective', 'objective'],
    'agnews': ['World', 'Sports', 'Business', 'Tech'],
    'yelp-5': ['terrible', 'bad', 'neutral', 'good', 'great'],
    'yahoo': ['culture', 'science', 'health', 'education', 'computer', 
              'sports', 'business', 'music', 'family', 'politics'],
    'snli': ['Yes', 'Unknown', 'No'],
    'piqa': ['A', 'B'],
}

def get_tokenizer_prefix(task_lm: str) -> str:
    """Get the token prefix for the given model."""
    if 'gemma' in task_lm:
        return '▁'
    elif any(model in task_lm for model in ['roberta', 'gpt2', 'llama']):
        return 'Ġ'
    else:
        raise ValueError(f"Unknown model tokenizer: {task_lm}")

def load_verbalizers(task_lm: str, dataset: str) -> List[str]:
    """Load verbalizers for given model and dataset."""
    if dataset not in VERBALIZERS:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    prefix = get_tokenizer_prefix(task_lm)
    return [prefix + word for word in VERBALIZERS[dataset]]

def get_classification_field(lst: list, key: str, default=None, reward_driven=None):
    """
    Retrieve the value from a list corresponding to the given key.
    
    lst: the list object
    key: the field name, e.g., 'prompt', 'accuracy', etc.
    default: the value to return if the key does not exist or the index is out of range
    reward_driven: bool-> if True => reward, else => accuracy
    """
    if reward_driven is not None:
        key = "reward" if reward_driven else "accuracy"
        return lst[KEY_TO_INDEX[key]]

    idx = KEY_TO_INDEX.get(key)
    if idx is None or idx >= len(lst):
        return default
    return lst[idx]

def should_be_evaluated_next_round_for_classification(
        eval_result: Any, 
        min_reward: float,
        reward_driven: bool
    ):
    if isinstance(eval_result, dict):
        # Dict form: {'accuracy': 0.85, 'reward': 0.90, ...}
        reward = (
            eval_result.get('reward', 0.0) if reward_driven 
            else eval_result.get('accuracy', 0.0)
        )
    elif isinstance(eval_result, tuple) and len(eval_result) >= 2:
        # Tuple form: (accuracy, reward)
        reward = (
            float(eval_result[1]) if reward_driven
            else float(eval_result[0])
        )
    else:
        raise ValueError(f"Unsupported eval_result: {type(eval_result)}")
    return reward >= min_reward

def extract_result_numbers(result):
    """Extract accuracy and reward from dict or tuple."""
    if isinstance(result, dict):
        return result.get('accuracy', 0.0), result.get('reward', 0.0)
    elif isinstance(result, tuple) and len(result) >= 2:
        return float(result[0]), float(result[1])
    else:
        raise ValueError(f"Unsupported eval_result type: {type(result)}")

def aggregate_classification_results(
        eval_results: List[Any],
    ) -> Any:
    assert len(eval_results) == 2, f"Two results only for aggregation!"
    eval_result, eval_result_ = eval_results
    assert type(eval_result) == type(eval_result_), "These results shall be at least of the same type."
    accuracy, reward = extract_result_numbers(eval_result)
    accuracy_, reward_ = extract_result_numbers(eval_result_)
    agg_accuracy = (accuracy + accuracy_) / 2.0
    agg_reward = (reward + reward_) / 2.0
    # Construct the new result
    if isinstance(eval_result, dict):
        eval_result_agg = eval_result.copy()
        eval_result_agg['reward'] = agg_reward
        eval_result_agg['accuracy'] = agg_accuracy
    elif isinstance(eval_result, tuple):
        eval_result_agg = (agg_accuracy, agg_reward) 

    return eval_result_agg

def save_pruned_prompts(prompt_queues: List[Tuple], output_path: str) -> None:
    """Save pruned prompts to CSV file"""
    processed_data = [(p, acc, r, l) for p, acc, r, l, m in prompt_queues]
    df = pd.DataFrame(processed_data, columns=['prompt', 'acc', 'reward', '#tokens'])
    df.to_csv(output_path, index=False)

class ClassificationPromptCandidate(dict):
    def __init__(self, 
                 prompt: str = "",
                 mask: List[bool] = None,
                 reward: float = 0.0,
                 length: int = 0,
                 **kwargs):
        super().__init__(
            prompt=prompt,
            mask=copy.deepcopy(mask) if mask else [],
            accuracy=kwargs.get("accuracy", 0),
            reward=reward,
            length=length,
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
            # Dict form: {'accuracy': 0.85, 'reward': 0.90, ...}
            accuracy = eval_result.get('accuracy', 0.0)
            reward = eval_result.get('reward', 0.0)
            extra = {k: v for k, v in eval_result.items() 
                    if k not in ['accuracy', 'reward']}
        elif isinstance(eval_result, tuple) and len(eval_result) >= 2:
            # Tuple form: (accuracy, reward)
            accuracy = float(eval_result[0])
            reward = float(eval_result[1])
            extra = {}
        else:
            raise ValueError(f"Unsupported eval_result: {type(eval_result)}")
        
        # Calculate prompt length
        length = len(tokenizer.tokenize(prompt)) if tokenizer else len(prompt.split())
        
        return cls(
            prompt=prompt,
            mask=mask,
            accuracy=accuracy,
            reward=reward,
            length=length,
            **extra
        )
    
    def get_fitness(self, reward_driven: bool = False) -> float:
        return self.get('reward', 0.0) if reward_driven else self.get('accuracy', 0.0)

    def to_list_copy(self) -> tuple:
        """
        Convert the candidate object to a list representation.

        Returns (Only Selective):
            list: [prompt, accuracy, reward, length, mask_copy]
                - prompt (str)
                - accuracy (float)
                - reward (float)
                - length (int)
                - mask_copy (list[bool]): deep copy of mask
        """
        return [
            self.get('prompt'),
            self.get('accuracy'),
            self.get('reward'),
            self.get('length'),
            copy.deepcopy(self.get('mask'))
        ]
