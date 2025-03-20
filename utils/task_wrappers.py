"""
Miscellaneous Utility Functions
@ RLPrompt (EMNLP 2022): https://github.com/mingkaid/rl-prompt/tree/main
"""
import click
import warnings
from typing import Dict, Any, Optional, List
import dataclasses
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig


def get_hydra_output_dir():
    return HydraConfig.get().run.dir


def compose_hydra_config_store(
    name: str, 
    configs: List[dataclass]
) -> ConfigStore:
    config_fields = []
    for config_cls in configs:
        for config_field in dataclasses.fields(config_cls):
            config_fields.append((config_field.name, config_field.type,
                                  config_field))
    Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
    cs = ConfigStore.instance()
    cs.store(name=name, node=Config)
    return cs


def add_prefix_to_dict_keys_inplace(
        d: Dict[str, Any],
        prefix: str,
        keys_to_exclude: Optional[List[str]] = None,
) -> None:
    # https://stackoverflow.com/questions/4406501/change-the-name-of-a-key-in-dictionary
    keys = list(d.keys())
    for key in keys:
        if keys_to_exclude is not None and key in keys_to_exclude:
            continue

        new_key = f"{prefix}{key}"
        d[new_key] = d.pop(key)

        
def colorful_print(string: str, *args, **kwargs) -> None:
    print(click.style(string, *args, **kwargs))

    
def colorful_warning(string: str, *args, **kwargs) -> None:
    warnings.warn(click.style(string, *args, **kwargs))

    
def unionize_dicts(dicts: List[Dict]) -> Dict:
    union_dict: Dict = {}
    for d in dicts:
        for k, v in d.items():
            if k in union_dict.keys():
                raise KeyError
            union_dict[k] = v

    return union_dict


class PromptedTaskWrapperBase:
    def __init__(self):
        self.task_type = None
    def Task_evaluation_forward(self):
        pass
    
    
def create_tabulist(tokenizer, prompt_tokens):
    tabu_list = []
    for i, token in enumerate(prompt_tokens):
        if i in tabu_list:
            continue
        if token in [":{", "{", "Ġ{", " {", '▁{']:
            if '{sentence_1}' in ''.join(prompt_tokens[i:i+len(tokenizer.tokenize(" {sentence_1}"))]):
                for j in range(i, i+len(tokenizer.tokenize(" {sentence_1}"))):
                    tabu_list.append(j)
            elif '{sentence_2}' in ''.join(prompt_tokens[i:i+len(tokenizer.tokenize(" {sentence_2}"))]):
                for j in range(i, i+len(tokenizer.tokenize(" {sentence_2}"))):
                    tabu_list.append(j)
            elif '{question_1}' in ''.join(prompt_tokens[i:i+len(tokenizer.tokenize(" {question_1}"))]):
                for j in range(i, i+len(tokenizer.tokenize(" {question_1}"))):
                    tabu_list.append(j)
            elif '{option_1}' in ''.join(prompt_tokens[i:i+len(tokenizer.tokenize(" {option_1}"))]):
                for j in range(i, i+len(tokenizer.tokenize(" {option_1}"))):
                    tabu_list.append(j)
            elif '{option_2}' in ''.join(prompt_tokens[i:i+len(tokenizer.tokenize(" {option_2}"))]):
                for j in range(i, i+len(tokenizer.tokenize(" {option_1}"))):
                    tabu_list.append(j)
            elif '{mask_token}' in ''.join(prompt_tokens[i:i+len(tokenizer.tokenize("{mask_token}"))]):
                for j in range(i, i+len(tokenizer.tokenize("{mask_token}"))):
                    tabu_list.append(j)
    return tabu_list