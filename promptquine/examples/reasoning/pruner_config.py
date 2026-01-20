# pruner_config.py
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    dataset: str = "gsm8k"
    dataset_seed: Optional[int] = None
    base_path: str = './data'
    max_size: Optional[int] = None

# Greedy Decoding
@dataclass
class ModelConfig:
    name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    # ICL prompts parameters
    ICL_shots: int = 1
    ICL_index: int = 0
    num_devices: int = 1

@dataclass
class PruningConfig:
    algorithm: str = "TAPruning"
    fix_prune_order: bool = True
    pruning_order_seed: int = 0
    TAPruning_threshold: float = 0.96

@dataclass
class PromptConfig: # Evaluation-Only
    prompt: str = "" # '{prompt} "{sentence_1}" "'
    is_pruned_prompt: bool = False

@dataclass
class PromptQuineConfig:
    initialize_duplicate: bool = False
    min_prompt_length: int = 15
    max_num_prompts: int = 10000
    algorithm_mode: str = "SSGA"
    population_size: int = 30
    reproduction_size: int = 50
    # Evaluation-only: re-rank
    top_percent_rerank: int = 10
    test_all_elites_for_debug: bool = False
    # Resource Allocation
    successive_halving: bool = True

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    prompt_quine: PromptQuineConfig = field(default_factory=PromptQuineConfig)
