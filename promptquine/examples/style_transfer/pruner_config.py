# pruner_config.py
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    dataset: str = "yelp"
    dataset_seed: Optional[int] = None
    direction: str = "1_to_0" 
    base_path: str = './data'
    max_size: Optional[int] = None
    max_length: Optional[int] = None
    max_length_tokenizer: Optional[str] = None

@dataclass
class ModelConfig:
    name: str = "openai-community/gpt2"
    # Decoding parameters
    end_punct: str = '"'
    task_top_k: int = 10 
    num_samples: int = 32
    pad_token: str = '<|endoftext|>'
    compute_zscore: bool = True  # Whether to compute z-score of rewards
    lower_outputs: bool = False  # Whether to convert all outputs to lower case
    control_output_length: bool = False
    # Style classifier parameters (HF-Only)
    style_classifier: str = 'test'
    style_tokenizer: Optional[str] = None
    style_batch_size: int = 32
    style_classifier_device_id: int = 1
    # ICL prompts parameters
    ICL_shots: int = 1
    ICL_index: int = 0
    num_devices: int = 1
    inference_engine: str = "vLLM"

@dataclass
class PruningConfig:
    algorithm: str = "TAPruning"
    fix_prune_order: bool = True
    pruning_order_seed: int = 0
    TAPruning_threshold: float = 0.96
    reward_driven: bool = False

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
