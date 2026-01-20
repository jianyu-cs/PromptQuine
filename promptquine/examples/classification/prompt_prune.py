import sys
import os
import json
import copy
import datetime
from typing import Optional, Union, List, Dict, Any
from itertools import compress

import pandas as pd
import torch
import hydra
from torch.utils.data import DataLoader
from hydra.core.config_store import ConfigStore
from transformers import AutoTokenizer, LlamaTokenizer

from promptquine.utils.classification import (
    get_prompts_path,
    get_output_path,
    save_pruned_prompts,
    is_masked_language_model
)
from pruner_config import Config
from promptquine.modules.TAPruner import TAPruner
from promptquine.utils import measure_time, colorful_print, load_prompts
from promptquine.modules.PromptQuinePruner import PromptQuinePruner
from dataset_helper import make_classification_dataset
from fsc_evaluator import PromptedClassificationEvaluator


def create_pruner(cfg, is_mask_lm: bool):
    """Create pruner instance based on type"""
    if cfg.pruning.algorithm == "PromptQuine":
        return PromptQuinePruner(
            cfg.model.name,
            "classification", 
            None, 
            cfg.model.inference_engine,
            initialize_duplicate=cfg.prompt_quine.initialize_duplicate,
            min_prompt_length=cfg.prompt_quine.min_prompt_length,
            max_prompts_in_replication=cfg.prompt_quine.max_num_prompts,
            genetic_algorithm_mode=cfg.prompt_quine.algorithm_mode,
            population_size=cfg.prompt_quine.population_size,
            reproduction_size=cfg.prompt_quine.reproduction_size,
            num_devices=cfg.model.num_devices,
            dataset=cfg.data.dataset,
            is_mask_lm=is_mask_lm
        )
    
    elif cfg.pruning.algorithm == "TAPruning":
        return TAPruner(
            cfg.model.name,
            "classification",
            None,
            cfg.model.inference_engine,
            threshold=cfg.pruning.TAPruning_threshold,
            num_devices=cfg.model.num_devices,
            dataset=cfg.data.dataset,
            is_mask_lm=is_mask_lm
        )
    
    raise ValueError(f"Unknown pruner type: {pruner_type}")

@measure_time
@hydra.main(version_base=None, config_path=".", config_name="pruner_config")
def main(cfg: Config):
    base_path = "./data"
    
    # 1. Load datasets
    is_mask_lm = is_masked_language_model(cfg.model.name)
    
    if cfg.pruning.algorithm == "PromptQuine":
        valid_dataset = make_classification_dataset(
            cfg.data.dataset, cfg.data.dataset_seed, base_path, cfg.model.name, True,
            cfg.data.num_shots, cfg.data.split, cfg.data.split_seed
        )
    else:
        valid_dataset = make_classification_dataset(
            cfg.data.dataset, cfg.data.dataset_seed, base_path, cfg.model.name, False,
        )
    
    valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=512,
        num_workers=4,
        drop_last=False
    )
    
    # 2. Load prompts
    prompts_path = get_prompts_path(cfg.data.dataset, is_mask_lm, cfg.model.ICL_shots, cfg.data.is_random_verbalizers)
    prompt_dict_list = load_prompts(prompts_path)
    prompt = prompt_dict_list[cfg.model.ICL_index]['prompt']
    
    # 3. Create Pruner
    pruner = create_pruner(cfg, is_mask_lm=is_mask_lm)
    
    # 4. Conduct pruning
    prompt_queues, num_iterations = pruner.forward(
        prompt=prompt,
        test_loader=valid_loader,
        reward_driven=cfg.pruning.reward_driven,
        fix_prune_order=cfg.pruning.fix_prune_order
    )
    
    # 5. Save results
    model_name = cfg.model.name.split("/")[1] if "/" in cfg.model.name else cfg.model.name
    output_path = get_output_path(cfg)
    
    save_pruned_prompts(prompt_queues, output_path)
    colorful_print(f"The pruning results have been saved into: {output_path}", "green")

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="pruner_config", node=Config)
    
    main()