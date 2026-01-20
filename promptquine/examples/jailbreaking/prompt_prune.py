"""Prompt pruning scripts for jailbreaking.

This module implements pruning script for ICL prompts (jailbreaking tasks).
Specifically, it includes two modes:

1. Few-shot priming: Just like traditional ICL, we do not explicitly add conversational tags among demonstrations;
2. In-context attacks: We explicitly separate demonstrations into different conversation turns with conversational tags asserted;
"""
import sys
import os
import json
import copy
import datetime
import argparse

import ray
import hydra
import pandas as pd
from itertools import compress
from torch.utils.data import DataLoader
from hydra.core.config_store import ConfigStore
from transformers import AutoTokenizer, LlamaTokenizer
from typing import Optional, Union, List, Dict, Any

from promptquine.utils.jailbreaking import (
    get_prompts_path,
    get_output_path,
    format_chat_prompt,
    save_pruned_prompts
)
from pruner_config import Config
from promptquine.modules.TAPruner import TAPruner
from promptquine.utils import measure_time, colorful_print, load_prompts
from jailbreaking_evaluator import PromptedJailbreakingEvaluator
from promptquine.modules.PromptQuinePruner import PromptQuinePruner
from dataset_helper import load_jailbreaking_dataset


def create_pruner(cfg):
    """Create pruner instance based on type"""
    if cfg.pruning.algorithm == "PromptQuine":
        return PromptQuinePruner(
            cfg.model.name,
            "jailbreaking", 
            None, 
            "vLLM",
            initialize_duplicate=cfg.prompt_quine.initialize_duplicate,
            min_prompt_length=cfg.prompt_quine.min_prompt_length,
            max_prompts_in_replication=cfg.prompt_quine.max_num_prompts,
            genetic_algorithm_mode=cfg.prompt_quine.algorithm_mode,
            population_size=cfg.prompt_quine.population_size,
            reproduction_size=cfg.prompt_quine.reproduction_size,
            num_devices=cfg.model.num_devices,
            dataset=cfg.data.dataset,
            is_mask_lm=False,
            pruning_metric=cfg.pruning.pruning_metric,
            priming=cfg.model.priming,
            gpus_per_bundle=cfg.model.gpus_per_bundle,
            cpus_per_bundle=cfg.model.cpus_per_bundle
        )

    elif cfg.pruning.algorithm == "TAPruning":
        return TAPruner(
            cfg.model.name,
            "jailbreaking",
            None,
            'vLLM',
            threshold=cfg.pruning.TAPruning_threshold,
            num_devices=cfg.model.num_devices,
            dataset=cfg.data.dataset,
            is_mask_lm=False,
            pruning_metric=cfg.pruning.pruning_metric,
            priming=cfg.model.priming,
            gpus_per_bundle=cfg.model.gpus_per_bundle,
            cpus_per_bundle=cfg.model.cpus_per_bundle
        )
    
    raise ValueError(f"Unknown pruner type: {pruner_type}")

@measure_time
@hydra.main(version_base=None, config_path=".", config_name="pruner_config")
def main(cfg: Config):
    # initialize ray
    ray.init(local_mode=True)
    # 1. load dataset
    base_path = "./data"
    source_instances = load_jailbreaking_dataset(
        cfg.data.dataset,
        "dev", # Full dev samples 
        base_path
    )
    source_input_queries = source_instances["prompt"].tolist()[:50]
    
    # 2. Load ICL Prompts for Pruning
    prompts_path = get_prompts_path(cfg.data.dataset, cfg.model.ICL_shots)
    prompt_dict_list = load_prompts(prompts_path)
    # Structure: {
    #     'prompt': str,          # Template with placeholders
    #     'inputs': List[str],    # Input examples
    #     'outputs': List[str]    # Corresponding outputs
    #     ...
    # }
    prompt_dict = prompt_dict_list[cfg.model.ICL_index] 
    # 3. Setup the Pruner
    pruner = create_pruner(cfg)
    # 4. Perform Pruning
    """
    Structure of `prompt_queues`:

    A list of tuples, where each tuple contains the following elements:
        (prompt, metric_result, prompt_length, mask)
    """
    prompt_queues, num_iterations = pruner.forward(
        prompt=prompt_dict,
        test_loader=source_input_queries,
        reward_driven=False,
        fix_prune_order=cfg.pruning.fix_prune_order,
        priming=cfg.model.priming
    )
    # 5. Release ray
    ray.shutdown()
    # 6. Save the collection of prompts
    model_name = cfg.model.name.split("/")[1] if "/" in cfg.model.name else cfg.model.name
    output_path = get_output_path(cfg)
    # Refactor
    save_pruned_prompts(prompt_queues, output_path)
    colorful_print(f"The pruning results have been saved into: {output_path}", "green")

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="pruner_config", node=Config)
    
    main()