"""Prompt pruning scripts for chain-of-thought reasoning (CoT).

This module implements pruning script for ICL prompts (CoT reasoning tasks).
"""
import sys
import os
import json
import copy
import datetime
import argparse

import hydra
import pandas as pd
from itertools import compress
from torch.utils.data import DataLoader
from hydra.core.config_store import ConfigStore
from transformers import AutoTokenizer, LlamaTokenizer
from typing import Optional, Union, List, Dict, Any

from promptquine.utils.reasoning import (
    get_prompts_path,
    get_output_path,
    save_pruned_prompts
)
from pruner_config import Config
from promptquine.modules.TAPruner import TAPruner
from promptquine.utils import measure_time, colorful_print, load_prompts
from promptquine.modules.PromptQuinePruner import PromptQuinePruner
from dataset_helper import load_reasoning_dataset


def create_pruner(cfg):
    """Create pruner instance based on type"""
    if cfg.pruning.algorithm == "PromptQuine":
        return PromptQuinePruner(
            cfg.model.name,
            "reasoning", 
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
        )

    elif cfg.pruning.algorithm == "TAPruning":
        return TAPruner(
            cfg.model.name,
            "reasoning",
            None,
            'vLLM',
            threshold=cfg.pruning.TAPruning_threshold,
            num_devices=cfg.model.num_devices,
            dataset=cfg.data.dataset,
            is_mask_lm=False,
        )
    
    raise ValueError(f"Unknown pruner type: {pruner_type}")

@measure_time
@hydra.main(version_base=None, config_path=".", config_name="pruner_config")
def main(cfg: Config):
    # 1. load dataset
    source_texts = load_reasoning_dataset(
        cfg.data.dataset,
        "train",
        cfg.data.max_size, # how many in-search samples
    )
    
    # 2. Load ICL Prompts for Pruning
    prompts_path = get_prompts_path(cfg.data.dataset, cfg.model.ICL_shots)
    prompt_dict_list = load_prompts(prompts_path)
    prompt = prompt_dict_list[cfg.model.ICL_index]['prompt']
    
    # 3. Setup the Pruner
    pruner = create_pruner(cfg)

    # 4. Perform Pruning
    """
    Structure of `prompt_queues`:

    A list of tuples, where each tuple contains the following elements:
        (prompt, accuracy, prompt_length, mask)
    """
    prompt_queues, num_iterations = pruner.forward(
        prompt=prompt,
        test_loader=source_texts,
        reward_driven=False,
        fix_prune_order=cfg.pruning.fix_prune_order,
        successive_halving=cfg.prompt_quine.successive_halving
    )

    # 5. Save the collection of prompts
    model_name = cfg.model.name.split("/")[1] if "/" in cfg.model.name else cfg.model.name
    output_path = get_output_path(cfg)

    save_pruned_prompts(prompt_queues, output_path)
    colorful_print(f"The pruning results have been saved into: {output_path}", "green")

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="pruner_config", node=Config)
    
    main()