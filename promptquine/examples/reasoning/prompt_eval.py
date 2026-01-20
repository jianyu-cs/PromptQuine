import argparse
import dataclasses
import json
import os
import sys
import pdb
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import hydra
import torch
import numpy as np
import pandas as pd
from vllm import LLM
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore

from pruner_config import Config
from dataset_helper import load_reasoning_dataset
from reasoning_evaluator import PromptedReasoningEvaluator
from promptquine.utils import measure_time, colorful_print, load_prompts
from promptquine.utils.reasoning import (
    get_prompts_path,
    get_output_path
)

def evaluate_prompts(prompts, prompted_generator,
                     source_texts):
    """Evaluate prompts on given dataset and return metrics."""
    metrics_lists = {key: [] for key in ['accuracy', 'reward']}

    for prompt in prompts:
        prompted_generator.template = prompt
        print("prompt:", prompted_generator.template)

        result = prompted_generator.forward(
            data = source_texts,
            prompt = prompt
        )

        colorful_print("Aggregate Scores:", "cyan")
        for k, v in result.items():
            colorful_print(f"{k}: {v}", "yellow")

        for key in metrics_lists:
            metrics_lists[key].append(float(result[key]))

    return metrics_lists

@measure_time
@hydra.main(version_base=None, config_path=".", config_name="pruner_config")
def main(cfg):
    # 1. load dataset
    # dev (max_size) & test
    source_texts = load_reasoning_dataset(
        cfg.data.dataset,
        "dev",
        200, # how many dev samples
    )

    test_source_texts = load_reasoning_dataset(
        cfg.data.dataset,
        "test", # Full test samples
    )
    # 2. Load Prompts
    unpruned_prompt = None
    # Loading Prompt in the command line
    if cfg.prompt.prompt:
        prompts = [cfg.prompt.prompt]
        output_content = {
            "prompt": prompts,
        }
    else:
        # Loading Prompts (UnPruned)
        if not cfg.prompt.is_pruned_prompt:
            prompts_path = get_prompts_path(cfg.data.dataset, cfg.model.ICL_shots)
            prompt_dict_list = load_prompts(prompts_path)

            if cfg.model.ICL_index != None:
                prompts = [prompt_dict_list[cfg.model.ICL_index]['prompt']]
            else:
                prompts = [prompts_dict['prompt'] for prompts_dict in prompt_dict_list]
            
            output_content = {
                "prompt": prompts,
            }
        # Loading Prompts (Pruned)
        else:
            prompts_path = get_output_path(cfg)
            metric = "acc" # unless otherwise stated
            if cfg.pruning.algorithm == "PromptQuine":
                # Re-ranking for PromptQuine
                prompts_df = pd.read_csv(prompts_path)
                prompt_percent_threshold = np.percentile(prompts_df[metric], 100 - cfg.prompt_quine.top_percent_rerank)
                top_df = prompts_df[prompts_df[metric] >= prompt_percent_threshold].copy()
                top_df['rank'] = top_df[metric].rank(ascending=False, method='min')
                top_df = top_df.sort_values(by='rank')
                prompts = top_df['prompt'].tolist()
                output_content = {
                    "prompt": prompts,
                }
            elif cfg.pruning.algorithm == "TAPruning":
                # Pick the one with the highest validation score during pruning
                prompts_df = pd.read_csv(prompts_path)
                max_joint = prompts_df[metric].max()
                last_max_row = prompts_df[prompts_df[metric] == max_joint].index[-1]

                prompts, val_acc, val_num_tokens = [
                    [prompts_df.iloc[last_max_row][col]] 
                    for col in ['prompt', 'acc', '#tokens']
                ]
                output_content = {
                    "prompt": prompts,
                    "val_acc": val_acc,
                    "num_tokens": val_num_tokens
                }
    # 3. Setup the evaluation
    print('Test Size:', len(test_source_texts))
    print('Examples:', test_source_texts[:5])
    prompted_generator = PromptedReasoningEvaluator(
        task_lm = cfg.model.name,
        dataset = cfg.data.dataset,
        prompt = cfg.prompt.prompt,
        num_devices = cfg.model.num_devices,
    )
    # 4. Start evaluation
    test_metrics_results = evaluate_prompts(
        prompts, prompted_generator,
        test_source_texts
    )
    if cfg.pruning.algorithm == "PromptQuine":
        dev_metrics_results = evaluate_prompts(
            prompts, prompted_generator,
            source_texts
        )
    # 5. Prepare prompt file
    if cfg.prompt.prompt:
        # Save into Temporary Directory
        base_saved_dir = "Prompt_cache"
        os.makedirs(base_saved_dir, exist_ok=True)

        prompt_saved_file_path = os.path.join(
            base_saved_dir,
            "prompt_cot_reasoning.csv"
        )
    elif not cfg.prompt.is_pruned_prompt:
        # Save into Temporary Directory
        base_saved_dir = os.path.join(
            "ICLPrompts_Unpruned",
            cfg.data.dataset,
            cfg.model.name.replace('/', '-'),
            f"{cfg.model.ICL_shots}-shots_ICL",
        )
        os.makedirs(base_saved_dir, exist_ok=True)

        prompt_saved_file_path = os.path.join(
            base_saved_dir,
            f"{cfg.model.ICL_index}-index-ICL.csv"
        )
    elif cfg.pruning.algorithm == "PromptQuine":
        # Create directory otherwise noted
        if cfg.prompt_quine.test_all_elites_for_debug:
            suffix = f"Test_all_top{cfg.prompt_quine.top_percent_rerank}_percent"
        else:
            suffix = "Ranked_top1"

        base_saved_dir = os.path.join(
            "PrunedPrompts_by_PromptQuine",
            cfg.prompt_quine.algorithm_mode,
            cfg.model.name.replace('/', '-'),
            f"{cfg.model.ICL_shots}-shots_ICL",
            cfg.data.dataset,
            "Eval",
            suffix,
        )
        os.makedirs(base_saved_dir, exist_ok=True)
        successive_halving_str = "halving" if cfg.prompt_quine.successive_halving else "no_halving"
        if cfg.prompt_quine.initialize_duplicate:
            filename = (
                f"{successive_halving_str}_initialize_with_duplicates"
                f"_{cfg.data.max_size}-samples-in-train"
                f"_{cfg.prompt_quine.population_size}-population-size"
                f"_{cfg.prompt_quine.reproduction_size}-reproduction-size"
                f"_{cfg.model.ICL_index}-index-ICL.csv"
            )
        else:
            filename = (
                f"{successive_halving_str}_initialize_with_random_pruning"
                f"_{cfg.data.max_size}-samples-in-train"
                f"_{cfg.prompt_quine.population_size}-population-size"
                f"_{cfg.prompt_quine.reproduction_size}-reproduction-size"
                f"_{cfg.model.ICL_index}-index-ICL.csv"
            )
        prompt_saved_file_path = os.path.join(base_saved_dir, filename)
    elif cfg.pruning.algorithm == "TAPruning":
        # Create directory otherwise noted
        local_dir_name = "Ranked_top1"
        base_saved_dir = os.path.join(
            "PrunedPrompts_by_TAPruning",
            cfg.model.name.replace('/', '-'),
            f"{cfg.model.ICL_shots}-shots_ICL",
            cfg.data.dataset,
            "Eval",
            local_dir_name
        )
        os.makedirs(base_saved_dir, exist_ok=True)

        if cfg.pruning.fix_prune_order:
            filename = (
                f"200-samples_{cfg.pruning.TAPruning_threshold}-threshold"
                f"_fixed-prune-order"
                f"_{cfg.model.ICL_index}-index-ICL.csv"
            )
        else:
            filename = (
                f"200-samples_{cfg.pruning.TAPruning_threshold}-threshold"
                f"_shuffled-prune-order"
                f"_{cfg.model.ICL_index}-index-ICL.csv"
            )
        prompt_saved_file_path = os.path.join(base_saved_dir, filename)
    # 6. Save evaluation results
    # === Prepare DataFrame ===
    # + Parse test_metrics_results and dev_metrics_results (Conditional) 
    # - Only applies when cfg.pruning.algorithm == "PromptQuine"
    unpack = lambda m: (m["accuracy"], m["reward"])
    
    test_acc, test_reward = unpack(test_metrics_results)
    if cfg.pruning.algorithm == "PromptQuine":
        dev_acc, dev_reward = unpack(dev_metrics_results)
        output_content.update({
            "val_acc": dev_acc,
            "test_acc": test_acc,
        })
    else:
        output_content.update({
            "test_acc": test_acc,
        })
    # === Prepare DataFrame ===
    saved_df = pd.DataFrame(output_content)
    # Save to CSV
    saved_df.to_csv(prompt_saved_file_path, index=False)

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="pruner_config", node=Config)
    
    main()