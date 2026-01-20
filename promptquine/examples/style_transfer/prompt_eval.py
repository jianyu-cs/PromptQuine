import argparse
import dataclasses
import json
import os
import sys
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
from tst_evaluator import PromptedStyleTransferEvaluator
from dataset_helper import load_text_style_transfer_dataset, get_style_classifier
from tst_modules import (
    PromptedGenerator,
    TextStyleTransferOutputSelector,
)
from promptquine.utils import measure_time, colorful_print, load_prompts
from promptquine.utils.style_transfer import (
    get_prompts_path,
    get_output_path
)

def evaluate_prompts(prompts, prompted_generator,
                     source_texts, target_labels, ref_texts):
    """Evaluate prompts on given dataset and return metrics."""
    metrics_lists = {key: [] for key in ['joint_score', 'gm', 'content', 'style', 'fluency']}

    for prompt in prompts:
        prompted_generator.template = prompt
        print("prompt:", prompted_generator.template)

        result = prompted_generator.forward(
            data_lists = [source_texts, target_labels],
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
    # 1. load classifier
    style_classifier_path = \
        os.path.join('.', get_style_classifier(cfg.model.style_classifier, cfg.data.dataset))
    # 2. load dataset
    # dev (max_size) & test
    source_texts, target_labels = \
            load_text_style_transfer_dataset(
            cfg.data.dataset,
            cfg.data.direction, "dev", 0,
            base_path=cfg.data.base_path, max_size=cfg.data.max_size, # How many dev samples
            max_length=cfg.data.max_length,
            max_length_tokenizer=cfg.data.max_length_tokenizer)

    test_source_texts, test_target_labels = \
            load_text_style_transfer_dataset(
            cfg.data.dataset,
            cfg.data.direction, "test", 0,
            base_path=cfg.data.base_path, max_size=None, # Full test samples
            max_length=cfg.data.max_length,
            max_length_tokenizer=cfg.data.max_length_tokenizer)

    ref_texts, _ = \
            load_text_style_transfer_dataset(
            cfg.data.dataset,
            cfg.data.direction, "ref", 0,
            base_path=cfg.data.base_path, max_size=None, # Full ref samples
            max_length=cfg.data.max_length,
            max_length_tokenizer=cfg.data.max_length_tokenizer)
    # 3. Load Prompts
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
            prompts_path = get_prompts_path(cfg.model.ICL_shots, cfg.data.direction)
            prompt_dict_list = load_prompts(prompts_path)

            prompts = [prompt_dict_list[cfg.model.ICL_index]['prompt']]
            
            output_content = {
                "prompt": prompts,
            }
        # Loading Prompts (Pruned)
        else:
            prompts_path = get_output_path(cfg)
            metric = "joint" # unless otherwise stated
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
                prompts, val_joint, val_gm, val_content, val_style, val_fluency, val_num_tokens = [
                    [prompts_df.iloc[last_max_row][col]] 
                    for col in ['prompt', 'joint', 'gm', 'content', 'style', 'fluency', '#tokens']
                ]
                output_content = {
                    "prompt": prompts,
                    "val_joint": val_joint,
                    "val_gm": val_gm,
                    "val_content": val_content,
                    "val_style": val_style,
                    "val_fluency": val_fluency,
                    "num_tokens": val_num_tokens
                }
    # 4. Setup the evaluation
    print('Test Size:', len(test_source_texts))
    print('Examples:', test_source_texts[:5])
    prompted_generator = PromptedStyleTransferEvaluator(
        task_lm = cfg.model.name,
        dataset = cfg.data.dataset,
        prompt = cfg.prompt.prompt,
        num_devices = cfg.model.num_devices,
        style_batch_size = cfg.model.style_batch_size,
        style_classifier_path = style_classifier_path,
        style_classifier_device_id = cfg.model.style_classifier_device_id,
        num_samples = cfg.model.num_samples,
        task_top_k = cfg.model.task_top_k,
    )
    # 5. Start evaluation
    test_metrics_results = evaluate_prompts(
        prompts, prompted_generator,
        test_source_texts, test_target_labels, ref_texts,
    )
    if cfg.pruning.algorithm == "PromptQuine":
        dev_metrics_results = evaluate_prompts(
            prompts, prompted_generator,
            source_texts, target_labels, ref_texts,
        )
    # 6. Prepare prompt file
    if cfg.prompt.prompt:
        # Save into Temporary Directory
        base_saved_path = "Prompt_cache"
        os.makedirs(base_saved_path, exist_ok=True)

        prompt_saved_file_path = os.path.join(
            base_saved_path,
            "prompt_style_transfer.csv"
        )
    elif not cfg.prompt.is_pruned_prompt:
        # Save into Temporary Directory
        base_saved_path = os.path.join(
            "ICLPrompts_Unpruned",
            cfg.data.dataset,
            cfg.data.direction,
            cfg.model.name.replace('/', '-'),
            f"{cfg.model.ICL_shots}-shots_ICL"
        )
        os.makedirs(base_saved_path, exist_ok=True)

        prompt_saved_file_path = os.path.join(
            base_saved_path,
            f"{cfg.model.ICL_index}-index-ICL.csv"
        )
    elif cfg.pruning.algorithm == "PromptQuine":
        # Create directory otherwise noted
        if cfg.prompt_quine.test_all_elites_for_debug:
            suffix = f"Test_all_top{cfg.prompt_quine.top_percent_rerank}_percent"
        else:
            suffix = "Ranked_top1"

        base_saved_path = os.path.join(
            "PrunedPrompts_by_PromptQuine",
            cfg.prompt_quine.algorithm_mode,
            cfg.model.name.replace('/', '-'),
            f"{cfg.model.ICL_shots}-shots_ICL",
            cfg.data.dataset,
            cfg.data.direction,
            "Eval",
            suffix,
        )
        os.makedirs(base_saved_path, exist_ok=True)
        successive_halving_str = "halving" if cfg.prompt_quine.successive_halving else "no_halving"
        if cfg.prompt_quine.initialize_duplicate:
            filename = (
                f"{successive_halving_str}_initialize_with_duplicates"
                f"_{cfg.prompt_quine.reproduction_size}-reproduction-size"
                f"_{cfg.data.max_size}-samples"
                f"_{cfg.model.ICL_index}-index-ICL.csv"
            )
        else:
            filename = (
                f"{successive_halving_str}_initialize_with_random_pruning"
                f"_{cfg.prompt_quine.reproduction_size}-reproduction-size"
                f"_{cfg.data.max_size}-samples"
                f"_{cfg.model.ICL_index}-index-ICL.csv"
            )
        prompt_saved_file_path = os.path.join(base_saved_path, filename)
    elif cfg.pruning.algorithm == "TAPruning":
        # Create directory otherwise noted
        local_dir_name = "Ranked_top1"
        base_saved_path = os.path.join(
            output_dir,
            cfg.model.name.replace('/', '-'),
            f"{cfg.model.ICL_shots}-shots_ICL",
            cfg.data.dataset,
            cfg.data.direction,
            "Eval",
            local_dir_name
        )
        os.makedirs(base_saved_path, exist_ok=True)

        if cfg.pruning.fix_prune_order:
            filename = (
                f"{cfg.data.max_size}-samples_{cfg.pruning.TAPruning_threshold}-threshold"
                f"_fixed-prune-order"
                f"_{cfg.model.ICL_index}-index-ICL.csv"
            )
        else:
            filename = (
                f"{cfg.data.max_size}-samples_{cfg.pruning.TAPruning_threshold}-threshold"
                f"_shuffled-prune-order"
                f"_{cfg.model.ICL_index}-index-ICL.csv"
            )
        prompt_saved_file_path = os.path.join(base_saved_path, filename)
    # 7. Save evaluation results
    # === Prepare DataFrame ===
    # + Parse test_metrics_results and dev_metrics_results (Conditional) 
    # - Only applies when cfg.pruning.algorithm == "PromptQuine"
    unpack = lambda m: (m["content"], m["style"], m["fluency"], m["joint_score"], m["gm"])
    test_content, test_style, test_fluency, test_joint, test_gm = unpack(test_metrics_results)
    if cfg.pruning.algorithm == "PromptQuine":
        dev_content, dev_style, dev_fluency, dev_joint, dev_gm = unpack(dev_metrics_results)
        output_content.update({
            "val_joint": dev_joint,
            "val_gm": dev_gm,
            "val_content": dev_content,
            "val_style": dev_style,
            "val_fluency": dev_fluency,
            "test_joint": test_joint,
            "test_gm": test_gm,
            "test_content": test_content,
            "test_style": test_style,
            "test_fluency": test_fluency,
        })
    else:
        output_content.update({
            "test_joint": test_joint,
            "test_gm": test_gm,
            "test_content": test_content,
            "test_style": test_style,
            "test_fluency": test_fluency,
        })
    # === Prepare DataFrame ===
    saved_df = pd.DataFrame(output_content)
    # Save to CSV
    saved_df.to_csv(prompt_saved_file_path, index=False)

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="pruner_config", node=Config)
    
    main()