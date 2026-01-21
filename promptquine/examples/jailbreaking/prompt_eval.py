"""
We always pick the worst prompt for final text outputs, to alleviate reward hacking.
This one is only for debugging.
"""
import argparse
import dataclasses
import json
import os
import sys
import pdb
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

import ray
import hydra
import torch
import numpy as np
import pandas as pd
from vllm import LLM
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from ray.util.placement_group import placement_group

from pruner_config import Config
from dataset_helper import load_jailbreaking_dataset
from jailbreaking_evaluator import PromptedJailbreakingEvaluator
from promptquine.utils import measure_time, colorful_print, load_prompts
from promptquine.utils.jailbreaking import (
    get_prompts_path,
    get_output_path,
    ensure_dirs
)


def evaluate_prompts(
    prompts: List[str],
    prompted_generator,
    test_source_instances,
) -> List[List[str]]:
    """
    Run prompts on given dataset and model, and return generated output texts and metric results.
    
    Args:
        prompts: List of prompt templates.
        prompted_generator: Model interface with a `forward(data)` method.
        test_source_instances: DataFrame or dict containing 'prompt' column.

    Returns:
        A list of lists containing generated texts for each prompt.
    """
    
    source_texts = test_source_instances["prompt"].tolist()
    full_output_texts = []
    full_evaluation_results = []

    for prompt in prompts:
        # Generate outputs
        evaluation_results, output_texts = prompted_generator.forward(
            input_queries=source_texts,
            prompt=prompt,
            Eval=True
        )
        full_output_texts.append(output_texts)
        full_evaluation_results.append(evaluation_results)

    return full_output_texts, full_evaluation_results

@measure_time
@hydra.main(version_base=None, config_path=".", config_name="pruner_config")
def main(cfg):
    # 0. initialize ray
    ray.init()
    # 1. load dataset
    # dev (max_size) & test
    base_path = "./data"
    source_instances = load_jailbreaking_dataset(
        cfg.data.dataset,
        "dev", # Full dev samples
        base_path 
    )

    test_source_instances = load_jailbreaking_dataset(
        cfg.data.dataset,
        "test", # Full test samples
        base_path
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

            prompts = [prompt_dict_list[cfg.model.ICL_index]['prompt']]
            demo_inputs = [prompt_dict_list[cfg.model.ICL_index]['inputs']]
            demo_outputs = [prompt_dict_list[cfg.model.ICL_index]['outputs']]

            if cfg.model.priming is False:
                prompts = [
                format_chat_prompt("{sentence_1}", demo_inputs, demo_outputs, cfg.model.name)
                for _ in prompts
            ]

            output_content = {
                "prompt": prompts,
            }
        # Loading Prompts (Pruned)
        # Requirement: All prompts have been formatted already (e.g., conversational tags)
        else:
            prompts_path = get_output_path(cfg)
            metric = "ASR-Fitness" # unless otherwise stated
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
                # Branching
                # max_joint = prompts_df[metric].max()
                # last_max_row = prompts_df[prompts_df[metric] == max_joint].index[-1]
                sorted_indices = prompts_df[metric].sort_values(ascending=False).index
                last_max_row = sorted_indices[cfg.pruning.selection_top_k]

                prompts, val_fitness, val_num_tokens = [
                    [prompts_df.iloc[last_max_row][col]] 
                    for col in ['prompt', 'ASR-Fitness', '#tokens']
                ]
                output_content = {
                    "prompt": prompts,
                    "val_fitness": val_fitness,
                    "num_tokens": val_num_tokens
                }
    # 3. Setup the evaluation
    num_bundles = 3 if cfg.pruning.pruning_metric == "ASR-SV" else 2
    pg = placement_group(
        name="llm_pg",
        bundles=[{"GPU": cfg.model.gpus_per_bundle, "CPU": cfg.model.cpus_per_bundle} for _ in range(num_bundles)],
        strategy="STRICT_PACK"
    )
    prompted_generator = PromptedJailbreakingEvaluator(
        task_lm = cfg.model.name,
        dataset = cfg.data.dataset,
        num_devices = cfg.model.num_devices,
        prompt = None, # Optional
        placement_group = pg,
        guard_model_name = "meta-llama/Llama-Guard-3-8B",
        priming = cfg.model.priming,
        pruning_metric = cfg.pruning.pruning_metric
    )
    # 4. Start evaluation
    priming_conditioner = not cfg.prompt.is_pruned_prompt and cfg.model.priming
    test_output_texts, test_evaluation_results = evaluate_prompts(
        prompts, prompted_generator,
        test_source_instances,
    )
    dev_output_texts, dev_evaluation_results = evaluate_prompts(
        prompts, prompted_generator,
        source_instances,
    )
    # 5. Prepare generated file (output texts)
    if cfg.prompt.prompt:
        # Save into Temporary Directory
        base_saved_dir = f"Prompt_cache/{cfg.pruning.pruning_metric}"
        dirs = ensure_dirs(base_saved_dir, ["prompted_generated_texts", "prompted_pruned_results"])
        base_generated_texts_dir = dirs["prompted_generated_texts"]
        base_pruned_results_dir = dirs["prompted_pruned_results"]

        prompt_output_saved_file_path = os.path.join(
            base_generated_texts_dir,
            f"{cfg.pruning.pruning_metric}_prompt_jailbreaking_outputs.csv"
        )
        prompt_result_saved_file_path = os.path.join(
            base_pruned_results_dir,
            f"{cfg.pruning.pruning_metric}_prompt_jailbreaking_outputs.csv"
        )

    elif not cfg.prompt.is_pruned_prompt:
        # Save into Temporary Directory
        base_saved_dir = os.path.join(
            "ICLPrompts_Unpruned",
            cfg.data.dataset,
            cfg.model.name.replace('/', '-'),
            f"{cfg.model.ICL_shots}-shots_ICL",
            cfg.pruning.pruning_metric,
        )
        dirs = ensure_dirs(base_saved_dir, ["prompted_generated_texts", "prompted_pruned_results"])
        base_generated_texts_dir = dirs["prompted_generated_texts"]
        base_pruned_results_dir = dirs["prompted_pruned_results"]

        prompt_output_saved_file_path = os.path.join(
            base_generated_texts_dir,
            f"{cfg.pruning.pruning_metric}_{cfg.model.ICL_index}-index-ICL.csv"
        )
        prompt_result_saved_file_path = os.path.join(
            base_pruned_results_dir,
            f"{cfg.pruning.pruning_metric}_{cfg.model.ICL_index}-index-ICL.csv"
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
            cfg.pruning.pruning_metric,
            suffix,
        )
        dirs = ensure_dirs(base_saved_dir, ["prompted_generated_texts", "prompted_pruned_results"])
        base_generated_texts_dir = dirs["prompted_generated_texts"]
        base_pruned_results_dir = dirs["prompted_pruned_results"]

        successive_halving_str = "halving" if cfg.prompt_quine.successive_halving else "no_halving"
        if cfg.prompt_quine.initialize_duplicate:
            filename = (
                f"{successive_halving_str}_initialize_with_duplicates"
                f"_{cfg.data.max_size}-samples-in-train"
                f"_{cfg.prompt_quine.population_size}-population-size"
                f"_{cfg.prompt_quine.reproduction_size}-reproduction-size"
                f"_{cfg.pruning.pruning_metric}_{cfg.model.ICL_index}-index-ICL.csv"
            )
        else:
            filename = (
                f"{successive_halving_str}_initialize_with_random_pruning"
                f"_{cfg.data.max_size}-samples-in-train"
                f"_{cfg.prompt_quine.population_size}-population-size"
                f"_{cfg.prompt_quine.reproduction_size}-reproduction-size"
                f"_{cfg.pruning.pruning_metric}_{cfg.model.ICL_index}-index-ICL.csv"
            )
        prompt_output_saved_file_path = os.path.join(
            base_generated_texts_dir,
            filename
        )
        prompt_result_saved_file_path = os.path.join(
            base_pruned_results_dir,
            filename
        )
    elif cfg.pruning.algorithm == "TAPruning":
        # Create directory otherwise noted
        local_dir_name = "Ranked_top1"
        base_saved_dir = os.path.join(
            "PrunedPrompts_by_TAPruning",
            cfg.model.name.replace('/', '-'),
            f"{cfg.model.ICL_shots}-shots_ICL",
            cfg.data.dataset,
            cfg.pruning.pruning_metric,
            local_dir_name
        )
        dirs = ensure_dirs(base_saved_dir, ["prompted_generated_texts", "prompted_pruned_results"])
        base_generated_texts_dir = dirs["prompted_generated_texts"]
        base_pruned_results_dir = dirs["prompted_pruned_results"]

        if cfg.pruning.fix_prune_order:
            filename = (
                f"200-samples_{cfg.pruning.TAPruning_threshold}-threshold"
                f"_fixed-prune-order"
                f"_{cfg.pruning.pruning_metric}_{cfg.model.ICL_index}-index-ICL.csv"
            )
        else:
            filename = (
                f"200-samples_{cfg.pruning.TAPruning_threshold}-threshold"
                f"_shuffled-prune-order"
                f"_{cfg.pruning.pruning_metric}_{cfg.model.ICL_index}-index-ICL.csv"
            )
        prompt_output_saved_file_path = os.path.join(
            base_generated_texts_dir,
            filename
        )
        prompt_result_saved_file_path = os.path.join(
            base_pruned_results_dir,
            filename
        )
    # 6. Release ray
    ray.shutdown()
    # 7. Save generated outputs
    # === Prepare DataFrame ===
    # === Dev Results ===
    dev_em_score = [r["EM_score"] for r in dev_evaluation_results]
    dev_sv_score = [r["SV_score"] for r in dev_evaluation_results]
    dev_llm_score = [r["Guard_score"] for r in dev_evaluation_results]
    # === Test Results ===
    test_em_score = [r["EM_score"] for r in test_evaluation_results]
    test_sv_score = [r["SV_score"] for r in test_evaluation_results]
    test_llm_score = [r["Guard_score"] for r in test_evaluation_results]
    # Aggregate metric results
    output_content.update({
        "val_ASR-EM": dev_em_score,
        "val_ASR-LLM": dev_llm_score,
        "val_ASR_SV": dev_sv_score,
        "test_ASR-EM": test_em_score,
        "test_ASR-LLM": test_llm_score,
        "test_ASR_SV": test_sv_score,
    })
    # Aggregate generated texts (prompts[0])
    generated_texts_content = {
        #"val_input": source_instances["prompt"],
        "test_input": test_source_instances["prompt"],
        #"val_output": dev_output_texts[0],
        "test_output": test_output_texts[-1],
    }
    print("=== Debug Info ===")
    # === Prepare DataFrame ===
    saved_df = pd.DataFrame(output_content)
    saved_texts_df = pd.DataFrame(generated_texts_content)
    # Save to CSV
    saved_df.to_csv(prompt_result_saved_file_path, index=False)
    saved_texts_df.to_csv(prompt_output_saved_file_path, index=False)

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="pruner_config", node=Config)
    
    main()