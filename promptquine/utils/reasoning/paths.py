"""Utilities for path manipulation."""
import os
import json
from typing import Any
from promptquine.env import PROJECT_ROOT 

def read_reasoning_data(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def get_prompts_path(dataset: str, icl_shots: int) -> str:
    """Retrieve the file path for prompt templates"""
    base_path = f"{PROJECT_ROOT}/prompts/reasoning_prompts/{dataset}"

    shot_suffix = f"{icl_shots}shot"

    return f"{base_path}/{dataset}_few_shot_natural_prompts_{shot_suffix}.jsonl"

def get_output_path(config: Any) -> str:
    """
    Generates and creates a unique output path based on the provided configuration object.

    Args:
        config (Config): The configuration object containing all experiment parameters.

    Returns:
        str: The fully constructed output file path.
    """
    # 1. Unpack sub-configs from the main config object for better readability.
    data_cfg = config.data
    model_cfg = config.model
    pruning_cfg = config.pruning
    pq_cfg = config.prompt_quine

    # 2. Construct the path and filename using values from the config.
    output_dir = f"{PROJECT_ROOT}/examples/reasoning/PrunedPrompts_by_{pruning_cfg.algorithm}/"

    # --- Branch 1: Handle the "TAPruning" algorithm ---
    if pruning_cfg.algorithm == "TAPruning":
        # Create the base dir
        base_dir = os.path.join(
            output_dir,
            model_cfg.name.replace('/', '-'),
            f"{model_cfg.ICL_shots}-shots_ICL",
            data_cfg.dataset,
            "Train"
        )
        os.makedirs(base_dir, exist_ok=True)
        # Build the saved file path
        if pruning_cfg.fix_prune_order:
            filename = (
                f"200-samples_{pruning_cfg.TAPruning_threshold}-threshold"
                f"_fixed-prune-order"
                f"_{model_cfg.ICL_index}-index-ICL.csv"
            )
        else:
            filename = (
                f"200-samples_{pruning_cfg.TAPruning_threshold}-threshold"
                f"_shuffled-prune-order"
                f"_{model_cfg.ICL_index}-index-ICL.csv"
            )
        return os.path.join(base_dir, filename)

    # --- Branch 2: Handle the "PromptQuine" algorithm ---
    if pruning_cfg.algorithm == "PromptQuine":
        # PromptQuine has its own subdirectory based on its algorithm mode.
        subdir = os.path.join(output_dir, pq_cfg.algorithm_mode)
        base_dir = os.path.join(
            subdir,
            model_cfg.name.replace('/', '-'),
            f"{model_cfg.ICL_shots}-shots_ICL",
            data_cfg.dataset,
            "Train"
        )
        os.makedirs(base_dir, exist_ok=True)
        successive_halving_str = "halving" if pq_cfg.successive_halving else "no_halving"
        if pq_cfg.initialize_duplicate:
            filename = (
                f"{successive_halving_str}_initialize_with_duplicates"
                f"_{data_cfg.max_size}-samples-in-train"
                f"_{pq_cfg.population_size}-population-size"
                f"_{pq_cfg.reproduction_size}-reproduction-size"
                f"_{model_cfg.ICL_index}-index-ICL.csv"
            )
        else:
            filename = (
                f"{successive_halving_str}_initialize_with_random_pruning"
                f"_{data_cfg.max_size}-samples-in-train"
                f"_{pq_cfg.population_size}-population-size"
                f"_{pq_cfg.reproduction_size}-reproduction-size"
                f"_{model_cfg.ICL_index}-index-ICL.csv"
            )
        return os.path.join(base_dir, filename)

    # Raise an error for any unknown algorithm to prevent silent failures.
    raise ValueError(f"Unknown pruning algorithm: {pruning_cfg.algorithm}")
