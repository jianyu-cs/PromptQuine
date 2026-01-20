import sys
import os
import pdb
import copy
import datetime
import json
import argparse

import hydra
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from hydra.core.config_store import ConfigStore

from pruner_config import Config
from dataset_helper import make_classification_dataset
from fsc_evaluator import PromptedClassificationEvaluator
from promptquine.utils import measure_time, colorful_print, load_prompts
from promptquine.utils.classification import (
    get_prompts_path,
    get_output_path,
    is_masked_language_model
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@measure_time
@hydra.main(version_base=None, config_path=".", config_name="pruner_config")
def main(cfg: Config):
    # === Load datasets ===
    base_path = "./data"
    # === Build validation dataset (TAPruning dev set) ===
    valid_dataset = make_classification_dataset(
        cfg.data.dataset, cfg.data.dataset_seed, base_path, cfg.model.name, False, mode = "reduce"
    )
    eval_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=512,
        num_workers=4,
        drop_last=False
    )
    # === Build test dataset ===
    test_dataset = make_classification_dataset(
            cfg.data.dataset, cfg.data.dataset_seed, base_path, cfg.model.name, False, mode = "test"
        )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=512,
        num_workers=4,
        drop_last=False
    )
    # === Task configurations (e.g., verbalizers) ===
    is_mask_lm = is_masked_language_model(cfg.model.name)
    
    # === Load Prompts ===
    # Loading Prompt in the command line
    if cfg.prompt.prompt:
        prompts = [cfg.prompt.prompt]
        output_content = {
            "prompt": prompts,
        }
    else:
        # Loading Prompts (UnPruned) -- TODO: [[]] - prompt: []
        if not cfg.prompt.is_pruned_prompt:
            prompts_path = get_prompts_path(cfg.data.dataset, is_mask_lm, cfg.model.ICL_shots, cfg.data.is_random_verbalizers)
            prompts_dict_list = load_prompts(prompts_path)

            if cfg.model.ICL_index != None:
                prompts = [prompts_dict_list[cfg.model.ICL_index]['prompt']]
            else:
                prompts = [prompts_dict['prompt'] for prompts_dict in prompts_dict_list]

            output_content = {
                "prompt": prompts,
            }
        # Loading Prompts (Pruned)
        else:
            prompts_path = get_output_path(cfg)
            metric = "reward" if cfg.pruning.reward_driven else "acc"
            if cfg.pruning.algorithm == "PromptQuine":
                # Re-ranking for PromptQuine
                prompts_df = pd.read_csv(prompts_path)
                prompt_percent_threshold = np.percentile(prompts_df[metric], 100 - cfg.prompt_quine.top_percent_rerank)
                top_10_df = prompts_df[prompts_df[metric] >= prompt_percent_threshold].copy()
                top_10_df['rank'] = top_10_df[metric].rank(ascending=False, method='min')
                top_10_df = top_10_df.sort_values(by='rank')
                prompts = top_10_df['prompt'].tolist()
                output_content = {
                    "prompt": prompts,
                }
            elif cfg.pruning.algorithm == "TAPruning":
                # Pick the one with the highest validation score during pruning
                prompts_df = pd.read_csv(prompts_path)
                max_acc = prompts_df[metric].max()
                last_max_row = prompts_df[prompts_df[metric] == max_acc].index[-1]
                #pdb.set_trace()
                prompts = [prompts_df.iloc[last_max_row]['prompt']]
                val_accs = [prompts_df.iloc[last_max_row]['acc']]
                #pdb.set_trace()
                output_content = {
                    "prompt": prompts,
                    "val_acc": val_accs
                }
    # === Setup the evaluation ===
    tester = PromptedClassificationEvaluator(
        task_lm = cfg.model.name,
        is_mask_lm = is_mask_lm,
        dataset = cfg.data.dataset,
        prompt = prompts[0], # placeholder
        mode = cfg.model.inference_engine, # auto convert to HF if mask_lm 
        num_devices = cfg.model.num_devices
    )
    # === Start evaluation ===
    if "val_acc" not in output_content:
        val_accs = []
        for _, prompt in enumerate(prompts):
            #pdb.set_trace()
            validation_result = tester.forward(eval_loader, prompt)
            val_accs.append(validation_result['accuracy'])
        output_content['val_acc'] = val_accs
    # Evaluating on the testing set
    if cfg.prompt.prompt or not cfg.prompt.is_pruned_prompt:
        test_accs = []
        test_result = tester.forward(test_loader, prompts[0])
        test_accs.append(test_result['accuracy'])

    elif cfg.pruning.algorithm == "PromptQuine":
        test_accs = []
        if not cfg.prompt_quine.test_all_elites_for_debug:
            # prompt selection
            best_prompt, max_val_acc = max(
                zip(output_content["prompt"], output_content["val_acc"]),
                key=lambda x: x[1]
            )
            colorful_print(f"Prompt: \n{best_prompt}", fg='blue')
            colorful_print(f"Validation Accuracy: \n{max_val_acc}", fg='green')
            #pdb.set_trace()
            test_result = tester.forward(test_loader, best_prompt)
            test_accs.append(test_result['accuracy'])
            output_content['prompt'] = [best_prompt]
            output_content['val_acc'] = [max_val_acc]
        else:
            for _, prompt in enumerate(prompts):
                #pdb.set_trace()
                test_result = tester.forward(test_loader, prompt)
                test_accs.append(test_result['accuracy'])

    elif cfg.pruning.algorithm == "TAPruning":
        test_accs = []
        for _, prompt in enumerate(prompts):
            #pdb.set_trace()
            test_result = tester.forward(test_loader, prompt)
            test_accs.append(test_result['accuracy'])

    output_content['test_acc'] = test_accs
    colorful_print(f"Max Testing Accuracy: {max(test_accs)}", fg='red')
    # === Save the results into Local Directory ===
    fitness_str = "reward" if cfg.pruning.reward_driven else "accuracy"
    if cfg.prompt.prompt:
        # Save into Temporary Directory
        base_saved_path = "Prompt_cache"
        os.makedirs(base_saved_path, exist_ok=True)

        prompt_saved_file_path = os.path.join(
            base_saved_path,
            "prompt_classification.csv"
        )
    elif not cfg.prompt.is_pruned_prompt:
        # Save into Temporary Directory
        base_saved_path = f"ICLPrompts_Unpruned/{cfg.data.dataset}"
        os.makedirs(base_saved_path, exist_ok=True)

        prompt_saved_file_path = os.path.join(
            base_saved_path,
            cfg.model.name.replace('/', '-'),
            f"{cfg.model.ICL_shots}-shots_ICL",
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
            "Eval",
            suffix,
        )
        os.makedirs(base_saved_path, exist_ok=True)
        if data_cfg.split == True: # PromptQuine Only
            data_cfg.num_shots = data_cfg.num_shots // 2
        if cfg.prompt_quine.initialize_duplicate:
            filename = (
                f"{cfg.model.inference_engine}_{fitness_str}"
                f"_initialize_with_duplicates"
                f"_{cfg.data.num_shots}-shots-samples-in-train"
                f"_{pq_cfg.population_size}-population-size"
                f"_{cfg.prompt_quine.reproduction_size}-reproduction-size"
                f"_{cfg.data.dataset_seed}-data-seed"
                f"_{cfg.model.ICL_index}-index-ICL.csv"
            )
        else:
            filename = (
                f"{cfg.model.inference_engine}_{fitness_str}"
                f"_initialize_with_random_pruning"
                f"_{cfg.data.num_shots}-shots-samples-in-train"
                f"_{pq_cfg.population_size}-population-size"
                f"_{cfg.prompt_quine.reproduction_size}-reproduction-size"
                f"_{cfg.data.dataset_seed}-data-seed"
                f"_{cfg.model.ICL_index}-index-ICL.csv"
            )
        prompt_saved_file_path = os.path.join(base_saved_path, filename)
    elif cfg.pruning.algorithm == "TAPruning":
        # Create directory otherwise noted
        local_dir_name = "Ranked_top1"
        base_saved_path = os.path.join(
            "PrunedPrompts_by_TAPruning",
            cfg.model.name.replace('/', '-'),
            f"{cfg.model.ICL_shots}-shots_ICL",
            cfg.data.dataset,
            "Eval",
            local_dir_name
        )
        os.makedirs(base_saved_path, exist_ok=True)

        if cfg.pruning.fix_prune_order:
            filename = (
                f"{cfg.model.inference_engine}_{fitness_str}"
                f"_200-samples_{cfg.pruning.TAPruning_threshold}-threshold"
                f"_fixed-prune-order"
                f"_{cfg.model.ICL_index}-index-ICL.csv"
            )
        else:
            filename = (
                f"{cfg.model.inference_engine}_{fitness_str}"
                f"_200-samples_{cfg.pruning.TAPruning_threshold}-threshold"
                f"_shuffled-prune-order"
                f"_{cfg.model.ICL_index}-index-ICL.csv"
            )

        prompt_saved_file_path = os.path.join(base_saved_path, filename)

    # Prepare DataFrame
    saved_df = pd.DataFrame(output_content)
    # Save to CSV
    saved_df.to_csv(prompt_saved_file_path, index=False)

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="pruner_config", node=Config)
    
    main()