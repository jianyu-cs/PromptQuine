import os
import sys
import pdb
import json
import copy
import time
import datetime
from statistics import mean
from typing import Optional, Union, List, Dict, Any, Tuple, Type

import torch
import random
import hydra
import tiktoken
import numpy as np
import pandas as pd

from .Pruner import Pruner
from itertools import compress
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from ray.util.placement_group import placement_group
from promptquine.utils import colorful_print, create_tabu_list
from promptquine.core import (
    TaskStrategy, 
    ClassificationStrategy, 
    StyleTransferStrategy, 
    ReasoningStrategy, 
    JailbreakingStrategy
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PromptQuinePruner(Pruner):
    """The PromptQuinePruner class for PromptQuine implementations."""

    _strategies: Dict[str, Type[TaskStrategy]] = {
        "classification": ClassificationStrategy,
        "style_transfer": StyleTransferStrategy,
        "reasoning": ReasoningStrategy,
        "jailbreaking": JailbreakingStrategy
    }

    def __init__(self, 
                task_lm: str,
                evaluator_task: str,
                prompt: Optional[str] = None,
                mode: str = "vLLM",
                # PromptQuine Arguments
                initialize_duplicate: bool = True,
                min_prompt_length: int = 15,
                max_prompts_in_replication: int = 10000,
                genetic_algorithm_mode: str = "GGA",
                population_size: int = 30,
                reproduction_size: int = 50,
                # used for ClassificationEvaluator
                dataset: str = "",
                is_mask_lm: str = False,
                num_devices: int = 4,
                # used for StyleTransferEvaluator
                style_batch_size: int = 32,
                style_classifier_path: str = "",
                style_classifier_device_id: int = 1,
                # used for StyleTransferEvaluator: decoding-only
                task_top_k: int = 10,
                num_samples: int = 32,
                # used for JailbreakingEvaluator
                pruning_metric: Optional[str] = None,
                priming: bool = False,
                ## Ray parameters
                gpus_per_bundle: int = 1,
                cpus_per_bundle: int = 4
                ) -> None:
        assert population_size < max_prompts_in_replication
        # Context arguments
        self.task_lm = task_lm
        self.mode = mode
        self.evaluator_task = evaluator_task
        # Implementation Matters
        self.supports_batch = (
            True if self.mode == "vLLM"
            else False
        )
        self.task_strategy = self._strategies.get(evaluator_task)
        # Evaluator arguments
        evaluator_args = {
            "task_lm": task_lm,
            "dataset": dataset,
            "prompt": prompt,
            "num_devices": num_devices
        }
        if evaluator_task == "classification":
            evaluator_args.update({
                "mode": mode,
                "is_mask_lm": is_mask_lm
            })
        elif evaluator_task == "style_transfer":
            evaluator_args.update({
                "task_top_k": task_top_k,
                "num_samples": num_samples,
                "style_batch_size": style_batch_size,
                "style_classifier_path": style_classifier_path,
                "style_classifier_device_id": style_classifier_device_id,
            })
        elif evaluator_task == "jailbreaking":
            num_bundles = 3 if pruning_metric == "ASR-SV" else 2
            evaluator_args.update({
                "pruning_metric": pruning_metric,
                "guard_model_name": "meta-llama/Llama-Guard-3-8B",
                "placement_group": placement_group(
                    name="llm_pg",
                    bundles=[{"GPU": gpus_per_bundle, "CPU": cpus_per_bundle} for _ in range(num_bundles)],
                    strategy="STRICT_PACK"
                )
            })

        self.tester = self.task_strategy.create_evaluator(**evaluator_args)
        # PromptQuine arguments
        self.population_size = population_size
        self.min_prompt_length = min_prompt_length
        self.reproduction_size = reproduction_size
        self.genetic_algorithm_mode = genetic_algorithm_mode
        self.initialize_duplicate = initialize_duplicate
        self.max_prompts_in_replication = max_prompts_in_replication

    def forward(self, prompt: Union[str, Dict], test_loader: Any, 
                reward_driven: bool = False,
                fix_prune_order: bool = True,
                successive_halving: bool = False,
                priming: Optional[bool] = None) -> List[Tuple]:
        """Conduct PromptQuine on the specified prompts"""
        colorful_print(f"Prompt: {prompt}", fg='blue')
        # Successive Halving Checks
        if successive_halving:
            if self.evaluator_task == "classification":
                raise ValueError(f"Unsupported setup for {self.evaluator_task}")
        # In-context Attack (Jailbreaking)
        if priming is False and isinstance(prompt, dict):
            prompt = self._parse_prompt_dict(prompt)
        elif priming and isinstance(prompt, dict):
            prompt = prompt["prompt"]
        # Start pruning
        num_iterations = 1 # counting ~ how many prompts we explored.
        # Experimental setups
        population_size = self.population_size
        min_prompt_length = self.min_prompt_length
        reproduction_size = self.reproduction_size
        genetic_algorithm_mode = self.genetic_algorithm_mode
        initialize_duplicate = self.initialize_duplicate
        max_prompts_in_replication = self.max_prompts_in_replication

        tokenizer = AutoTokenizer.from_pretrained(self.task_lm)
        prompt_tokens = tokenizer.tokenize(prompt)
        prompt_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)
        prompt_len = len(prompt_tokens) 
        mask = [True for _ in range(prompt_len)]
        # Record initial prompt's performance
        init_result = self.tester.forward(test_loader, prompt)
        init_individual = self.task_strategy.create_candidate(
            prompt=prompt,
            mask=mask,
            eval_result=init_result,
            tokenizer=tokenizer
        )
        # initialize pruned-prompt-queues
        max_performance = init_individual.get_fitness(reward_driven)
        prompt_queues = [init_individual.to_list_copy()] 
        
        # Tracked Variable Setups
        prompt_populations = []
        tabu_list = create_tabu_list(tokenizer, prompt_tokens)
        Prune_allowed_indices = [i for i in range(len(prompt_ids)) if i not in tabu_list]

        # initialize prompt population
        temp_prompts = []
        for _ in range(population_size):
            prompt_offspring_mask = copy.copy(mask)
            random.seed(time.time())
            if not initialize_duplicate:
                prune_ratio = random.choice(list(np.linspace(1/20, 1/5, 10)))

                for _ in Prune_allowed_indices:
                    if prompt_offspring_mask[_] == False:
                        continue
                    random.seed(time.time())
                    if random.random() < prune_ratio:
                        prompt_offspring_mask[_] = False
            
            prompt_offspring_ids = list(compress(prompt_ids, prompt_offspring_mask))
            prompt_offspring = tokenizer.decode(prompt_offspring_ids)
            temp_prompts.append([prompt_offspring, prompt_offspring_mask])
        # prompt population evaluation
        # copy.deepcopy for safety
        if not initialize_duplicate:
            num_iterations = population_size
            if not self.supports_batch:
                for temp_prompt, temp_mask in temp_prompts:
                    temp_result = self.tester.forward(test_loader, temp_prompt)
                    temp_individual = self.task_strategy.create_candidate(
                        prompt=temp_prompt,
                        mask=temp_mask,
                        eval_result=temp_result,
                        tokenizer=tokenizer
                    )
                    prompt_populations.append(
                        temp_individual.to_list_copy()
                    )
                    prompt_queues.append(
                        temp_individual.to_list_copy()
                    )
            else: # vLLM
                temp_input_prompts = [p for p, _ in temp_prompts]
                temp_results_list = self.tester.forward_batch(test_loader, temp_input_prompts)
                for _, (temp_prompt, temp_mask) in enumerate(temp_prompts):
                    temp_result = temp_results_list[_]
                    temp_individual = self.task_strategy.create_candidate(
                        prompt=temp_prompt,
                        mask=temp_mask,
                        eval_result=temp_result,
                        tokenizer=tokenizer
                    )
                    prompt_populations.append(
                        temp_individual.to_list_copy()
                    )
                    prompt_queues.append(
                        temp_individual.to_list_copy()
                    )
        else:
            for _ in range(population_size):
                prompt_populations.append(
                    init_individual.to_list_copy()
                )
        # Start evolution
        random.seed(time.time())
        random.shuffle(prompt_populations)
        # Convergence Condition #1: number of prompts explored exceeds a threshold
        while num_iterations < max_prompts_in_replication:
            # Main loop
            colorful_print(f"Number of Iterations: {num_iterations}", fg='green')

            avg_prompt_len = mean(
                self.task_strategy.get_field(pc, "length") for pc in prompt_populations
            )
            # Successive Halving Used Only
            min_metric = min(
                self.task_strategy.get_field(pc, "performance", reward_driven=reward_driven) 
                for pc in prompt_populations
            )
            # Convergence Condition #2: mean prompt length is smaller than a threshold
            if avg_prompt_len <= min_prompt_length:
                break

            print("num_iterations:", num_iterations)
            print("num of prompts in population:", len(prompt_populations))

            random.seed(time.time()) 
            gen_count = 0
            # GGA
            temp_prompts = []
            temp_results = []
            if self.genetic_algorithm_mode == "GGA":
                while gen_count < reproduction_size:
                    prompt_pool = list(self.task_strategy.get_field(pc, "prompt") for pc in prompt_populations)
                    population_scores = list(
                        self.task_strategy.get_field(pc, "performance", reward_driven=reward_driven)
                        for pc in prompt_populations
                    )
                    print("Mean Population Score:", mean(population_scores))
                    # reproduction selection: parent
                    if len(prompt_populations) != 1:
                        num_samples = max(1, int(len(prompt_populations) * 0.2))
                        sampled_indices = random.choices(range(len(prompt_populations)), k=num_samples)
                        temp_sampled_distributions = [population_scores[sampled_index] for sampled_index in sampled_indices]
                        print("temp sampled_distribution:", temp_sampled_distributions)
                        if len(temp_sampled_distributions) == 0:
                            num_iterations = max_prompts_in_replication
                            break
                        # Tournament selection
                        sorted_indices = sorted(range(len(temp_sampled_distributions)), key=lambda i: temp_sampled_distributions[i], reverse=True)
                        parent_idx = sampled_indices[sorted_indices[0]]
                    else:
                        parent_idx = 0
                    
                    parent = prompt_populations[parent_idx] 
                    prompt_offspring_mask = copy.deepcopy(self.task_strategy.get_field(parent, "mask"))
                    parent_fitness = self.task_strategy.get_field(parent, "performance", reward_driven=reward_driven)

                    print("Parent's fitness: ", parent_fitness)
                    
                    num_prunable_tokens = sum(prompt_offspring_mask) - len(tabu_list)
                    # Mutation Rate
                    mutation_feasible_test_count = 0
                    while mutation_feasible_test_count < 10:
                        mutation_ratio = random.choice([1, 2, 3, 4])
                        if mutation_ratio < num_prunable_tokens:
                            break
                        mutation_feasible_test_count += 1

                    if mutation_feasible_test_count >= 10:
                        gen_count += 1
                        # prompt_populations.pop(parent_idx)
                        continue

                    Current_Prune_allowed_indices = [ind for ind in Prune_allowed_indices if prompt_offspring_mask[ind] == True]
                    Current_Pruned_indices = random.sample(Current_Prune_allowed_indices, mutation_ratio)
                    for Pruned_index in Current_Pruned_indices:
                        prompt_offspring_mask[Pruned_index] = False
                    
                    prompt_offspring_ids = list(compress(prompt_ids, prompt_offspring_mask))
                    prompt_offspring = tokenizer.decode(prompt_offspring_ids)
                    print("Offspring's length:", len(prompt_offspring_ids))

                    if prompt_offspring in prompt_pool:
                        gen_count += 1
                        continue

                    temp_prompts.append([prompt_offspring, prompt_offspring_mask])
                    gen_count += 1
                # GGA, population-level evaluation
                # copy.deepcopy for safety
                num_iterations += reproduction_size
                if successive_halving: 
                    # Generation/ Reasoning Only
                    sh_indices = []
                    if not self.supports_batch:
                        for temp_prompt, temp_mask in temp_prompts:
                            temp_result = self.tester.forward(test_loader, temp_prompt, successive_halving="stage_1")
                            if self.task_strategy.should_evaluate_next_round(
                                    temp_result,
                                    min_metric,
                                    reward_driven,
                                ):
                                temp_result_left = self.tester.forward(test_loader, temp_prompt, successive_halving="stage_2")
                                # Aggregation
                                eval_result_to_use = self.task_strategy.aggregate_results([temp_result, temp_result_left])
                            else:
                                eval_result_to_use = temp_result
                        
                            temp_individual = self.task_strategy.create_candidate(
                                prompt=temp_prompt,
                                mask=temp_mask,
                                eval_result=eval_result_to_use,
                                tokenizer=tokenizer
                            )
                            prompt_populations.append(temp_individual.to_list_copy())
                            prompt_queues.append(temp_individual.to_list_copy())
                    else:
                        temp_input_prompts = [p for p, _ in temp_prompts]
                        temp_results_list = self.tester.forward_batch(test_loader, temp_input_prompts, successive_halving="stage_1")
                        for i, (temp_prompt, temp_mask) in enumerate(temp_prompts):
                            temp_result = temp_results_list[i]
                            if self.task_strategy.should_evaluate_next_round(
                                    temp_result,
                                    min_metric,
                                    reward_driven,
                                    ):
                                sh_indices.append(i)
                            temp_results.append(temp_result)
                    # Organize results for stage_2
                    if self.supports_batch:
                        extracted_temp_input_prompts = [temp_input_prompts[_] for _ in sh_indices]
                        temp_results_list_left = self.tester.forward_batch(
                            test_loader, 
                            extracted_temp_input_prompts, 
                            successive_halving="stage_2"
                        )
                        # Aggregation
                        for i,sh_idx in enumerate(sh_indices):
                            eval_result_to_use = self.task_strategy.aggregate_results(
                                [temp_results[sh_idx], temp_results_list_left[i]]
                            )
                            temp_results[sh_idx] = eval_result_to_use
                        
                        for i,temp_result in enumerate(temp_results):
                            temp_individual = self.task_strategy.create_candidate(
                                prompt=temp_input_prompts[i],
                                mask=temp_prompts[i][1],
                                eval_result=temp_result,
                                tokenizer=tokenizer
                            )
                            prompt_populations.append(temp_individual.to_list_copy())
                            prompt_queues.append(temp_individual.to_list_copy())
                else:
                    # No successive halving
                    if self.mode == "HF":
                        for temp_prompt, temp_mask in temp_prompts:
                            temp_result = self.tester.forward(test_loader, temp_prompt)
                            temp_results.append(temp_result)
                    else: # vLLM
                        temp_input_prompts = [p for p, _ in temp_prompts]
                        temp_results_list = self.tester.forward_batch(test_loader, temp_input_prompts)
                        for _, (temp_prompt, temp_mask) in enumerate(temp_prompts):
                            temp_result = temp_results_list[_]
                            temp_results.append(temp_result)
                    
                    for i,temp_result in enumerate(temp_results):
                        temp_prompt, temp_mask = temp_prompts[i]
                        temp_individual = self.task_strategy.create_candidate(
                            prompt=temp_prompt,
                            mask=temp_mask,
                            eval_result=temp_result,
                            tokenizer=tokenizer
                        )
                        prompt_populations.append(
                            temp_individual.to_list_copy()
                        )
                        prompt_queues.append(
                            temp_individual.to_list_copy()
                        )
                # approximate regularized evolution
                # Refactor, where is the score TODO
                temp_prompt_populations = prompt_populations[-reproduction_size:] 
                temp_population_scores = [
                    self.task_strategy.get_field(pc, "performance", reward_driven=reward_driven) for pc in temp_prompt_populations
                ]
                sorted_indices = sorted(range(len(temp_population_scores)), key=lambda i: temp_population_scores[i], reverse=True)[:population_size]
                prompt_populations = [temp_prompt_populations[sorted_index] for sorted_index in sorted_indices]
            # SSGA
            elif self.genetic_algorithm_mode == "SSGA":
                while gen_count < reproduction_size:
                    prompt_pool = list(self.task_strategy.get_field(pc, "prompt") for pc in prompt_populations)
                    population_scores = list(
                        self.task_strategy.get_field(pc, "performance", reward_driven=reward_driven)
                        for pc in prompt_populations
                    )
                    print("Mean Population Score:", mean(population_scores))
                    # reproduction selection: parent
                    if len(prompt_populations) != 1:
                        num_samples = max(1, int(len(prompt_populations) * 0.2))
                        sampled_indices = random.choices(range(len(prompt_populations)), k=num_samples)
                        temp_sampled_distributions = [population_scores[sampled_index] for sampled_index in sampled_indices]
                        print("temp sampled_distribution:", temp_sampled_distributions)
                        if len(temp_sampled_distributions) == 0:
                            num_iterations = max_prompts_in_replication
                            break
                        # Tournament selection
                        sorted_indices = sorted(range(len(temp_sampled_distributions)), key=lambda i: temp_sampled_distributions[i], reverse=True)
                        parent_idx = sampled_indices[sorted_indices[0]]
                    else:
                        parent_idx = 0

                    parent = prompt_populations[parent_idx] 
                    prompt_offspring_mask = copy.deepcopy(self.task_strategy.get_field(parent, "mask"))
                    parent_fitness = self.task_strategy.get_field(parent, "performance", reward_driven=reward_driven)

                    print("Parent's fitness: ", parent_fitness)
                    
                    num_prunable_tokens = sum(prompt_offspring_mask) - len(tabu_list)
                    # Mutation Rate
                    mutation_feasible_test_count = 0
                    while mutation_feasible_test_count < 10:
                        mutation_ratio = random.choice([1, 2, 3, 4])
                        if mutation_ratio < num_prunable_tokens:
                            break
                        mutation_feasible_test_count += 1

                    if mutation_feasible_test_count >= 10:
                        gen_count += 1
                        # prompt_populations.pop(parent_idx)
                        continue

                    Current_Prune_allowed_indices = [ind for ind in Prune_allowed_indices if prompt_offspring_mask[ind] == True]
                    Current_Pruned_indices = random.sample(Current_Prune_allowed_indices, mutation_ratio)
                    for Pruned_index in Current_Pruned_indices:
                        prompt_offspring_mask[Pruned_index] = False
                    
                    prompt_offspring_ids = list(compress(prompt_ids, prompt_offspring_mask))
                    prompt_offspring = tokenizer.decode(prompt_offspring_ids)
                    print("Offspring's length:", len(prompt_offspring_ids))

                    if prompt_offspring in prompt_pool:
                        gen_count += 1
                        continue

                    num_iterations += 1
                    # Offspring evaluation (SSGA feature)
                    temp_result = self.tester.forward(test_loader, prompt_offspring)
                    temp_individual = self.task_strategy.create_candidate(
                        prompt=prompt_offspring,
                        mask=prompt_offspring_mask,
                        eval_result=temp_result,
                        tokenizer=tokenizer
                    )
                    prompt_populations.append(
                        temp_individual.to_list_copy()
                    )
                    prompt_queues.append(
                        temp_individual.to_list_copy()
                    )
                    gen_count += 1
                # approximate regularized evolution
                temp_prompt_populations = prompt_populations[-reproduction_size:]
                temp_population_scores = [
                    self.task_strategy.get_field(pc, "performance", reward_driven=reward_driven) for pc in temp_prompt_populations
                ]
                sorted_indices = sorted(range(len(temp_population_scores)), key=lambda i: temp_population_scores[i], reverse=True)[:population_size]
                prompt_populations = [temp_prompt_populations[sorted_index] for sorted_index in sorted_indices]
            else:
                raise NotImplementedError("Not implemented such algorithm_mode. Please check.")

        return prompt_queues, num_iterations