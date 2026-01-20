import os
import sys
import json
import copy
import sys
import time
import datetime
import random
from typing import Optional, Union, List, Dict, Any, Tuple, Type

import hydra
import tiktoken
import pandas as pd
from itertools import compress
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from ray.util.placement_group import placement_group
from transformers import AutoTokenizer, LlamaTokenizer

from .Pruner import Pruner
from promptquine.utils import colorful_print, create_tabu_list
from promptquine.core import (
    TaskStrategy, 
    ClassificationStrategy, 
    StyleTransferStrategy, 
    ReasoningStrategy, 
    JailbreakingStrategy
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TAPruner(Pruner):
    """The TAPruner class for TAPruning implementations."""

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
                threshold: int = 1, 
                num_devices: int = 1,
                # used for ClassificationEvaluator
                dataset: str = "", 
                is_mask_lm: str = False, 
                # used for StyleTransferEvaluator
                style_batch_size: int = -1, 
                style_classifier_path: str = None, 
                style_classifier_device_id: int = None, 
                num_samples: int = 1, 
                task_top_k: int = 1,
                # used for JailbreakingEvaluator
                pruning_metric: Optional[str] = None,
                priming: bool = False,
                ## Ray parameters
                gpus_per_bundle: int = 1,
                cpus_per_bundle: int = 4
                ):
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
        self.task_lm = task_lm
        self.threshold = threshold
    
    def forward(self, prompt: Union[str, Dict], test_loader: Any, 
                reward_driven: bool = False,
                fix_prune_order: bool = True,
                priming: Optional[bool] = None,
                **kwargs) -> List[Tuple]:
        """Conduct TAPruning on the specified prompts
        Args:
            test_loader: DataLoader for other tasks, except Style Transfer with [source_texts, target_labels, ref_texts]
        """
        colorful_print(f"Prompt: {prompt}", fg='blue')
        # In-context Attack (Jailbreaking)
        if priming is False and isinstance(prompt, dict):
            prompt = self._parse_prompt_dict(prompt)
        elif priming and isinstance(prompt, dict):
            prompt = prompt["prompt"]
        # Start pruning
        num_iterations = 1 # counting ~ how many prompts we explored.
        # Experimental setups
        tokenizer = AutoTokenizer.from_pretrained(self.task_lm)
        prompt_tokens = tokenizer.tokenize(prompt)
        prompt_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)
        prompt_len = len(prompt_tokens) 
        mask = [True for _ in range(len(prompt_ids))]
        # initialize pruned-prompt-queues
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
        print(f"Initial Performance: {max_performance}")
    
        # Tracked Variable Setups
        outer_prompt_length = len(prompt_ids)
        inner_optimal_prompt = copy.copy(prompt)
        tabu_list = create_tabu_list(tokenizer, prompt_tokens)
        Prune_allowed_indices = [i for i in range(len(prompt_ids)) if i not in tabu_list]

        random.seed()
        if not fix_prune_order:
            random.shuffle(Prune_allowed_indices)

        admissable_len = len(Prune_allowed_indices)
        IND2POS = {}
        for _ in range(admissable_len):
            if _ not in IND2POS:
                IND2POS[_] = Prune_allowed_indices[_]
        # Main Pruning Loops    
        counter = 0
        while True:
            counter = 0 
            while counter < admissable_len:
                colorful_print(f"Count: {counter}, Token: {prompt_tokens[IND2POS[counter]]}", fg='green')
                # leave-one-token-out
                mask[IND2POS[counter]] = False
                temp_prompt = tokenizer.decode(list(compress(prompt_ids, mask)))
                colorful_print(f"Prompt: {temp_prompt}", fg='red')
                temp_result = self.tester.forward(test_loader, temp_prompt)
                temp_individual = self.task_strategy.create_candidate(
                    prompt=temp_prompt,
                    mask=mask,
                    eval_result=temp_result,
                    tokenizer=tokenizer
                )
                performance = temp_individual.get_fitness(reward_driven)
                print(f"Fitness: {performance}")
                
                num_iterations += 1

                if performance/max_performance < self.threshold:
                    # cancel this masking
                    colorful_print(f"Keep the token, as {performance} is smaller than {max_performance} under threshold: {self.threshold}", fg='blue')
                    mask[IND2POS[counter]] = True
                else:
                    max_performance = performance if performance > max_performance else max_performance
                    inner_optimal_prompt = temp_prompt  
                    colorful_print(f"Updated Prompt: {inner_optimal_prompt}", fg='green') 
                    prompt_queues.append(temp_individual.to_list_copy())

                counter += 1
            colorful_print("Next Iteration!", fg='green')
            colorful_print(f"Prompt: {inner_optimal_prompt}", fg='green')
            counter = 0
            
            prompt_tokens = tokenizer.tokenize(inner_optimal_prompt)
            prompt_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)
            
            if outer_prompt_length == len(prompt_tokens):
                # No tokens can be further pruned!
                break
            else:
                outer_prompt_length = len(prompt_tokens)
            
            prompt_len = len(prompt_ids)
            mask = [True for _ in range(prompt_len)]

            tabu_list = create_tabu_list(tokenizer, prompt_tokens)
            Prune_allowed_indices = [i for i in range(len(prompt_ids)) if i not in tabu_list]
            random.seed()
            if not fix_prune_order:
                random.shuffle(Prune_allowed_indices)
            admissable_len = len(Prune_allowed_indices)
            IND2POS = {}
            for _ in range(admissable_len):
                if _ not in IND2POS:
                    IND2POS[_] = Prune_allowed_indices[_]
                    
        return prompt_queues, num_iterations
        