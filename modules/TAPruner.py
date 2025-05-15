import hydra
import tiktoken
import os
import sys
sys.path.append("..")
import json
import copy
import sys
import time
import datetime
import pandas as pd
import random
from .Pruner import Pruner
from itertools import compress
from omegaconf import OmegaConf

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaTokenizer
from typing import Optional, Union, List, Dict, Any, Tuple
from utils.task_wrappers import colorful_print, create_tabulist, PromptedTaskWrapperBase
# TODO
from examples.classification.fsc_evaluator import PromptedClassificationEvaluator
from examples.reasoning.reasoning_evaluator import PromptedReasoningEvaluator


class TAPruner(Pruner):
    """The TAPruner class for TAPruning implementations."""
    
    def __init__(self, 
                 prompt_evaluator: Optional[PromptedClassificationEvaluator, 
                             PromptedReasoningEvaluator, TextStyleTransferEvaluator], 
                             #PromptedStyleTransferEvaluator, PromptedMathReasoningEvaluator],
                 task_lm: str, evaluator_task: str, prompt: Optional[str] = None,
                 mode: str = "vLLM", threshold: int = 1, num_devices: int = 1,
                 # used for ClassificationEvaluator
                 dataset: str = "", is_mask_lm: str = False) -> None:
        assert evaluator_task in ["classification", "style_transfer", "math_reasoning"]
        if evaluator_task == "classification":
            self.tester = prompt_evaluator(task_lm, is_mask_lm, dataset, prompt, mode, num_devices)
        elif evaluator_task == "reasoning":
            self.tester = prompt_evaluator(task_lm, dataset, prompt, num_devices)
        elif evaluator_task == "style_transfer":
            self.tester = prompt_evaluator(task_lm, dataset, prompt, mode, num_devices)
        
        self.task_lm = task_lm
        self.threshold = threshold # 1 => greedy
    
    def forward(self, prompt: str, test_loader: Any, 
                reward_driven: bool = False,
                fix_prune_order: bool = True) -> List[Tuple]:
        """Conduct TAPruning on the specified prompts"""
        colorful_print(f"Prompt: {prompt}", fg='blue')
        # Start pruning
        num_iterations = 1 # counting ~ how many prompts we explored.
        # Experimental setups
        tokenizer = AutoTokenizer.from_pretrained(self.task_lm)
        prompt_tokens = tokenizer.tokenize(prompt)
        prompt_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)
        prompt_len = len(prompt_tokens) 
        # Record initial prompt's performance
        initial_acc, initial_reward = self.tester.forward(test_loader, prompt)
        max_performance = initial_acc if not reward_driven else initial_reward
        # initialize pruned-prompt-queues
        mask = [True for _ in range(len(prompt_ids))]
        prompt_queues = [(prompt, initial_acc, initial_reward, prompt_len, mask)] 
        
        # Tracked Variable Setups
        outer_prompt_length = len(prompt_ids)
        inner_optimal_prompt = copy.copy(prompt)
        tabu_list = create_tabulist(tokenizer, prompt_tokens)
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
                acc, reward = self.tester.forward(test_loader, temp_prompt)
                performance = acc if not reward_driven else reward
                num_iterations += 1

                if performance/max_performance < self.threshold:
                    # cancel this masking
                    colorful_print(f"Cancel, as {performance} is smaller than {max_performance} under threshold: {self.threshold}", fg='red')
                    mask[IND2POS[counter]] = True
                else:
                    max_performance = performance if performance > max_performance else max_performance
                    inner_optimal_prompt = temp_prompt  
                    colorful_print(f"Updated Prompt: {inner_optimal_prompt}", fg='green') 
                    prompt_queues.append((temp_prompt, acc, reward, len(tokenizer.tokenize(temp_prompt)), copy.deepcopy(mask)))
                    
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

            tabu_list = create_tabulist(tokenizer, prompt_tokens)
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
        