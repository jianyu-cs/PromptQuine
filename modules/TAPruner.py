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
                 dataset: str = "", is_mask_lm: str = False, 
                 # used for StyleTransferEvaluator
                 style_classifier_path: str = None, style_batch_size: int = -1, 
                 style_classifier_device_id: int = None, num_samples: int = 1, task_top_k: int = 1):
        assert evaluator_task in ["classification", "style_transfer", "reasoning"] # Delete soon
        
        evaluator_args = {
            "task_lm": task_lm,
            "dataset": dataset,
            "prompt": prompt,
            "num_devices": num_devices
        }

        if evaluator_task == "classification":
            evaluator_args["mode"] = mode
            evaluator_args["is_mask_lm"] = is_mask_lm
        elif evaluator_task == "reasoning":
            
            self.tester = prompt_evaluator(task_lm, dataset, prompt, num_devices)
        elif evaluator_task == "style_transfer":
            assert style_batch_size != -1 and style_classifier_device_id != None and style_classifier_path != None
            self.tester = prompt_evaluator(task_lm, dataset, prompt, mode, num_devices, 
                            style_batch_size, style_classifier_path, style_classifier_device_id, num_samples, task_top_k)
        
        self.tester = prompt_evaluator(**evaluator_args)
        self.task = evaluator_task
        self.task_lm = task_lm
        self.threshold = threshold
    
    def forward(self, prompt: str, test_loader: Any, 
                reward_driven: bool = False,
                fix_prune_order: bool = True) -> List[Tuple]:
        """Conduct TAPruning on the specified prompts
        Args:
            test_loader: DataLoader for other tasks, except Style Transfer with [source_texts, target_labels, ref_texts]
        """
        colorful_print(f"Prompt: {prompt}", fg='blue')
        # Start pruning
        num_iterations = 1 # counting ~ how many prompts we explored.
        # Experimental setups
        tokenizer = AutoTokenizer.from_pretrained(self.task_lm)
        prompt_tokens = tokenizer.tokenize(prompt)
        prompt_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)
        prompt_len = len(prompt_tokens) 
        mask = [True for _ in range(len(prompt_ids))]
        # Record initial prompt's performance
        if self.task != "style_transfer"
            initial_acc, initial_reward = self.tester.forward(test_loader, prompt)
            max_performance = initial_acc if not reward_driven else initial_reward
            # initialize pruned-prompt-queues
            prompt_queues = [(prompt, initial_acc, initial_reward, prompt_len, mask)]
            print(f"Initial Accuracy: {initial_acc}, Initial Reward: {initial_reward}")
        else:
            initial_joint_score, initial_gm, initial_content, initial_style, initial_fluency = \
                                    self.tester.forward(test_loader, prompt)
            max_performance = initial_joint_score
            # initialize pruned-prompt-queues
            prompt_queues = [(prompt, initial_joint_score, initial_gm, initial_content, initial_style, 
                              initial_fluency, prompt_len, mask)] 
        
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
                if self.task != "style_transfer"
                    acc, reward = self.tester.forward(test_loader, temp_prompt)
                    performance = acc if not reward_driven else reward
                    print(f"Accuracy: {acc}, Reward: {reward}")
                else:
                    joint_score, gm, content, style, fluency = self.tester.forward(test_loader, prompt)
                    performance = joint_score 
                    
                num_iterations += 1

                if performance/max_performance < self.threshold:
                    # cancel this masking
                    colorful_print(f"Cancel, as {performance} is smaller than {max_performance} under threshold: {self.threshold}", fg='red')
                    mask[IND2POS[counter]] = True
                else:
                    max_performance = performance if performance > max_performance else max_performance
                    inner_optimal_prompt = temp_prompt  
                    colorful_print(f"Updated Prompt: {inner_optimal_prompt}", fg='green') 
                    if self.task != "style_transfer":
                        prompt_queues.append((temp_prompt, acc, reward, len(tokenizer.tokenize(temp_prompt)), copy.deepcopy(mask)))
                    else:
                        prompt_queues.append((temp_prompt, joint, gm, content, style, fluency, 
                                              len(tokenizer.tokenize(temp_prompt)), copy.deepcopy(mask)))
                    
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
        