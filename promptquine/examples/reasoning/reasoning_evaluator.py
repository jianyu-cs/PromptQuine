import re
import sys
import pdb
import copy
from typing import Optional, Tuple, List, Any, Dict

import hydra
import numpy as np
import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          AutoModelForMaskedLM)
from vllm import LLM, SamplingParams


def get_answer(sentence):
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    pred_answer = float(pred[-1])
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer

def process_outputs(outputs: Any, test_data: List[Dict], dataset: str):
    assert len(outputs) == len(test_data), "outputs and training data should have the same length"
    correct = 0
    dataset_mapping = {"gsm8k": "answer", "mawps": "ans"}
    for i in range(len(outputs)):
        generations = []
        ans_pred_dict = {}
        for j in range(len(outputs[i].outputs)):
            generation = {
                "generation": outputs[i].outputs[j].text,
                "answer_pred": get_answer(outputs[i].outputs[j].text),
            }
            generations.append(generation)
            if generation["answer_pred"] not in ans_pred_dict:
                ans_pred_dict[generation["answer_pred"]] = 1
            else:
                ans_pred_dict[generation["answer_pred"]] += 1
        sorted_ans_pred = sorted(ans_pred_dict, reverse=True)
        pred_answer = sorted_ans_pred[0]
        if pred_answer == float(test_data[i][dataset_mapping[dataset]]):
            correct += 1
    accuracy = correct / len(test_data)
    
    return accuracy

class PromptedReasoningEvaluator:
    def __init__(
        self,
        task_lm: str,
        dataset: str,
        prompt: Optional[str],
        num_devices: int,
    ):
        super().__init__()
        self.dataset = dataset
        self.task_lm = task_lm
        print("Task LM:", self.task_lm)
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.task_lm, pad_token='<|endoftext|>')
        self._generator = LLM(task_lm, tensor_parallel_size=num_devices, dtype="half")
        # self._generator.config.pad_token_id = self._tokenizer.pad_token_id

        self.template = prompt or "{sentence_1} {prompt}"

    def _render_template(self, tmpl: str, values: dict) -> str:
        for k, v in values.items():
            tmpl = tmpl.replace(f"{{{k}}}", v)
        return tmpl

    def _format_prompts(
        self,
        prompts: List[str],
        use_internal_template: bool,
        *source_strs: List[List[str]],
    ) -> List[str]:
        """Format prompts without changing self.template"""

        results = []

        dataset = self.dataset
        dataset_mapping = {"gsm8k": "instruction", "mawps": "original_text"}
        slots = ["sentence_1"]

        for idx, prompt in enumerate(prompts):
            # Choose template
            tmpl = str(self.template) if use_internal_template else str(prompt)

            # Build slot values
            values = {}
            for slot, source in zip(slots, source_strs):
                values[slot] = source[idx][dataset_mapping[dataset]]

            if use_internal_template:
                values["prompt"] = prompt

            # Render
            results.append(self._render_template(tmpl, values))

        return results

    @torch.no_grad()
    def forward(
        self,
        data: List[Dict],
        prompt: Optional[str],
        **kwargs
    ) -> float:
        """Only support vLLM here."""
        source_texts = data # TODO
        # Successive Halving
        sh_value = None
        valid_stages = ["stage_1", "stage_2"]
        if "successive_halving" in kwargs:
            sh_value = kwargs["successive_halving"]
            
            if sh_value not in valid_stages and sh_value is not None:
                raise ValueError(
                    f"Invalid successive_halving value: '{sh_value}'. "
                    f"Expected one of {valid_stages}"
                )
        if sh_value in valid_stages:
            half_num_texts = len(data) // 2
            if sh_value == "stage_1":
                source_texts = data[:half_num_texts]
            else:
                source_texts = data[half_num_texts:]
        # Main Func
        num_of_examples = len(source_texts)
        correct_sum = 0
        
        if prompt:
            self.template = prompt
            
        # vLLM configurations
        stop_tokens = ["<|endoftext|>", "Question"]
        sampling_params = SamplingParams(temperature=0., max_tokens=512, n=1, stop=stop_tokens)
        # Dataset Parsing
        temp_inputs = [source_texts]
        # Prompt Setups
        current_prompts = [prompt for _ in range(num_of_examples)]
        formatted_templates = self._format_prompts(current_prompts, False, *temp_inputs)
        # vLLM inference
        outputs = self._generator.generate(formatted_templates, sampling_params)
        # Post-hoc calculation
        eval_accuracy = process_outputs(outputs, source_texts, self.dataset)
        # Organize results
        result = {
            "accuracy": eval_accuracy,
            "reward": eval_accuracy
        }
        return result

    @torch.no_grad()
    def forward_batch(
        self,
        data: List[Dict],
        prompts: List[str],
        **kwargs
    ) -> Dict:
        """
        Batch evaluation using vLLM for multiple prompts.

        Args:
            data: List[Dict].
            prompts: List of prompt strings to be evaluated in batch.

        Returns:
            results_list: List[Dict], all numbers we care for a specific prompt eval.
        """
        # Successive Halving
        sh_value = None
        valid_stages = ["stage_1", "stage_2"]
        if "successive_halving" in kwargs:
            sh_value = kwargs["successive_halving"]
            
            if sh_value not in valid_stages and sh_value is not None:
                raise ValueError(
                    f"Invalid successive_halving value: '{sh_value}'. "
                    f"Expected one of {valid_stages}"
                )
        if sh_value in valid_stages:
            half_num_texts = len(data) // 2
            if sh_value == "stage_1":
                source_texts = data[:half_num_texts]
            else:
                source_texts = data[half_num_texts:]
        # Main Func
        correct_sum = 0
        results_list = []
        num_of_examples = len(source_texts)
        # vLLM configurations
        stop_tokens = ["<|endoftext|>", "Question"]
        sampling_params = SamplingParams(temperature=0., max_tokens=512, n=1, stop=stop_tokens)
        # Dataset Parsing
        temp_inputs = [source_texts]
        # Extended prompts (batch)
        expanded_prompts = []
        for prompt in prompts:
            current_prompts = [prompt for _ in range(num_of_examples)]
            formatted = self._format_prompts(current_prompts, False, *temp_inputs)
            if isinstance(formatted, list):
                expanded_prompts.extend(copy.deepcopy(formatted))
            else:
                expanded_prompts.append(str(formatted))
        # vLLM inference
        outputs = self._generator.generate(expanded_prompts, sampling_params)
        # Post-hoc calculation
        grouped_outputs = [
            outputs[i : i + num_of_examples]
            for i in range(0, len(outputs), num_of_examples)
        ]
        for grouped_output in grouped_outputs:
            eval_accuracy = process_outputs(grouped_output, source_texts, self.dataset)
            result = {
                "accuracy": eval_accuracy,
                "reward": eval_accuracy
            }
            results_list.append(result)
        return results_list