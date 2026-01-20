import sys
import copy
from typing import Optional, Tuple, List, Any, Dict

import hydra
import numpy as np
import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          AutoModelForMaskedLM)
from vllm import LLM, SamplingParams

from promptquine.utils.classification import load_verbalizers


DATASET_SLOTS = {
    "default": ["sentence_1"],
    "snli": ["sentence_1", "sentence_2"],
    "piqa": ["question_1", "option_1", "option_2"],
}

class PromptedClassificationEvaluator:
    def __init__(
        self,
        task_lm: str,
        is_mask_lm: Optional[bool],
        dataset: str,
        prompt: Optional[str],
        mode: str = "vLLM",
        num_devices: int = 1
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.mode = mode
        self.dataset = dataset
        self.task_lm = task_lm
        print("Task LM:", self.task_lm)
        if is_mask_lm is None: 
            # If False, then treat as left-to-right LM
            self.is_mask_lm = True if 'bert' in self.task_lm else False
        else:
            self.is_mask_lm = is_mask_lm  
        if self.is_mask_lm:
            self._tokenizer = AutoTokenizer.from_pretrained(self.task_lm,
                                truncation_side="left")
            self._generator = (AutoModelForMaskedLM
                               .from_pretrained(self.task_lm)
                               .to(self.device))
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.task_lm, pad_token='<|endoftext|>')
            if mode == "HF":
                self._generator = (AutoModelForCausalLM
                                   .from_pretrained(self.task_lm)
                                   .to(self.device))
                self._generator.config.pad_token_id = self._tokenizer.pad_token_id
            elif mode == "vLLM":
                self._generator = LLM(task_lm, tensor_parallel_size=num_devices, 
                                      dtype="half")

        verbalizer_texts = load_verbalizers(self.task_lm, dataset)
        # Encode each one individually
        verbalizers = [self._tokenizer.encode(v, add_special_tokens=False) for v in verbalizer_texts]

        self.verbalizer_ids = [self._tokenizer.convert_tokens_to_ids(v)
                               for v in load_verbalizers(self.task_lm, dataset)]
        if prompt is None:
            self.template = self.load_default_template()  # prompt templates
        else:
            self.template = prompt

    # Adapted from
    # https://huggingface.co/docs/transformers/v4.21.1/en/task_summary#masked-language-modeling
    def _get_mask_token_index(self, input_ids: torch.Tensor) -> np.ndarray:
        mask_token_index = torch.where(
            input_ids == self._tokenizer.mask_token_id)[1]
        return mask_token_index
    
    def _render_template(self, tmpl: str, values: dict) -> str:
        for k, v in values.items():
            tmpl = tmpl.replace(f"{{{k}}}", v)
        return tmpl

    def load_default_template(self) -> List[str]:
        if self.is_mask_lm:
            template = "{sentence_1} {prompt} <mask> ."
        else:
            # Template for left-to-right LMs like GPT-2
            template = "{sentence_1} {prompt}"

        return template

    @torch.no_grad()
    def _get_logits(
        self,
        texts: List[str]
    ) -> torch.Tensor:
        # for MLM, add mask token
        batch_size = len(texts)
        encoded_inputs = self._tokenizer(texts, padding='longest',
                                         truncation=True, return_tensors="pt",
                                         add_special_tokens=True)

        if self.is_mask_lm:
            token_logits = self._generator(
                **encoded_inputs.to(self.device)).logits
            mask_token_indices = \
                self._get_mask_token_index(encoded_inputs['input_ids'])
            out_logits = token_logits[range(batch_size), mask_token_indices, :]
        else:
            token_logits = self._generator(
                **encoded_inputs.to(self.device)).logits
            input_lengths = encoded_inputs['attention_mask'].sum(dim=1)
            out_logits = token_logits[range(batch_size), input_lengths - 1, :]

        return out_logits

    def _format_prompts(
        self,
        prompts: List[str],
        use_internal_template: bool,
        *source_strs: List[List[str]],
    ) -> List[str]:
        """Format prompts without changing self.template"""

        results = []

        dataset = self.dataset
        slots = DATASET_SLOTS.get(dataset, DATASET_SLOTS["default"])

        if dataset in ["snli", "piqa"]:
            source_strs = source_strs[0]

        mask_token = self._tokenizer.mask_token if self.is_mask_lm else None

        for idx, prompt in enumerate(prompts):
            # Choose template
            tmpl = str(self.template) if use_internal_template else str(prompt)

            # Build slot values
            values = {}
            for slot, source in zip(slots, source_strs):
                values[slot] = source[idx]

            if use_internal_template:
                values["prompt"] = prompt

            if mask_token is not None:
                values["mask_token"] = mask_token

            # Render
            results.append(self._render_template(tmpl, values))

        return results

    def forward(
        self,
        dataloader: Any,
        prompt: Optional[str],
        **kwargs
    ) -> Dict:
        """
        Batch evaluation using vLLM/HF for single prompt.

        Args:
            dataloader: Data loader yielding batches with 'source_texts' and 'class_labels'.
            prompts: List of prompt strings to be evaluated in batch.

        Returns:
            accuracy: float, accuracies for each prompt.
            mean_gap: torch.Tensor, average margin gap rewards for each prompt.
        """
        num_of_examples = dataloader.dataset.__len__()
        original_template = str(self.template)
        correct_sum = 0
        gap_sums = []
        
        if prompt:
            self.template = prompt
            
        if self.mode == 'vLLM':
            # vLLM configurations
            sampling_params = SamplingParams(temperature=1, top_k=-1, max_tokens=1, allowed_token_ids=self.verbalizer_ids, logprobs=len(self.verbalizer_ids))

        for i, batch in enumerate(dataloader):
            # Dataset Parsing
            inputs = batch['source_texts']  # List
            if type(inputs[0]) != list:
                inputs = [inputs]
            targets = batch['class_labels']  # Tensor
            batch_size = targets.size(0)
            # Prompt Setups
            current_prompts = [prompt for _ in range(batch_size)]
            formatted_templates = self._format_prompts(current_prompts, True, *inputs)
            # Mode: Huggingface
            if self.mode == 'HF':
                all_logits = self._get_logits(formatted_templates)
                class_probs = torch.softmax(all_logits[:, self.verbalizer_ids], -1)
                # Rewards - RLPrompt
                label_probs = class_probs[range(batch_size), targets]
                # [batch_size, 1]
                not_label_probs = torch.where(
                    class_probs == label_probs.unsqueeze(1),
                    torch.Tensor([-1]).to(self.device), class_probs)
                # [batch_size, num_classes]
                max_not_label_probs, _ = torch.max(not_label_probs, -1)
                gap = label_probs - (max_not_label_probs) #- max_not_label_probs
                # compute gaps
                correct = (gap > 0).long()
                gap_rewards = gap * (200 * correct \
                                 + 180 * (1 - correct))
                
                gap_sums.append(gap_rewards.mean())
                
                # Get labels
                predicted_labels = torch.argmax(class_probs, dim=-1)
                label_agreement = torch.where(
                    targets.cuda() == predicted_labels, 1, 0)
                # Compute accuracy
                correct_sum += label_agreement.sum().item()
            elif self.mode == 'vLLM':
                # vLLM configurations
                outputs = self._generator.generate(formatted_templates, sampling_params)
                # vLLM Post-processing
                for index, output in enumerate(outputs):
                    full_label_probs = []
                    if output.outputs[0].logprobs == []:
                        print("index:", output.outputs)
                        continue
                    output_logprobs = output.outputs[0].logprobs[0]
                    target_nll = -output_logprobs[self.verbalizer_ids[targets[index].item()]].logprob
                    full_label_probs.append(np.exp(-output_logprobs[self.verbalizer_ids[targets[index]]].logprob))
                    
                    incorrect_ids = [verbalizer for verbalizer in self.verbalizer_ids if verbalizer != self.verbalizer_ids[targets[index].item()]]
                    other_nlls = []
                    for incorrect_id in incorrect_ids:
                        other_nlls.append(-output_logprobs[incorrect_id].logprob)
                        full_label_probs.append(np.exp(-output_logprobs[incorrect_id].logprob))
                    other_nll = min(other_nlls)
                    full_label_probs = [label/sum(full_label_probs) for label in full_label_probs]
                    full_label_probs_2index = [self.verbalizer_ids[targets[index].item()]] + incorrect_ids
                    
                    predict_label = full_label_probs_2index[np.argmin(full_label_probs)]
                    if self.verbalizer_ids[targets[index].item()] == predict_label:
                        correct_sum += 1
                    # Rewards - RLPrompt
                    marginprob_i = np.exp(-target_nll) - np.exp(-other_nll)
                    if marginprob_i > 0:
                        marginprob_i *= 200
                    else:
                        marginprob_i *= 180
                    
                    gap_sums.append(marginprob_i)

        accuracy = correct_sum/num_of_examples
        self.template = str(original_template)
        result = {
            'accuracy': accuracy,
            'reward': torch.mean(torch.tensor(gap_sums)).item()
        }
        return result
    
    def forward_batch(
        self,
        dataloader: Any,
        prompts: List[str],
        **kwargs
    ) -> Dict:
        """
        Batch evaluation using vLLM for multiple prompts.

        Args:
            dataloader: Data loader yielding batches with 'source_texts' and 'class_labels'.
            prompts: List of prompt strings to be evaluated in batch.

        Returns:
            results_list: List[Dict], accuracy & reward.
        """
        num_of_examples = dataloader.dataset.__len__()
        num_prompts = len(prompts)

        # intialize counters for all the prompts
        correct_counts = [0 for _ in range(num_prompts)]
        gap_sums_list = [[] for _ in range(num_prompts)]

        # vLLM configurations (vLLM Only)
        all_targets = []
        all_templates = []
        for batch in dataloader:
            inputs = batch['source_texts']
            if type(inputs[0]) != list:
                inputs = [inputs]
            targets = batch['class_labels']
            if isinstance(targets, torch.Tensor):
                targets = targets.tolist()
            all_targets.extend(targets)
            all_templates.extend(inputs)

        sampling_params = SamplingParams(temperature=1, top_k=-1, max_tokens=1, allowed_token_ids=self.verbalizer_ids, logprobs=len(self.verbalizer_ids))

        expanded_prompts = []
        for prompt in prompts:
            current_prompts = [prompt for _ in range(num_of_examples)]
            formatted = self._format_prompts(current_prompts, False, *all_templates)
            if isinstance(formatted, list):
                expanded_prompts.extend(copy.deepcopy(formatted))
            else:
                expanded_prompts.append(str(formatted))

        outputs = self._generator.generate(expanded_prompts, sampling_params)
        # Iterate over the outputs and assign results to the corresponding prompt statistics
        for index, output in enumerate(outputs):
            prompt_idx = index // num_of_examples
            sample_idx = index % num_of_examples
            # Use prompt_idx to group and aggregate accuracy, gap, etc.
            full_label_probs = []
            if output.outputs[0].logprobs == []:
                print("index:", output.outputs)
                continue
            output_logprobs = output.outputs[0].logprobs[0]
            target_nll = -output_logprobs[self.verbalizer_ids[all_targets[sample_idx]]].logprob
            full_label_probs.append(np.exp(-output_logprobs[self.verbalizer_ids[all_targets[sample_idx]]].logprob))
            
            incorrect_ids = [verbalizer for verbalizer in self.verbalizer_ids if verbalizer != self.verbalizer_ids[all_targets[sample_idx]]]
            other_nlls = []
            for incorrect_id in incorrect_ids:
                other_nlls.append(-output_logprobs[incorrect_id].logprob)
                full_label_probs.append(np.exp(-output_logprobs[incorrect_id].logprob))
            other_nll = min(other_nlls)
            full_label_probs = [label/sum(full_label_probs) for label in full_label_probs]
            full_label_probs_2index = [self.verbalizer_ids[all_targets[sample_idx]]] + incorrect_ids
            
            predict_label = full_label_probs_2index[np.argmin(full_label_probs)]
            if self.verbalizer_ids[all_targets[sample_idx]] == predict_label:
                correct_counts[prompt_idx] += 1
            # Rewards - RLPrompt
            marginprob_i = np.exp(-target_nll) - np.exp(-other_nll)
            if marginprob_i > 0:
                marginprob_i *= 200
            else:
                marginprob_i *= 180
    
            gap_sums_list[prompt_idx].append(marginprob_i)
        acc_list = [correct_count/num_of_examples for correct_count in correct_counts]
        rewards_list = [torch.mean(torch.tensor(gap_sums)) for gap_sums in gap_sums_list]
        
        results_list = [
            {'accuracy': acc, 'reward': reward.item()}
            for acc, reward in zip(acc_list, rewards_list)
        ]
        return results_list



