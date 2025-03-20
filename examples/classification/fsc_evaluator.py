"""
Adapted from RLPrompt: https://aclanthology.org/2022.emnlp-main.222/
"""

import sys
sys.path.append('..')
import hydra
import numpy as np
import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          AutoModelForMaskedLM)
from vllm import LLM, SamplingParams

from typing import Optional, Tuple, List


def load_verbalizers(task_lm: str, dataset: str) -> List[str]:
    if task_lm in ['FacebookAI/roberta-large', 'openai-community/gpt2']:
        if dataset == 'sst-2':
            return ['Ġterrible', 'Ġgreat']
        elif dataset == 'subj':
            return ['Ġsubjective', 'Ġobjective']
        elif dataset == 'agnews':
            return ['ĠWorld', 'ĠSports', 'ĠBusiness', 'ĠTech']
        elif dataset == 'yelp-5':
            return ['Ġterrible', 'Ġbad', 'Ġneutral', 'Ġgood', 'Ġgreat']
        elif dataset == 'yahoo':
            return ['Ġculture', 'Ġscience', 'Ġhealth', 'Ġeducation', 'Ġcomputer', 'Ġsports', 'Ġbusiness', 'Ġmusic', 'Ġfamily', 'Ġpolitics']
        elif dataset == 'snli':
            return ['ĠYes', 'ĠUnknown', 'ĠNo']
        elif dataset == "piqa":
            return ['ĠA', 'ĠB']
    elif task_lm == 'google/gemma-7b-it':
        if dataset == 'sst-2':
            return ['▁terrible', '▁great']
        elif dataset == 'subj':
            return ['▁subjective', '▁objective']
        elif dataset == 'agnews':
            return ['▁World', '▁Sports', '▁Business', '▁Tech']
        elif dataset == 'yelp-5':
            return ['▁terrible', '▁bad', '▁neutral', '▁good', '▁great']
        elif dataset == 'yahoo':
            return ['▁culture', '▁science', '▁health', '▁education', '▁computer', '▁sports', '▁business', '▁music', '▁family', '▁politics']
        elif dataset == 'snli':
            return ['▁Yes', '▁Unknown', '▁No']
        elif dataset == "piqa":
            return ['▁A', '▁B']
    elif task_lm in ['meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3-8B', 
                     'meta-llama/Meta-Llama-3-70B', 'meta-llama/Meta-Llama-3-70B-Instruct']:
        if dataset == 'sst-2':
            return ['Ġterrible', 'Ġgreat']
        elif dataset == 'subj':
            return ['Ġsubjective', 'Ġobjective']
        elif dataset == 'agnews':
            return ['ĠWorld', 'ĠSports', 'ĠBusiness', 'ĠTech']
        elif dataset == 'yelp-5':
            return ['Ġterrible', 'Ġbad', 'Ġneutral', 'Ġgood', 'Ġgreat']
        elif dataset == 'yahoo':
            return ['Ġculture', 'Ġscience', 'Ġhealth', 'Ġeducation', 'Ġcomputer', 'Ġsports', 'Ġbusiness', 'Ġmusic', 'Ġfamily', 'Ġpolitics']
        elif dataset == 'snli':
            return ['ĠYes', 'ĠUnknown', 'ĠNo']
        elif dataset == "piqa":
            return ['ĠA', 'ĠB']
        
        
class PromptedClassificationEvaluator:
    def __init__(
        self,
        task_lm: str,
        is_mask_lm: Optional[bool],
        dataset: str,
        prompt: Optional[str],
        mode: str = "vLLM",
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
            elif mode == "vLLM":
                self._generator = LLM(task_lm, tensor_parallel_size=config.num_devices, 
                                      dtype="half")
            self._generator.config.pad_token_id = self._tokenizer.pad_token_id
        self.verbalizers = self._tokenizer.encode(load_verbalizers(self.task_lm, dataset))

        self.verbalizer_ids = [self._tokenizer.convert_tokens_to_ids(v)
                               for v in self.verbalizers]
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
            # self.ensure_exactly_one_mask_token(encoded_inputs) TODO
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
        *source_strs: List[List[str]]
    ) -> List[str]:
        """Use str.replace, instead of str.format, for special prompts"""
        if self.is_mask_lm and self.dataset not in ['snli', 'piqa']:
            mask_token = self._tokenizer.mask_token
            return [self.template.replace("{sentence_1}", s_1).replace("{prompt}", prompt).replace("{mask_token}", 
                    mask_token) for s_1, prompt in zip(source_strs[0], prompts)]
        elif self.dataset not in ['snli', 'piqa']:
            return [self.template.replace("{sentence_1}", s_1).replace("{prompt}", prompt)
                    for s_1, prompt in zip(source_strs[0], prompts)]
        elif self.is_mask_lm and self.dataset == 'snli':
            mask_token = self._tokenizer.mask_token
            return [self.template.replace("{sentence_1}", s_1).replace("{sentence_2}", s_2)
                    .replace("{prompt}", prompt).replace("{mask_token}", mask_token) 
                    for s_1, s_2, prompt in zip(source_strs[0], source_strs[1], prompts)]
        elif self.dataset == 'snli':
            return [self.template.replace("{sentence_1}", s_1).replace("{sentence_2}", s_2)
                    .replace("{prompt}", prompt) for s_1, s_2, prompt in zip(source_strs[0], source_strs[1], prompts)]
        elif self.is_mask_lm and self.dataset == 'piqa':
            mask_token = self._tokenizer.mask_token
            return [self.template.replace("{question_1}", s_1).replace("{option_1}", s_2).replace("{option_2}", s_3)
                    .replace("{prompt}", prompt).replace("{mask_token}", mask_token) 
                    for s_1, s_2, s_3, prompt in zip(source_strs[0], source_strs[1], source_strs[2], prompts)]
        elif self.dataset == 'piqa':
            return [self.template.replace("{question_1}", s_1).replace("{option_1}", s_2)
                    .replace("{option_2}", s_3).replace("{prompt}", prompt)
                    for s_1, s_2, s_3, prompt in zip(source_strs[0], source_strs[1], source_strs[2], prompts)]

    def forward(
        self,
        dataloader: Any,
        prompt: Optional[str]
    ) -> float:
        num_of_examples = dataloader.dataset.__len__()
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
            formatted_templates = self._format_prompts(current_prompts, *inputs)
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
                correct_sum += label_agreement.sum()
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

                    nlls.append(target_nll)
                    incorrect_nlls.append(other_nll)
                    
                    predict_label = full_label_probs_2index[np.argmin(full_label_probs)]
                    if self.verbalizer_ids[targets[index].item()] == predict_label:
                        correct_sum += 1
                # Rewards - RLPrompt
                for i, nll in enumerate(nlls):
                    marginprob_i = np.exp(-nll) - np.exp(-incorrect_nlls[i])
                    if marginprob_i > 0:
                        marginprob_i *= 200
                    else:
                        marginprob_i *= 180
                    
                    gap_sums.append(marginprob_i)
                    
        accuracy = correct_sum/num_of_examples
        return accuracy, torch.mean(torch.tensor(gap_sums))