from statistics import mean
from typing import List, Union, Optional, Any, Dict

import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import pipeline, AutoTokenizer
from bert_score import BERTScorer
import sacrebleu as scb
from promptquine.examples.style_transfer.tst_modules import PromptedGenerator, TextStyleTransferOutputSelector


class PromptedStyleTransferEvaluator:
    def __init__(
        self,
        task_lm: str,
        dataset: str,
        prompt: Optional[str],
        num_devices: int,
        style_batch_size: int,
        style_classifier_path: str,
        style_classifier_device_id: int,
        num_samples: int,
        task_top_k: int,
    ):
        super().__init__()
        self.mode = 'vLLM'
        self.dataset = dataset
        self.task_lm = task_lm
        
        # Decoding parameters
        self.num_samples = num_samples
        self.task_top_k = task_top_k
        
        # Initializing modules
        self._tokenizer = AutoTokenizer.from_pretrained(
            task_lm, pad_token='<|endoftext|>')
        self._generator = PromptedGenerator(
            model=task_lm,
            end_punct='"',
            pad_token='<|endoftext|>',
            lower_outputs=False,
            control_output_length=False,
            num_devices=num_devices,
        )
        self._selector = TextStyleTransferOutputSelector(
            style_classifier=style_classifier_path,
            style_batch_size=style_batch_size,
            device_id=style_classifier_device_id
        )
        self.template = prompt or "{sentence_1} {prompt}"

    @torch.no_grad()
    def forward(self, data_lists: Any, prompt: Optional[str], **kwargs) -> dict:
        """Complete evaluation."""
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
        # Standard forward generation
        source_texts, target_labels = data_lists
        
        if prompt:
            self.template = prompt
            self._generator.template = prompt

        if sh_value in valid_stages:
            half_num_texts = len(source_texts) // 2
            if sh_value == "stage_1":
                source_texts = source_texts[:half_num_texts]
                target_labels = target_labels[:half_num_texts]
            else:
                source_texts = source_texts[half_num_texts:]
                target_labels = target_labels[half_num_texts:]

        # Generation
        generated_texts = self._generator.sample_generate_batch(
            prompt, source_texts, self.num_samples, self.task_top_k
        )
        
        # Best-of-N Selection
        output_texts, rewards, contents, styles = self._selector.select_outputs_batch(
            source_texts, generated_texts, target_labels
        )
        
        # Evaluation
        content, style, fluency, joint_score, gm, bleu, bertscore, ppl, _ = \
            self._selector.evaluate_output(source_texts, output_texts, target_labels)
        
        result = {
            'content': content,
            'style': style,
            'fluency': fluency,
            'joint_score': joint_score,
            'gm': gm,
            'bleu': bleu,
            'bertscore': bertscore,
            'ppl': ppl,
            'content_score': mean(contents),
            'style_score': mean(styles),
        }
        
        print('Aggregate Scores:', result)
        return result

    @torch.no_grad()
    def forward_batch(
        self,
        data_lists: Any,
        prompts: List[str],
        **kwargs
    ) -> Dict:
        """
        Batch evaluation using vLLM for multiple prompts.

        Args:
            data_lists: Data lists yielding batches with 'source_texts' and 'class_labels'.
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
        # Standard forward generation
        results_list = []
        source_texts, target_labels = data_lists

        if sh_value in valid_stages:
            half_num_texts = len(source_texts) // 2
            if sh_value == "stage_1":
                source_texts = source_texts[:half_num_texts]
                target_labels = target_labels[:half_num_texts]
            else:
                source_texts = source_texts[half_num_texts:]
                target_labels = target_labels[half_num_texts:]
        
        # Generation
        generated_texts_list = self._generator.parallel_sample_generate_batch(
            prompts, source_texts, self.num_samples, self.task_top_k
        )
        # Best-of-N Selection
        for generated_texts in generated_texts_list:
            output_texts, rewards, contents, styles = self._selector.select_outputs_batch(
                source_texts, generated_texts, target_labels
            )
            # Evaluation
            content, style, fluency, joint_score, gm, bleu, bertscore, ppl, _ = \
                self._selector.evaluate_output(source_texts, output_texts, target_labels)
            
            result = {
                'content': content,
                'style': style,
                'fluency': fluency,
                'joint_score': joint_score,
                'gm': gm,
                'bleu': bleu,
                'bertscore': bertscore,
                'ppl': ppl,
                'content_score': mean(contents),
                'style_score': mean(styles),
            }
            results_list.append(result)
        return results_list



