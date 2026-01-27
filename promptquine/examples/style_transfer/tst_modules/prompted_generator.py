import os
import time
import json
from typing import Optional, List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, AutoConfig, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from vllm import LLM, SamplingParams

from promptquine.utils import colorful_print


class PromptedGenerator:
    def __init__(
        self,
        model: str,
        end_punct: str = '"',
        pad_token: str = '<|endoftext|>',
        lower_outputs: bool = False,
        control_output_length: bool = False,
        num_devices: int = 1,
    ):
        """
        Initializes the vLLM-based generator.
        """
        self.generator = LLM(model, dtype='half', tensor_parallel_size=num_devices)
        self.tokenizer = AutoTokenizer.from_pretrained(model, pad_token=pad_token)
        # Default configurations
        self.template = "{prompt} \"{sentence_1}\" \""
        self.end_punct = end_punct
        self.lower_outputs = lower_outputs
        self.control_output_length = True

    def _generate_core(
        self,
        formatted_prompts: List[str],
        num_samples: int,
        top_k: Optional[int],
    ) -> List[List[str]]:
        """
        The private core generation logic. It takes a flat list of prompts
        and returns a flat list of generated samples.
        """
        if not formatted_prompts:
            return []

        # 1. Calculate a single global max_new_tokens for the entire batch
        max_len = 0
        if self.control_output_length:
            for prompt_str in formatted_prompts:
                # Using the prompt string itself to estimate length is more accurate
                src_len = len(self.tokenizer(prompt_str)['input_ids'])
                current_max_len = self._get_max_new_tokens(src_len)
                if current_max_len is not None and current_max_len > max_len:
                    max_len = current_max_len
        
        # 2. Set a single temperature for the batch
        temperature = 0 if top_k == 1 else 1.0

        # 3. Create a single SamplingParams object for the batch
        sampling_params = SamplingParams(
            n=num_samples,
            temperature=temperature,
            top_k=top_k if top_k is not None and top_k > 0 else -1,
            max_tokens=max_len if max_len > 0 else 50, # Provide a default value
        )

        # 4. Call the vLLM generator once for all prompts
        outputs = self.generator.generate(formatted_prompts, sampling_params)

        # 5. Perform post-processing on the results
        all_generated_texts = []
        for output in outputs:
            samples = [self.postprocess_output(s.text) for s in output.outputs]
            all_generated_texts.append(samples)
        
        return all_generated_texts

    def sample_generate(
        self,
        prompt: str,
        source_text: str,
        num_samples: int,
        top_k: Optional[int]
    ) -> List[str]:
        """
        Generates samples for a single source text.
        This is now a thin wrapper around _generate_core.
        """
        formatted_prompt = self.template.replace("{prompt}", prompt).replace("{sentence_1}", source_text)

        return self._generate_core([formatted_prompt], num_samples, top_k)

    def sample_generate_batch(
        self,
        prompt: str,
        source_texts: List[str],
        num_samples: int,
        top_k: Optional[int]
    ) -> List[List[str]]:
        """
        Generates samples for a batch of source texts with a single prompt.
        This is now a thin wrapper around _generate_core.
        """
        formatted_prompts = [
            self.template.replace("{prompt}", prompt).replace("{sentence_1}", text)
            for text in source_texts
        ]

        return self._generate_core(formatted_prompts, num_samples, top_k)

    def parallel_sample_generate_batch(
        self,
        input_prompts: List[str],
        source_texts: List[str],
        num_samples: int,
        top_k: Optional[int]
    ) -> List[List[List[str]]]:
        """
        Performs Cartesian product generation (each prompt for each source text).
        This method now prepares the data and reshapes the output from _generate_core.
        """
        if not input_prompts or not source_texts:
            return []

        # 1. Prepare data: Create a flattened list of prompts
        flat_prompts = [
            p_template.replace("{sentence_1}", s_text)
            for p_template in input_prompts
            for s_text in source_texts
        ]
       
        # 2. Call the core logic: Process all combinations at once
        flat_results = self._generate_core(flat_prompts, num_samples, top_k)

        # 3. Reshape the results: Convert the flat list back to a 3D structure
        num_sources = len(source_texts)
        reshaped_outputs = []
        for i in range(len(input_prompts)):
            start_index = i * num_sources
            end_index = start_index + num_sources
            
            # Extract all results belonging to the current prompt_template
            prompt_specific_batch = flat_results[start_index:end_index]
            reshaped_outputs.append(prompt_specific_batch)
            
        return reshaped_outputs

    def _get_max_new_tokens(self, seq_len: int) -> Optional[int]:
        """
        Calculates max new tokens based on self.control_output_length.
        Simplified to remove optional arguments.
        """
        if self.control_output_length:
            return max(int(1.5 * seq_len), seq_len + 10)
        return None

    def postprocess_output(self, text: str) -> str:
        """
        Cleans the raw output text. Simplified to use self attributes directly.
        """
        try:
            end_idx = text.index(self.end_punct)
            text = text[:end_idx]
        except ValueError:
            pass 

        text = text.strip()
        
        try:
            end = text.index('.')
        except ValueError:
            end = len(text)
        try:
            end = min(end, text.index('!'))
        except ValueError:
            pass
        try:
            end = min(end, text.index('?'))
        except ValueError:
            pass

        text = text[:end+1].strip()

        if self.lower_outputs:
            text = text.lower()

        return text
