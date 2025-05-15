from transformers import AutoTokenizer, pipeline, AutoConfig, AutoModelForCausalLM
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import tiktoken
import deepspeed
from typing import Optional, List
import torch
import openai
import time
import os
from rlprompt.utils.utils import colorful_print
from vllm import LLM, SamplingParams
    

class PromptedGenerator:
    def __init__(
        self,
        config,
        model: str,
        template: str,
        end_punct: str,
        pad_token: str,
        device_id: int,
        lower_outputs: bool,
        control_output_length: bool,
        mp_size=4,
        batch_size=1
    ):
        # vLLM support
        self.tokenizer = AutoTokenizer.from_pretrained(retrieve_model_link(config, model),
                                            pad_token=pad_token)
        self.generator = LLM(retrieve_model_link(config, model), dtype='float16', tensor_parallel_size=config.num_devices)
         
        self.model = model
        self.template = template
        self.batch_size=batch_size
        self.end_punct = end_punct
        self.lower_outputs = lower_outputs
        self.device_id = device_id
        self.control_output_length = control_output_length

    def _get_max_new_tokens(
        self,
        seq_len: int,
        control_output_length: Optional[bool] = None
    ) -> Optional[int]:
        if control_output_length is None:
            control_output_length = self.control_output_length
        if control_output_length:
            # This hack tends to speed up generation compared to default
            return max(1.5 * seq_len, seq_len + 10)
        else:
            return None

    def sample_generate(
        self,
        prompt: str,
        source_text: str,
        num_samples: int,
        top_k: Optional[int],
        top_p: float,
        lower_outputs: Optional[bool] = None,
        control_output_length: Optional[bool] = None,
        **kwargs
    ) -> List[str]:
        
        # Used for controlling output length
        formatted_template = self.template.replace("{prompt}", prompt).replace("{sentence_1}", source_text)

        src_len = len(self.tokenizer(source_text)['input_ids'])
        max_new_token = self._get_max_new_tokens(
            src_len, control_output_length=control_output_length)

        sampling_params = SamplingParams(temperature=1, top_k=top_k, max_tokens=max_new_token, n=num_samples)
        outputs = self.generator.generate([formatted_template], sampling_params)

        all_generated_texts = []
        for output in outputs:
            generated_texts = []
            for i in range(len(output.outputs)):
                generated_texts.append(self.postprocess_output(output.outputs[i].text))

            all_generated_texts.append(generated_texts)
        return all_generated_texts
    
    def parallel_sample_generate_batch(
        self,
        input_prompts: List[str],
        source_texts: List[str],
        num_samples: int,
        top_k: Optional[int],
        top_p: float,
        lower_outputs: Optional[bool] = None,
        control_output_length: Optional[bool] = None,
        **kwargs
    ) -> List[List[str]]:
        all_whole_generated_texts = []
                    
        prompts = []
        whole_max_new_tokens = []
        
        prompt_separate_indices = []

        separate_len = len(source_texts)
        for prompt_index, input_prompt in enumerate(input_prompts):
            max_lengths = []
            for i, source_text in enumerate(source_texts):
                
                prompts.append(input_prompt.replace("{prompt}", input_prompt).replace("{sentence_1}", source_text))

                src_len = len(self.tokenizer(source_text)['input_ids'])
                max_new_tokens = self._get_max_new_tokens(
                                    src_len, control_output_length=control_output_length)
                max_lengths.append(max_new_tokens)

            max_new_token=max(max_lengths)
            whole_max_new_tokens.append(max_new_token)
            prompt_separate_indices.append(prompt_index * len(source_texts))
        
        max_new_token = max(whole_max_new_tokens)
        # no sampled slicing, top_k
        if top_k == 1:
            temperature = 0

        else:
            temperature = 1

        sampling_params = SamplingParams(temperature=temperature, top_k=top_k, max_tokens=max_new_token, n=num_samples)
        outputs = self.generator.generate(prompts, sampling_params)
        
        for _ in prompt_separate_indices:
            all_generated_texts = []
            for output in outputs[_:_+len(source_texts)]:
                generated_texts = []
                for i in range(len(output.outputs)):
                    generated_texts.append(self.postprocess_output(output.outputs[i].text))
                all_generated_texts.append(generated_texts)
            all_whole_generated_texts.append(all_generated_texts)
        
        return all_whole_generated_texts
    
    def sample_generate_batch(
        self,
        prompt: str,
        source_texts: List[str],
        num_samples: int,
        top_k: Optional[int],
        top_p: float,
        lower_outputs: Optional[bool] = None,
        control_output_length: Optional[bool] = None,
        **kwargs
    ) -> List[List[str]]:
        
        all_generated_texts = []
        prompts = []
        max_lengths = []

        for i, source_text in enumerate(source_texts):
            prompts.append(self.template.replace("{prompt}", prompt).replace("{sentence_1}", source_text))

            src_len = len(self.tokenizer(source_text)['input_ids'])
            max_new_tokens = self._get_max_new_tokens(
                                src_len, control_output_length=control_output_length)
            max_lengths.append(max_new_tokens)

        max_new_token=max(max_lengths)
        if top_k == 1:
            temperature = 0
        else:
            temperature = 1

        sampling_params = SamplingParams(temperature=temperature, top_k=top_k, max_tokens=max_new_token, n=num_samples)
        outputs = self.generator.generate(prompts, sampling_params)

        for output in outputs:
            generated_texts = []
            for i in range(len(output.outputs)):
                generated_texts.append(self.postprocess_output(output.outputs[i].text))
            all_generated_texts.append(generated_texts)
        
        return all_generated_texts

    def postprocess_output(
        self,
        text: str,
        end_punct: Optional[str] = None,
        lower_outputs: Optional[bool] = None
    ) -> str:
        if end_punct is None:
            end_punct = self.end_punct
        if lower_outputs is None:
            lower_outputs = self.lower_outputs

        try:
            end = text.index(end_punct)
        except ValueError:
            end = len(text)
        text = text[:end].strip()

        try:
            end = text.index('.')
        except ValueError:
            end = len(text)
        try:
            end = min(end, text.index('!'))
        except ValueError:
            end = end
        try:
            end = min(end, text.index('?'))
        except ValueError:
            end = end

        text = text[:end+1].strip()
        if lower_outputs:
            text = text.lower()

        return text
