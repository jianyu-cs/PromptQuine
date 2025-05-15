from typing import List, Union, Optional
from torch.utils.data import Dataset
import numpy as np
from transformers import pipeline, AutoTokenizer
from bert_score import BERTScorer
import sacrebleu as scb
from .tst_modules import PromptedGenerator, TextStyleTransferOutputSelector

class PromptedStyleTransferEvaluator:
    def __init__(
        self,
        task_lm: str,
        dataset: str,
        prompt: Optional[str],
        num_devices: int,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.dataset = dataset
        self.task_lm = task_lm
        print("Task LM:", self.task_lm)
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.task_lm, pad_token='<|endoftext|>')
        self._generator = PromptedGenerator(model=task_lm, end_punct='"',
            pad_token='<|endoftext|>', lower_outputs=False, 
            control_output_length=False, num_devices=num_devices)
        
        self._generator.config.pad_token_id = self._tokenizer.pad_token_id

        if prompt is None:
            self.template = self.load_default_template()  # prompt templates
        else:
            self.template = prompt

    def load_default_template(self) -> List[str]:
        template = "{sentence_1} {prompt}"
        return template

    def _format_prompts(
        self,
        prompts: List[str],
        *source_strs: List[List[str]]
    ) -> List[str]:
        """Use str.replace, instead of str.format, for special prompts"""
        return [self.template.replace("{sentence_1}", s_1).replace("{prompt}", prompt)
                for s_1, prompt in zip(source_strs[0], prompts)]

    @torch.no_grad()
    def forward(
        self,
        dataloader: Any,
        prompt: Optional[str]
    ) -> float:
        """Only support vLLM here."""
        num_of_examples = dataloader.dataset.__len__()
        correct_sum = 0
        
        if prompt:
            self.template = prompt
            
        # vLLM configurations
        sampling_params = SamplingParams(temperature=1, top_k=-1, max_tokens=1, allowed_token_ids=self.verbalizer_ids, logprobs=len(self.verbalizer_ids))
        all_outputs = []
        all_inputs = []
        for i, batch in enumerate(dataloader):
            # Dataset Parsing
            inputs = batch['source_texts']  # List
            if type(inputs[0]) != list:
                inputs = [inputs]
            targets = batch['target_labels']  # Tensor
            batch_size = targets.size(0)
            # Prompt Setups
            current_prompts = [prompt for _ in range(batch_size)]
            formatted_templates = self._format_prompts(current_prompts, *inputs)
            # vLLM configurations
            outputs = self._generator.generate(formatted_templates, sampling_params)
            # vLLM Post-processing
            all_inputs.extend(inputs)
            all_outputs.extend(outputs)
            
        test_generations, test_accuracy = process_outputs(all_outputs, all_inputs)
        return test_accuracy, -1