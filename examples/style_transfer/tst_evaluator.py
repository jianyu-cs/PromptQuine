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
        style_batch_size: int,
        style_classifier_path: str,
        style_classifier_device_id: int,
        # generation hyper-parameters
        num_samples: int,
        task_top_k: int,
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
        self._selector = TextStyleTransferOutputSelector(style_classifier=style_classifier_path,
                style_batch_size=style_batch_size, device_id=style_classifier_device_id)
        
        self._generator.config.pad_token_id = self._tokenizer.pad_token_id

        if prompt is None:
            self.template = self.load_default_template()  # prompt templates
        else:
            self.template = prompt
        
        # generation decoding hyper-parameters
        self.num_samples = num_samples
        self.task_top_k = task_top_k

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
        data_lists: Any,
        prompt: Optional[str]
    ) -> float:
        """Only support vLLM here."""
        num_of_examples = len(data_lists[0])
        correct_sum = 0
        
        if prompt:
            self.template = prompt
            self._generator.template = prompt
            
        source_texts = data_lists[0]
        target_labels = data_lists[1]
        ref_texts = data_lists[2] # placeholder
            
        # vLLM configurations
        sampling_params = SamplingParams(temperature=1, top_k=-1, max_tokens=1, allowed_token_ids=self.verbalizer_ids, logprobs=len(self.verbalizer_ids))
        generated_texts = self._generator.sample_generate_batch(
                prompt, source_texts, self.num_samples, self.task_top_k, 1.0)
        output_texts, rewards, contents, styles = self._selector.select_outputs_batch(
                source_texts, generated_texts, target_labels)
        (content, style, fluency, joint_score, gm, bleu, bertscore, ppl, _) = \
                    self._selector.evaluate_output(source_texts, output_texts,
                              target_labels, ref_texts)
        print('Printing Aggregate Scores')
        print('Content:', content, 'Style:', style, 'Fluency:', fluency,
              'Joint:', joint_score, 'GM:', gm, 'BLEU:', bleu,
              'BERTScore:', bertscore, 'PPL:', ppl, 
              'Reward:', mean(rewards), 'Content_score:', mean(contents),
              'Style_score:', mean(styles)
            )
        return joint_score, gm, content, style, fluency