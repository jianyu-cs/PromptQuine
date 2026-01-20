from typing import Tuple, List, Union

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from bert_score import BERTScorer


class FastClassifier:
    def __init__(self, model_name_or_path: str, device: str = "cuda", 
                 tokenizer_name: str = None):
        tokenizer_name = tokenizer_name or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path
        ).to(device)
        self.model.eval()
        self.device = device
        self.label_map = {i: f"LABEL_{i}" for i in range(self.model.config.num_labels)}

    def __call__(self, dataset, batch_size: int = 8, truncation: bool = True):
        results = []
        texts = list(dataset)
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors='pt', padding=True, 
                                   truncation=truncation).to(self.device)
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = F.softmax(logits, dim=-1)
            
            for prob in probs:
                score, idx = prob.max(dim=0)
                label = self.label_map[idx.item()]
                results.append({"label": label, "score": score.item()})
        
        return results

class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __getitem__(self, index):
        return self.data_list[index]
    
    def __len__(self):
        return len(self.data_list)


class TextStyleTransferOutputSelector:
    def __init__(
        self,
        style_classifier: str,
        style_batch_size: int,
        device_id: int,
    ):
        self.device = device_id
        self.style_batch_size = style_batch_size
        self.fluency_batch_size = 4
        
        self.style_classifier = self._init_classifier(style_classifier, "google-bert/bert-base-uncased")
        self.fluency_classifier = FastClassifier(
            model_name_or_path="cointegrated/roberta-large-cola-krishna2020",
            device=self.device
        )
        
        self.bert_scorer = BERTScorer(
            'roberta-large',
            device=f"cuda:{self.device}",
            rescale_with_baseline=True,
            lang='en'
        )

    def _init_classifier(self, model_path: str, tokenizer_name: str = "google-bert/bert-base-uncased"):
        if 'facebook' in model_path:
            return pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.device
            )
        else:
            return FastClassifier(
                model_name_or_path=model_path,
                tokenizer_name=tokenizer_name,
                device=self.device
            )

    def _classify_batch(self, texts: List[str], classifier, batch_size: int) -> List[dict]:
        """Batch classification (maintaining generality)"""
        results = []
        for result in classifier(ListDataset(texts), batch_size=batch_size, truncation=True):
            results.append(result)
        return results

    def compute_sample_rewards(self, source_text: str, generated_texts: List[str], 
                              target_label: str) -> Tuple[List[float], List[float], List[float]]:
        """Compute the reward for a single sample"""
        srcs = [source_text] * len(generated_texts)
        
        # Content rewards
        ctc_scores = self.bert_scorer.score(generated_texts, srcs)[2]
        content_rewards = [max(s, 0) * 100 for s in ctc_scores.tolist()]
        
        # Style rewards (prob of classifiers)
        style_prob_results = self._classify_batch(generated_texts, self.style_classifier, 
                                            self.style_batch_size)
        style_rewards = [
            (result['score'] if result['label'] == target_label else 1 - result['score']) * 100
            for result in style_prob_results
        ]
        
        # Sum: Content + Style
        sum_rewards = [(c + s) / 2 for c, s in zip(content_rewards, style_rewards)]
        
        return sum_rewards, content_rewards, style_rewards

    def select_outputs_batch(self, source_texts: List[str], generated_texts: List[List[str]],
                            target_labels: List[str]) -> Tuple[List[str], List[float], List[float], List[float]]:
        """Best-of-N Selection"""
        output_texts, output_rewards, output_contents, output_styles = [], [], [], []
        
        for src, hypos, label in zip(source_texts, generated_texts, target_labels):
            hypos = [h for h in hypos if len(h) > 0]
            
            if not hypos:
                output_texts.append("")
                output_rewards.append(0)
                output_contents.append(0)
                output_styles.append(0)
                continue
            
            rewards, contents, styles = self.compute_sample_rewards(src, hypos, label)
            best_idx = rewards.index(max(rewards))
            
            output_texts.append(hypos[best_idx])
            output_rewards.append(rewards[best_idx])
            output_contents.append(contents[best_idx])
            output_styles.append(styles[best_idx])
        
        return output_texts, output_rewards, output_contents, output_styles

    def evaluate_output(self, source_texts: List[str], output_texts: List[str],
                       target_labels: List[str]) -> Tuple:
        """Evaluate the generated text output (Style, Content, Fluency)"""
        
        # Content Score
        ctc_scores = self.bert_scorer.score(output_texts, source_texts)[2]
        content_scores = np.array([max(s, 0) * 100 for s in ctc_scores.tolist()])
        
        # Style Score
        style_prob_results = self._classify_batch(output_texts, self.style_classifier, 
                                            self.style_batch_size)
        style_corrects = np.array([result['label'] == target_labels[i] 
                                   for i, result in enumerate(style_prob_results)])
        
        # Fluency Score
        fluency_results = self._classify_batch(output_texts, self.fluency_classifier,
                                              self.fluency_batch_size)
        fluency_corrects = np.array([result['label'] == 'LABEL_0' 
                                     for result in fluency_results])
        
        # Post-processing
        content = round(content_scores.mean(), 1)
        style = round(100 * style_corrects.mean(), 1)
        fluency = round(100 * fluency_corrects.mean(), 1)
        
        joint_scores = content_scores * style_corrects * fluency_corrects
        joint_score = round(joint_scores.mean(), 1)
        
        prejoint_score = round((content_scores * style_corrects).mean(), 1)
        gm = round(np.exp((np.log(content) + np.log(style) + np.log(fluency)) / 3), 1)
        
        return (content, style, fluency, joint_score, gm, 0, content, 0, prejoint_score)

