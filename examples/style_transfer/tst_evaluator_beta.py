from typing import List, Union, Optional
from torch.utils.data import Dataset
import numpy as np
from transformers import pipeline, AutoTokenizer
from bert_score import BERTScorer
import sacrebleu as scb
from .tst_modules import FastClassifier


class TextStyleTransferEvaluator:
    def __init__(self,
                 style_classifier,
                 ppl_lm,
                 style_tokenizer='bert-base-uncased',  # TODO
                 style_batch_size=32,
                 fluency_batch_size=32,
                 device_id=0):
        self.device = device_id
        self.style_classifier = FastClassifier(
            model_name_or_path=style_classifier,
            tokenizer_name='/mnt/workspace/workgroup_dev/zhiqiang/huggingface_models/hub/bert-base-uncased/',
            device=self.device)
        self.style_batch_size = style_batch_size

        self.bert_scorer = BERTScorer('roberta-large',
                                      device=self.device,
                                      rescale_with_baseline=True,
                                      lang='en')

        # Grammaticality model
        self.fluency_classifier = FastClassifier(
            model_name_or_path='/mnt/workspace/workgroup_dev/jianyu/cointegrated',
            device=self.device)
        self.fluency_batch_size = fluency_batch_size

    def evaluate_output(self,
                        source_texts: List[str],
                        output_texts: List[str],
                        target_labels: List[str],
                        ref_texts: Union[List[str], List[List[str]]]):
        
        if isinstance(ref_texts[0], str):
            ref_texts = [[ref] for ref in ref_texts]
            
        output_dataset = ListDataset(output_texts)

        ctc_scores = self.bert_scorer.score(output_texts, source_texts)[2]
        content_scores = [max(s, 0) * 100 for s in ctc_scores.tolist()]
        content_scores = np.array(content_scores)
        content = round(content_scores.mean(), 1)

        style_corrects = []
        style_scores = []
        batch_size = self.style_batch_size
        for i, c in enumerate(self.style_classifier(output_dataset,
                                                    batch_size=batch_size,
                                                    truncation=True)):
            style_corrects.append(int(c['label'] == target_labels[i]))
            style_scores.append(100*((c['label'] == target_labels[i]) * c['score']
                    + (c['label'] != target_labels[i]) * (1 - c['score'])))
        style_corrects = np.array(style_corrects)
        style_scores = np.array(style_scores)
        style = round(100 * style_corrects.mean(), 1)

        fluency_corrects = []
        fluency_label = 'LABEL_0'
        batch_size = self.fluency_batch_size
        for i, c in enumerate(self.fluency_classifier(output_dataset,
                                                      batch_size=batch_size,
                                                      truncation=True)):
            fluency_corrects.append(int(c['label'] == fluency_label))
        fluency_corrects = np.array(fluency_corrects)
        fluency = round(100 * fluency_corrects.mean(), 1)

        joint_scores = content_scores * style_corrects * fluency_corrects
        joint_score = round(joint_scores.mean(), 1)
        
        prejoint_scores = content_scores * style_corrects
        prejoint_score = round(prejoint_scores.mean(), 1)

        gm = np.exp((np.log(content) + np.log(style) + np.log(fluency)) / 3)
        gm = round(gm, 1)

        # print('Computing BLEU')
        bleu_scores = [scb.sentence_bleu(hypothesis=out, references=ref).score
                       for out, ref in zip(output_texts, ref_texts)]
        bleu = round(np.array(bleu_scores).mean(), 1)

        # print('Computing BERTScore')
        bertscore_f1s = self.bert_scorer.score(output_texts, ref_texts)[2]
        bertscore_f1s = np.array([max(b, 0) for b in bertscore_f1s.tolist()])
        bertscore = round(100 * bertscore_f1s.mean(), 1)

        ppl=None

        return (content, style, fluency, joint_score, gm,
                bleu, bertscore, ppl, prejoint_score)


class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)
