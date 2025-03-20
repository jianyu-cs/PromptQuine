import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from transformers import pipeline
from bert_score import BERTScorer
from typing import Tuple, List, Union

class TextStyleTransferOutputSelector: 
    def __init__(
        self, 
        style_classifier: str,
        style_tokenizer: str,
        style_batch_size: int,
        device_id: int
    ): 
        self.device = device_id
        if 'facebook' in style_classifier:
            self.style_classifier = pipeline("zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=self.device)#self.device)
            self.nli = True
        else:
            self.style_classifier = pipeline("sentiment-analysis",
                                         model=style_classifier,
                                         tokenizer='/mnt/workspace/workgroup_dev/zhiqiang/huggingface_models/hub/bert-base-uncased/',
                                         device=self.device)#self.device)
            self.nli = False
        self.style_batch_size = style_batch_size
        self.bert_scorer = BERTScorer('roberta-large', 
                                      device=self.device,#self.device, 
                                      rescale_with_baseline=True, 
                                      lang='en')
        
        # Grammaticality model
        self.fluency_classifier = pipeline(
            'text-classification',
            model='/mnt/workspace/workgroup_dev/jianyu/cointegrated', #'cointegrated/roberta-large-cola-krishna2020',
            device=self.device)
        self.fluency_batch_size = 4#fluency_batch_size

    def compute_sample_rewards(
        self, 
        source_text: str, 
        generated_texts: List[str], 
        target_label: str
    ) -> Tuple[List[float]]: 

        srcs = [source_text for _ in generated_texts]
        hypos = generated_texts
        
        ctc_scores = self.bert_scorer.score(hypos, srcs)[2]
        content_rewards = [max(s, 0) * 100 for s in ctc_scores.tolist()]

        # Style probablility reward
        hypo_dataset = ListDataset(hypos)
        batch_size = self.style_batch_size
        style_rewards = []
        if True:
            for i, c in enumerate(self.style_classifier(hypo_dataset,
                                                    batch_size=batch_size, 
                                                    truncation=True)):
                prob = ((c['label'] == target_label) * c['score']
                    + (c['label'] != target_label) * (1 - c['score']))
                style_rewards.append(prob * 100)
                sum_rewards = [(c + s) / 2 \
                       for c, s in zip(content_rewards, style_rewards)]
        #elif nli == True:
        #    for i, c in enumerate(self.style_classifier(hypo_dataset,

        return sum_rewards, content_rewards, style_rewards
        
    def select_outputs_batch(
        self, 
        source_texts: List[str], 
        generated_texts: List[List[str]], 
        target_labels: List[str]
    ) -> Tuple[Union[List[str], List[float]]]:
        output_texts = []
        output_rewards = []
        output_contents = []
        output_styles = []
        # print("PREDICT", len(source_texts),"WOW", len(generated_texts)) 200. 200
        for i,(src, hypos, label) in enumerate(zip(source_texts, generated_texts, 
                                          target_labels)):
            
            hypos = [h for h in hypos if len(h) > 0]
            if len(hypos) == 0:
                output_texts.append("")
                output_rewards.append(0)
                output_contents.append(0)
                output_styles.append(0)
                continue

            rewards, contents, styles = self.compute_sample_rewards(src, hypos, 
                                                                    label)
            max_reward = max(rewards)#torch.tensor(rewards).float().max()
            #print(max_reward)
            top_index = rewards.index(max_reward)#.float())
            #print(top_index)
            #print(hypos[top_index])
            #print("78&&&&&&&&&&")

            output_texts.append(hypos[top_index])
            output_rewards.append(rewards[top_index])
            output_contents.append(contents[top_index])
            output_styles.append(styles[top_index])
                              
        return output_texts, output_rewards, output_contents, output_styles
    # Newly added function: TODO
    def evaluate_output(self,
                        source_texts: List[str],
                        output_texts: List[str],
                        target_labels: List[str],
                        ref_texts: Union[List[str], List[List[str]]]):
        # print(output_texts)
        # print(output_texts)
        if isinstance(ref_texts[0], str):
            ref_texts = [[ref] for ref in ref_texts]
            
        #srcs = [source_text for _ in generated_texts]
        #hypos = output_texts
        print("****")
        print(source_texts[-1])
        print(output_texts[-1])
        
        output_dataset = ListDataset(output_texts)

        # print('Computing Content Preservation')
        #print(output_texts)
        #print(source_texts)
        ctc_scores = self.bert_scorer.score(output_texts, source_texts)[2]
        content_scores = [max(s, 0) * 100 for s in ctc_scores.tolist()]
        content_scores = np.array(content_scores)
        content = round(content_scores.mean(), 1)

        # print('Computing Style Accuracy')
        style_corrects = []
        batch_size = self.style_batch_size
        for i, c in enumerate(self.style_classifier(output_dataset,
                                                    batch_size=batch_size,
                                                    truncation=True)):
            style_corrects.append(int(c['label'] == target_labels[i]))
        style_corrects = np.array(style_corrects)
        style = round(100 * style_corrects.mean(), 1)

        # print('Computing Fluency')
        fluency_corrects = []
        fluency_label = 'LABEL_0'
        batch_size = self.fluency_batch_size
        for i, c in enumerate(self.fluency_classifier(output_dataset,
                                                      batch_size=batch_size,
                                                      truncation=True)):
            fluency_corrects.append(int(c['label'] == fluency_label))
        fluency_corrects = np.array(fluency_corrects)
        fluency = round(100 * fluency_corrects.mean(), 1)

        # print('Computing Joint Score')
        joint_scores = content_scores * style_corrects * fluency_corrects
        joint_score = round(joint_scores.mean(), 1)
        
        prejoint_scores = content_scores * style_corrects
        prejoint_score = round(prejoint_scores.mean(), 1)

        # print('Computing Geometric Avg')
        gm = np.exp((np.log(content) + np.log(style) + np.log(fluency)) / 3)
        gm = round(gm, 1)

        return (content, style, fluency, joint_score, gm,
                0, content, 0, prejoint_score)

class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __getitem__(self, index):
        return self.data_list[index]
    
    def __len__(self):
        return len(self.data_list)
