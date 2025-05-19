import sys
import os
import json
import copy
import datetime
import argparse
import pandas as pd
from itertools import compress
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaTokenizer
from typing import Optional, Union, List, Dict, Any
from tst_evaluator import TextStyleTransferEvaluator
# TODO
sys.path.append("../../")
from modules.TAPruner import TAPruner
from modules.PromptQuinePruner import PromptQuinePruner
from dataset_helper import load_text_style_transfer_test_data, get_style_classifier


def main(args):
    # 1. load classifier
    style_classifier = \
        os.path.join('..', get_style_classifier(args.classifier_setup, args.dataset))
    # 2. load dataset
    base_path="./data"
    ## returning both source_texts for dev and ref_texts for test
    source_texts, target_labels, ref_texts = \
            load_text_style_transfer_test_data(
            args.direction, args.dataset,
            base_path=base_path, max_size=args.num_samples_in_search,
            max_length=None,
            max_length_tokenizer=args.model)
    
    # 1. Load ICL Prompts for Pruning
    direction_mapping = {"1_to_0": "negative", "0_to_1": "positive"}
    prompts_path = f"../../prompts/sentiment_transfer_prompts/"\
            f"few_shot_natural_prompts_{direction_mapping[args.direction]}_{args.ICL_shots}shot.jsonl"

    prompt_dict_list = [] 
    with open(prompts_path, 'r') as prompt_jsons:
        prompt_json_lists = list(prompt_jsons)
    for prompt_json in prompt_json_lists:
        prompt_dict_list.append(json.loads(prompt_json))
    prompt = prompt_dict_list[args.ICL_index]['prompt']
    
    # 2. Setup the Pruner
    if args.pruner == "PromptQuine":
        pass
    elif args.pruner == "TAPruning":
        pruner = TAPruner(TextStyleTransferEvaluator, args.model, "style_transfer", None, "vLLM",
                threshold=0.96, dataset = args.dataset, is_mask_lm = False, style_classifier_path = style_classifier, 
                style_batch_size = args.style_batch_size, style_classifier_device_id = args.style_classifier_device_id, 
                num_samples = args.num_samples_in_decoding, task_top_k = args.task_topk_in_decoding)
    # 3. Perform Pruning
    """
    Structure of prompt_queues: [(prompt, joint, gm, content, style, fluency, prompt_length, mask)] 
    """
    prompt_queues, num_iterations = pruner.forward(prompt=prompt, test_loader=[source_texts, target_labels, ref_texts], 
                                    reward_driven=False, fix_prune_order=args.fix_prune_order)
    # 4. Save the collection of prompts
    prompt_queues = [(p, joint.item(), gm.item(), content.item(), style.item(), fluency.item(), l, m) 
                     for p, joint, gm, content, style, fluency, l, m in prompt_queues]
    prompt_collection_df = pd.DataFrame(prompt_queues, columns=['prompt', 'joint', 'gm', 
                                        'content', 'style', 'fluency', '#tokens', "mask"])
    prompt_collection_df = prompt_collection_df.drop(columns=["mask"])
    
    model_name = args.model.split("/")[1] if "/" in args.model else args.model
    if not os.path.exists(f"./PrunedPrompts_by_{args.pruner}/"):
        os.mkdir(f"./PrunedPrompts_by_{args.pruner}/")
        
    if args.fix_prune_order:
        if args.pruner == "TAPruning":
            args.num_samples_in_search = 200
            prompt_collection_df.to_csv(
            f"./PrunedPrompts_by_{args.pruner}/{model_name}_{args.dataset}_{args.num_samples_in_search}-samples_{args.TAPruning_threshold}"
            f"_{args.reward_driven}_{args.ICL_shots}-shot_{args.ICL_index}.csv")
        elif args.pruner == "PromptQuine":
            prompt_collection_df.to_csv(
            f"./PrunedPrompts_by_{args.pruner}/{model_name}_{args.dataset}_{args.num_samples_in_search}-shots"
            f"_{args.reward_driven}_{args.ICL_shots}-shot_{args.ICL_index}.csv")
    else:
        if args.pruner == "TAPruning":
            args.num_samples_in_search = 200
            prompt_collection_df.to_csv(
            f"./PrunedPrompts_by_{args.pruner}/{model_name}_{args.dataset}_{args.num_samples_in_search}-samples_{args.TAPruning_threshold}"
            f"_prune-order-{args.prune_order_seed}_{args.reward_driven}_{args.ICL_shots}-shot_{args.ICL_index}.csv")
        elif args.pruner == "PromptQuine":
            prompt_collection_df.to_csv(
            f"./PrunedPrompts_by_{args.pruner}/{model_name}_{args.dataset}_{args.num_samples_in_search}-shots"
            f"_prune-order-{args.prune_order_seed}_{args.reward_driven}_{args.ICL_shots}-shot_{args.ICL_index}.csv")

            
if __name__ == "__main__":
    starttime = datetime.datetime.now()
    
    parser = argparse.ArgumentParser(description='Prompt pruning for style transfer.')
    parser.add_argument('--data_mode', type=str, default= "reduce", help='dataset mode for TAPruning')
    parser.add_argument('--model', type=str, default= "openai-community/gpt2", help='Full huggingface model name')
    parser.add_argument('--num_samples_in_search', type=int, default=100, help='Dataset samples for pruning')
    parser.add_argument('--proxy_samples_in_search', type=int, default=50, help='Dataset proxy samples for pruning (early stopping for PromptQuine)')
    parser.add_argument('--dataset', type=str, default= "yelp", \
                        choices = ["yelp"], help='Dataset description')
    parser.add_argument('--direction', type=str, default= "1_to_0", help='TST style transfer direction')
    parser.add_argument('--pruner', type=str, default= "TAPruning", choices = ["TAPruning", "PromptQuine"], help='Pruning algorithm used.')
    parser.add_argument('--fix_prune_order', type=bool, default=True, help='Indicator: whether to fix the pruning order (e.g., TAPruning)')
    parser.add_argument('--TAPruning_threshold', type=float, default=0.96, help='Threshold for TAPruning')
    parser.add_argument('--prune_order_seed', type=int, default=0, help='If not fixing the order, provide the seed (applicable to TAPruning Only)')
    parser.add_argument('--ICL_shots', type=int, default=2, help='Scaling the shots for ICL, tycically setting to 1')
    parser.add_argument('--ICL_index', type=int, default=0, help='Index of the ICL prompt')
    parser.add_argument('--classifier_setup', type=str, default="train", help='train/test for classifier')
    # classifier
    parser.add_argument('--style_batch_size', type=int, default=32, help='Batch size for style classifier inference (used for Best-of-N sampling)')
    parser.add_argument('--style_classifier_device_id', type=int, default=0, help='Which CUDA Device for style classifier (used for Best-of-N sampling)')
    # decoding
    parser.add_argument('--num_samples_in_decoding', type=int, default=1, help='Number of samples for Best-of-N sampling')
    parser.add_argument('--task_topk_in_decoding', type=int, default=1, help='Top-k sampling')
    
    
    
    args = parser.parse_args()
    
    main(args)
    
    endtime = datetime.datetime.now()
    print(endtime-starttime)
    