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
from fsc_evaluator import PromptedReasoningEvaluator
# TODO
sys.path.append("../../")
from modules.TAPruner import TAPruner
from modules.PromptQuinePruner import PromptQuinePruner
from dataset_helper import make_reasoning_dataset


def main(args):
    base_path = "./data"
    dataset_description = args.dataset
    if args.pruner == "PromptQuine":
        """
        source_texts_list, labels_list = make_balanced_classification_dataset(
            args.dataset, args.dataset_seed, base_path, args.num_shots,
            args.model, args.dataset_split, args.dataset_split_seed,
        )
        # TODO
        """
    elif args.pruner == "TAPruning":
        (valid_dataset, num_classes, verbalizers, template) = make_reasoning_dataset(
            args.dataset, args.data_mode, base_path, args.num_samples)
    
    # TODO, for TAPruning for now.
    valid_loader = DataLoader(valid_dataset,
                             shuffle=False,
                             batch_size=512,
                             drop_last=False)
    
    # 1. Load ICL Prompts for Pruning
    prompts_path = f"../../prompts/reasoning_prompts/{args.dataset}_few_shot_natural_prompts_{args.ICL_shots}shot.jsonl"
    
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
        pruner = TAPruner(PromptedReasoningEvaluator, args.model, "reasoning", None, "vLLM",
                 threshold=0.96, dataset = args.dataset, is_mask_lm = False)
    elif args.pruner == "SAHCPruning":
        pass
    # 3. Perform Pruning
    """
    Structure of prompt_queues: [(prompt, accuracy on valid set, reward (-1) on valid set, prompt_length, mask)] 
    """
    prompt_queues, num_iterations = pruner.forward(prompt=prompt, test_loader=valid_loader, reward_driven=False,
                fix_prune_order=args.fix_prune_order)
    # 4. Save the collection of prompts
    prompt_queues = [(p, acc.item(), r.item(), l, m) for p, acc, r, l, m in prompt_queues]
    prompt_collection_df = pd.DataFrame(prompt_queues, columns=['prompt', 'acc', 'reward', '#tokens', "mask"])
    prompt_collection_df = prompt_collection_df.drop(columns=["mask"])
    
    model_name = args.model.split("/")[1] if "/" in args.model else args.model
    if not os.path.exists(f"./PrunedPrompts_by_{args.pruner}/"):
        os.mkdir(f"./PrunedPrompts_by_{args.pruner}/")
        
    if args.fix_prune_order:
        if args.pruner == "TAPruning":
            prompt_collection_df.to_csv(
            f"./PrunedPrompts_by_{args.pruner}/{model_name}_{args.dataset}_{args.num_samples}-samples_{args.TAPruning_threshold}"
            f"_{args.ICL_shots}-shot_{args.ICL_index}.csv")
        elif args.pruner == "PromptQuine":
            prompt_collection_df.to_csv(
            f"./PrunedPrompts_by_{args.pruner}/{model_name}_{args.dataset}_{args.num_shots}-samples"
            f"_{args.ICL_shots}-shot_{args.ICL_index}.csv")
    else:
        if args.pruner == "TAPruning":
            prompt_collection_df.to_csv(
            f"./PrunedPrompts_by_{args.pruner}/{model_name}_{args.dataset}_{args.num_samples}-samples_{args.TAPruning_threshold}"
            f"_prune-order-{args.prune_order_seed}_{args.ICL_shots}-shot_{args.ICL_index}.csv")
        elif args.pruner == "PromptQuine":
            prompt_collection_df.to_csv(
            f"./PrunedPrompts_by_{args.pruner}/{model_name}_{args.dataset}_{args.num_shots}-samples"
            f"_prune-order-{args.prune_order_seed}_{args.ICL_shots}-shot_{args.ICL_index}.csv")

            
if __name__ == "__main__":
    starttime = datetime.datetime.now()
    
    parser = argparse.ArgumentParser(description='Prompt pruning for reasoning.')
    parser.add_argument('--data_mode', type=str, default= "dev", help='dataset mode for TAPruning')
    parser.add_argument('--model', type=str, default= "meta-llama/Meta-Llama-3-8B-Instruct", help='Full huggingface model name')
    parser.add_argument('--num_samples', type=int, default=200, help='Dataset shots for pruning')
    parser.add_argument('--proxy_samples', type=int, default=100, help='Dataset proxy shots for pruning (early stopping)')
    #parser.add_argument('--dataset_split', type=bool, default=True, help='Dataset split indicator (half)')
    #parser.add_argument('--dataset_split_seed', type=int, default=0, help='Dataset split seed (half)')
    parser.add_argument('--dataset', type=str, default= "gsm8k", 
                        choices = ["gsm8k", 'mawps'], help='Dataset description')
    parser.add_argument('--pruner', type=str, default= "TAPruning", choices = ["TAPruning", "SAHCPruning", "PromptQuine"], help='Pruning algorithm used.')
    parser.add_argument('--fix_prune_order', type=bool, default=True, help='Indicator: whether to fix the pruning order (e.g., TAPruning)')
    parser.add_argument('--TAPruning_threshold', type=float, default=0.96, help='Threshold for TAPruning')
    parser.add_argument('--prune_order_seed', type=int, default=0, help='If not fixing the order, provide the seed (applicable to TAPruning Only)')
    parser.add_argument('--ICL_shots', type=int, default= 1, help='Scaling the shots for ICL, tycically setting to 1')
    parser.add_argument('--ICL_index', type=int, default= 0, help='Index of the ICL prompt')
    
    args = parser.parse_args()
    
    main(args)
    
    endtime = datetime.datetime.now()
    print(endtime-starttime)
    