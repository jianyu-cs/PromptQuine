import sys
import os
import copy
import datetime
import argparse
from itertools import compress
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaTokenizer
from typing import Optional, Union, List, Dict, Any
from fsc_evaluator import PromptedClassificationEvaluator
# TODO
sys.path.append("../../")
from modules.TAPruner import TAPruner
from modules.PromptQuinePruner import PromptQuinePruner
from dataset_helper import make_balanced_classification_dataset

def main(args):
    base_path = "./data"
    dataset_description = args.dataset
    args.dataset = args.dataset.replace("-random", "")
    if args.pruner == "PromptQuine":
        source_texts_list, labels_list = make_balanced_classification_dataset(
            args.dataset,
            args.dataset_seed,
            base_path,
            args.num_shots,
            args.model,
            args.dataset_split,
            args.dataset_split_seed,
        )
        # TODO
    elif args.pruner == "TAPruning":
        (valid_dataset, num_classes, verbalizers, template) = make_classification_dataset(
            args.dataset, args.dataset_seed, base_path, args.model, args.data_mode)
    # TODO, for TAPruning for now.
    valid_loader = DataLoader(valid_dataset,
                             shuffle=False,
                             batch_size=512,
                             drop_last=False)
    
    # 1. Load ICL Prompts for Pruning
    if "random" in dataset_description:
        # Random Verbalizers (e.g., counter-intuitive)
        prompts_path = f"../../prompts/classification_prompts/{args.dataset}/randomlabelwords_few_shot_natural_prompts.jsonl"
    else:
        prompts_path = f"../../prompts/classification_prompts/{config.dataset}/few_shot_natural_prompts_{args.ICL_shots}shot.jsonl" \
            if args.is_mask_lm == False else \
                f"../../prompts/classification_prompts/{config.dataset}/few_shot_natural_prompts_{args.ICL_shots}shot_masked.jsonl"
    
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
        pruner = TAPruner(PromptedClassificationEvaluator, args.model, "classification", None, "vLLM", # automatically switch to HF if Mask LM
                 threshold=0.96, dataset = args.dataset, is_mask_lm = args.is_mask_lm)
    # 3. Perform Pruning
    """
    Structure of prompt_queues: [(prompt, accuracy on valid set, reward on valid set, prompt_length, mask)] 
    """
    prompt_queues, num_iterations = pruner.forward(prompt=prompt, test_loader=valid_loader, reward_driven=args.reward_driven,
                fix_prune_order=args.fix_prune_order)
    # 4. Save the collection of prompts
    prompt_collection_df = pd.DataFrame(prompt_queues, columns=['prompt', 'acc', 'reward', '#tokens', "mask"])
    prompt_collection_df = prompt_collection_df.drop(columns=["mask"])
    
    model_name = ags.model.split("/")[1] if "/" in args.model else args.model
    if not os.path.exists(f"./PrunedPrompts_by_{args.pruner}/"):
        os.mkdir(f"./PrunedPrompts_by_{args.pruner}/")
        
    if args.fix_prune_order:
        if args.pruner == "TAPruning":
            args.num_shots = 200
            prompt_collection_df.to_csv(
            f"./PrunedPrompts_by_{args.pruner}/{args.model}_{args.dataset}_{args.num_shots}-samples_{args.TAPruning_threshold}"
            f"_{args.reward_driven}_{args.ICL_shots}-shot_{args.ICL_index}.csv")
        elif args.pruner == "PromptQuine":
            prompt_collection_df.to_csv(
            f"./PrunedPrompts_by_{args.pruner}/{args.model}_{args.dataset}_{args.num_shots}-shots"
            f"_{args.reward_driven}_{args.ICL_shots}-shot_{args.ICL_index}.csv")
    else:
        if args.pruner == "TAPruning":
            args.num_shots = 200
            prompt_collection_df.to_csv(
            f"./PrunedPrompts_by_{args.pruner}/{args.model}_{args.dataset}_{args.num_shots}-samples_{args.TAPruning_threshold}"
            f"_prune-order-{args.prune_order_seed}_{args.reward_driven}_{args.ICL_shots}-shot_{args.ICL_index}.csv")
        elif args.pruner == "PromptQuine":
            prompt_collection_df.to_csv(
            f"./PrunedPrompts_by_{args.pruner}/{args.model}_{args.dataset}_{args.num_shots}-shots"
            f"_prune-order-{args.prune_order_seed}_{args.reward_driven}_{args.ICL_shots}-shot_{args.ICL_index}.csv")

            
if __name__ == "__main__":
    starttime = datetime.datetime.now()
    
    parser = argparse.ArgumentParser(description='Prompt pruning for classification.')
    parser.add_argument('--data_mode', type=str, default= "reduce", help='dataset mode for TAPruning')
    parser.add_argument('--model', type=str, default= "openai-community/gpt2", help='Full huggingface model name')
    parser.add_argument('--is_mask_lm', type=bool, default= False, help='Mask LM Indicator')
    parser.add_argument('--num_shots', type=int, default=16, help='Dataset shots for pruning')
    parser.add_argument('--reward_driven', type=bool, default=False, help='Indicator: whether to use reward by RLPrompt')
    parser.add_argument('--dataset_seed', type=int, default=0, help='Dataset seed')
    parser.add_argument('--dataset_split', type=bool, default=True, help='Dataset split indicator (half)')
    parser.add_argument('--dataset_split_seed', type=int, default=0, help='Dataset split seed (half)')
    parser.add_argument('--dataset', type=str, default= "sst-2", 
                        options = ["sst-2", "subj", "agnews", "snli", "yelp-5", "yahoo", "piqa",
                                  "sst-2-random", "subj-random", "agnews-random", "snli-random",
                                  "yelp-5-random", "yahoo-random", "piqa-random"], help='Dataset description')
    parser.add_argument('--pruner', type=str, default= "TAPruning", options = ["TAPruning", "PromptQuine"], help='Pruning algorithm used.')
    parser.add_argument('--fix_prune_order', type=bool, default=True, help='Indicator: whether to fix the pruning order (e.g., TAPruning)')
    parser.add_argument('--TAPruning_threshold', type=float, default=0.96, help='Threshold for TAPruning')
    parser.add_argument('--prune_order_seed', type=int, default=0, help='If not fixing the order, provide the seed (applicable to TAPruning Only)')
    parser.add_argument('--ICL_shots', type=int, default= 1, help='Scaling the shots for ICL, tycically setting to 1')
    parser.add_argument('--ICL_index', type=int, default= 0, help='Index of the ICL prompt')
    
    args = parser.parse_args()
    
    main(args)
    
    endtime = datetime.datetime.now()
    print(endtime-starttime)
    