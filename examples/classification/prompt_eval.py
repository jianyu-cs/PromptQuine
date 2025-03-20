import sys
import os
import copy
sys.path.append("..")
import json
#from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import pandas as pd
#from rlprompt.utils.utils import colorful_print
from fsc_helpers import (make_few_shot_classification_dataset, make_reduced_classification_dataset,
                         get_dataset_verbalizers, make_icl_val_classification_dataset,
                        make_reduced_test_classification_dataset, make_scheduled_reduced_classification_dataset)
from fsc_evaluator import PromptedClassificationEvaluator
import argparse

sst2_label_mapping = {0: 'terrible', 1: 'great'}
subj_label_mapping = {0: 'objective', 1: 'subjective'}
agnews_label_mapping = {0: 'World', 1: "Sports", 2: "Business", 3: "Tech"}
snli_label_mapping = {0: 'Yes', 1: 'Unknown', 2: 'No'}
yelp5_label_mapping = {0: 'terrible', 1: 'bad', 2: 'neutral', 3: 'good', 4: 'great'}
yahoo_label_mapping = {0: 'culture', 1: 'science', 2: 'health',
                      3: 'education', 4: 'computer', 5: 'sports',
                      6: 'business', 7: 'music', 8: 'family', 9: 'politics'}
mcq_label_mapping = {0: 'A', 1: 'B'}


def main(config):
    # Load datasets
    (test_dataset, num_classes, verbalizers, template) = \
        make_reduced_test_classification_dataset(config)
    (val_dataset, num_classes, verbalizers, template) = \
        make_reduced_classification_dataset(config)
    
    
    (scheduled_val_dataset, scheduled_num_classes, scheduled_verbalizers, scheduled_template) = \
        make_scheduled_reduced_classification_dataset(config)
    
    print('Test Size', len(test_dataset))
    print('Examples:', test_dataset[:5])
    
    # Prepare dataloaders
    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=512,
                             drop_last=False)
    val_loader = DataLoader(val_dataset,
                             shuffle=False,
                             batch_size=512,
                             drop_last=False)
    
    scheduled_val_loader = DataLoader(scheduled_val_dataset,
                             shuffle=False,
                             batch_size=512,
                             drop_last=False)
    
    # Configure the dataset details, e.g., verbalizers
    is_mask_lm = True if 'bert' in config.task_lm else False
    verbalizers = get_dataset_verbalizers(config.dataset, config.task_lm)
    num_classes = len(verbalizers)
    if config.dataset == 'agnews' and is_mask_lm:
        template = "<mask> {prompt} {sentence_1}"
    elif config.dataset == 'dbpedia' and is_mask_lm:
        template = "{prompt} <mask> : {sentence_1}"
    else: 
        template = None
    
    # LM, Shot Mapping
    LMs = {'gemma-it': "gemma-it", "llama3-it": "llama3-it", "llama3-it-70B": "llama3-it-70B",
          'gpt2': "gpt2", "mistral": "mistral", 'opt-iml-1.3b': "opt-iml-1.3b", "opt-30b": "OPT-30b"}
    Shots = {1: "", 2: "2shot", 4: "4shot", 8: "8shot"}
    
    icl_shot = Shots[config.prompt_shot]
    # Initialize prompts
    if config.prompt_type == "Unpruned":
        templates_path = f"../../prompts/classification_prompts/{config.dataset}/few_shot_natural_prompts_{icl_shot}.jsonl"
        templates_dict_list = [] 
        with open(templates_path, 'r') as template_jsons:
            template_json_lists = list(template_jsons)
        for template_json_str in template_json_lists:
            templates_dict_list.append(json.loads(template_json_str))
        if args.index != None:
            prompt = templates_dict_list[config.index]['prompt']
        prompts = [prompt]
        ans = {"prompt": prompts}
        
    elif config.prompt_type == "TAPruning":
        prompt_dir = f"./{config.task_lm}_TAPruning"
        assert f"{prompt_dir}/{config.dataset}_{config.index}_{config.task_lm}_TAPruning_{config.TAPruning_threshold}.csv" in os.listdir(prompt_dir)
        prompt_path = f"{prompt_dir}/{config.dataset}_{config.index}_{config.task_lm}_TAPruning_{config.TAPruning_threshold}.csv"
        prompt_meta = pd.read_csv(prompt_path)
        prompts = [prompt_meta[prompt_meta['acc']==max(prompt_meta['acc'])].iloc[-1]["prompt"]]
        val_acc = prompt_meta[prompt_meta['acc']==max(prompt_meta['acc'])].iloc[-1]["acc"]
        ans = {"prompt": prompts, "val_acc": val_acc}
        
    elif config.prompt_type == "PromptQuine-SSGA":
        prompt_dir = f"./{config.task_lm}_PromptQuine-SSGA"
        assert f"{prompt_dir}/{config.dataset}_{config.index}_{config.task_lm}_PromptQuine-SSGA-{config.evolution_seed}.csv" in os.listdir(prompt_dir)
        prompt_path = f"{prompt_dir}/{config.dataset}_{config.index}_{config.task_lm}_PromptQuine-SSGA-{config.evolution_seed}.csv"
        prompt_meta = pd.read_csv(prompt_path)
        prompts = prompt_meta["prompt"].tolist()
        val_acc = prompt_meta["val_acc"].tolist()
        if config.limit_prompts != -1:
            ans = prompt_meta.sort_values(by="val_acc", ascending=False)
            prompt_meta = ans.head(ans)
            prompts = prompt_meta["prompt"].tolist()
            val_acc = prompt_meta["val_acc"].tolist()
        ans = {"prompt": prompts, "val_acc": val_acc}
        
    elif config.prompt_type == "PromptQuine-GGA":
        prompt_dir = f"./{config.task_lm}_PromptQuine-GGA"
        assert f"{prompt_dir}/{config.dataset}_{config.index}_{config.task_lm}_PromptQuine-GGA-{config.evolution_seed}.csv" in os.listdir(prompt_dir)
        prompt_path = f"{prompt_dir}/{config.dataset}_{config.index}_{config.task_lm}_PromptQuine-GGA-{config.evolution_seed}.csv"
        prompt_meta = pd.read_csv(prompt_path)
        prompts = prompt_meta["prompt"].tolist()
        val_acc = prompt_meta["val_acc"].tolist()
        if config.limit_prompts != -1:
            ans = prompt_meta.sort_values(by="val_acc", ascending=False)
            prompt_meta = ans.head(ans)
            prompts = prompt_meta["prompt"].tolist()
            val_acc = prompt_meta["val_acc"].tolist()
        ans = {"prompt": prompts, "val_acc": val_acc}
    # TODO
        
    if config.mode == "valid-only":
        ans = {"prompt": prompts,
               "val_acc": [],
               "test_acc": []}
        
    elif config.mode == "test-only":
        ans = {"prompt": prompts,
               "test_acc": []}
        
    elif config.mode == "valid-test":
        ans = {"prompt": prompts,
               "val_acc": [],
               "test_acc": []}
    
    
    
    
    if config.ILPS_flag:
        prompt_dir = f"./{config.task_lm}_few_shot_natural"
        if not config.instruct_flag:
            if f"few_shot_natural_{config.dataset}_{config.index}_{config.task_lm}_mistral_instruct_0.96.csv" in os.listdir(prompt_dir):
                prompt_path = f"{prompt_dir}/few_shot_natural_{config.dataset}_{config.index}_{config.task_lm}_mistral_instruct_0.96.csv"
            elif f"few_shot_natural_{config.dataset}_{config.index}_{config.task_lm}_threshold-1_0.96-1.csv" in os.listdir(prompt_dir):
                prompt_path = f"{prompt_dir}/few_shot_natural_{config.dataset}_{config.index}_{config.task_lm}_threshold-1_0.96-1.csv"
        else:
            if f"few_shot_natural_{config.dataset}_{config.index}_{config.task_lm}_mistral_instruct_0.96-instruct.csv" in os.listdir(prompt_dir):
                prompt_path = f"{prompt_dir}/few_shot_natural_{config.dataset}_{config.index}_{config.task_lm}_mistral_instruct_0.96-instruct.csv"
            elif f"few_shot_natural_{config.dataset}_{config.index}_{config.task_lm}_threshold-1_0.96-1-instruct.csv" in os.listdir(prompt_dir):
                prompt_path = f"{prompt_dir}/few_shot_natural_{config.dataset}_{config.index}_{config.task_lm}_threshold-1_0.96-1-instruct.csv"
            
        prompt_meta = pd.read_csv(prompt_path)
        prompts = [prompt_meta[prompt_meta['acc']==max(prompt_meta['acc'])].iloc[-1]["prompt"]]
        val_acc = prompt_meta[prompt_meta['acc']==max(prompt_meta['acc'])].iloc[-1]["acc"]
        ans = {"prompt": prompts,
               "val_acc": val_acc, 
               "test_acc": []}
        
    if config.ILPS_random_flag:
        prompt_dir = f"./{config.task_lm}_few_shot_natural"
        prompt_path = f"{prompt_dir}/few_shot_natural_{config.dataset}_{config.index}_{config.task_lm}_threshold-1_{config.prompt_type}_0.96-1-random-{config.ILPS_random_index}.csv"
            
        prompt_meta = pd.read_csv(prompt_path)
        prompts = [prompt_meta[prompt_meta['acc']==max(prompt_meta['acc'])].iloc[-1]["prompt"]]
        val_acc = prompt_meta[prompt_meta['acc']==max(prompt_meta['acc'])].iloc[-1]["acc"]
        #if type(val_acc) == list:
        #    pass
        #else:
        #    val_acc = [val_acc]
        #print(val_acc)
        ans = {"prompt": prompts,
               "val_acc": [], 
               "test_acc": []}
        
    if config.evolution_flag:
        prompt_dir = f"{config.task_lm}_ICL_NSGA_{config.pareto_formulation}_compress_{config.prob_formulation}_{config.num_shots}_0_30_{config.sample_size}_0.2_Wheel_k_U_entropy" # f"{config.task_lm}_few_shot_natural"
        if not config.instruct_flag:
            prompt_path = os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_top100.csv")
        else:
            prompt_path = os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_instruct_top100.csv")
            
        prompt_meta = pd.read_csv(prompt_path)
        prompts = prompt_meta['prompt'].tolist()
        ans = {"prompt": prompts,
               "val_acc": [],
               "test_acc": []}
        
    if config.evo_parallel_flag:
        prompt_dir = f"{config.task_lm}_ICL_NSGA_{config.pareto_formulation}_compress_{config.prob_formulation}_{config.num_shots}_0_30_{config.sample_size}_0.2_Wheel_k_U_entropy" # f"{config.task_lm}_few_shot_natural"
        if not config.instruct_flag:
            prompt_path = os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_top100_True_False.csv")
        else:
            prompt_path = os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_instruct_top100_True_False.csv")
            
        print(prompt_dir)
        prompt_meta = pd.read_csv(prompt_path)
        prompts = prompt_meta['prompt'].tolist()
        ans = {"prompt": prompts,
               "performance": prompt_meta['performance'].tolist(),
               #"bin": prompt_meta['bin'].tolist(),
               "val_acc": [],
               "test_acc": []}
    if config.evo_parallel_mutate_flag:
        prompt_dir = f"{config.task_lm}_ICL_NSGA_{config.pareto_formulation}_compress_{config.prob_formulation}_{config.num_shots}_0_30_{config.sample_size}_0.2_Wheel_k_U_entropy" # f"{config.task_lm}_few_shot_natural"
        if not config.instruct_flag:
            prompt_path = os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_top100_True_False_mutation.csv")
        else:
            prompt_path = os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_instruct_top100_True_False_mutation.csv")
            
        print(prompt_dir)
        prompt_meta = pd.read_csv(prompt_path)
        prompts = prompt_meta['prompt'].tolist()
        ans = {"prompt": prompts,
               "performance": prompt_meta['performance'].tolist(),
               #"bin": prompt_meta['bin'].tolist(),
               "val_acc": [],
               "test_acc": []}
    
    if config.pile_pruned_flag:
        # collect pile instructions
        pile_instruction_path = "/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/prompt-waywardness/prompts/pile.json"
        with open(pile_instruction_path, 'r') as f:
            pile_instructions = json.load(f)#[json.loads(_) for _ in f] # [_.keys()[0]]
            pile_instructions = pile_instructions.values()
            
        if config.dataset != 'piqa':
            prompt_dir = f"{config.task_lm}_ICL_NSGA_{config.pareto_formulation}_compress_{config.prob_formulation}_{config.num_shots}_0_30_{config.sample_size}_0.2_Wheel_k_U_entropy" # f"{config.task_lm}_few_shot_natural"

            prompt_path = os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_instruct_top100_True_False_eval.csv")
        else:
            prompt_dir = f"globalEntropy_acc_marginprob_prob_true-global-balanced-acc_30_50_0.2_Wheel_k_U_entropy/" # f"{config.task_lm}_few_shot_natural"

            prompt_path = os.path.join(prompt_dir, f"final_{config.dataset}_{config.task_lm}_{config.index}_{config.prompt_type}_200_entropy_yahoo.csv")
   
        
        prompt_meta = pd.read_csv(prompt_path)
        #prompts = prompt_meta['prompt'].tolist()
        
        prompt = prompt_meta[prompt_meta['val_acc']==max(prompt_meta['val_acc'])]['prompt'].tolist()[0]
        prompts = []
        for pile_instruction in pile_instructions:
            prompts.append(f"{pile_instruction}\n{prompt}")        
        
        ans = {"prompt": prompts,
               #"performance": prompt_meta['performance'].tolist(),
               #"bin": prompt_meta['bin'].tolist(),
               "val_acc": [],
               "test_acc": []}
    if config.pile_original_flag:
        # collect pile instructions
        pile_instruction_path = "/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/prompt-waywardness/prompts/pile.json"
        with open(pile_instruction_path, 'r') as f:
            pile_instructions = json.load(f)#[json.loads(_) for _ in f] # [_.keys()[0]]
            pile_instructions = pile_instructions.values()
            
        prompt_dir = f"{config.task_lm}_ICL_NSGA_{config.pareto_formulation}_compress_{config.prob_formulation}_{config.num_shots}_0_30_{config.sample_size}_0.2_Wheel_k_U_entropy"
        if args.prompt_type == "long":
            relative_templates_path = f"classification_prompts/{args.dataset}/few_shot_natural_prompts_8shot.jsonl"
        elif args.prompt_type == "2shot":  
            relative_templates_path = f"classification_prompts/{args.dataset}/few_shot_natural_prompts_2shot.jsonl"
        elif args.prompt_type == "4shot":  
            relative_templates_path = f"classification_prompts/{args.dataset}/few_shot_natural_prompts_4shot.jsonl"
        elif args.prompt_type == "short":  
            relative_templates_path = f"classification_prompts/{args.dataset}/few_shot_natural_prompts.jsonl"

        absolute_templates_path = "/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/" + relative_templates_path
        templates_dict_list = [] 
        with open(absolute_templates_path, 'r') as template_jsons:
            template_json_lists = list(template_jsons)#[json.loads(line) for line in template_jsons]# list(template_jsons)
        for template_json_str in template_json_lists:
            templates_dict_list.append(json.loads(template_json_str))

        if args.index != None:
            prompt = templates_dict_list[args.index]['prompt']
            
        prompts = []
        for pile_instruction in pile_instructions:
            prompts.append(f"{pile_instruction}\n{prompt}")        
        
        ans = {"prompt": prompts,
               #"performance": prompt_meta['performance'].tolist(),
               #"bin": prompt_meta['bin'].tolist(),
               "val_acc": [],
               "test_acc": []}
    if config.orthogonal_pruned_flag:
        # collect pile instructions
        orthogonal_instruction_path = "/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/prompt-waywardness/prompts/natural_prompts.json"
        with open(orthogonal_instruction_path, 'r') as f:
            orthogonal_instructions = json.load(f)#[json.loads(_) for _ in f] # [_.keys()[0]]
            orthogonal_instructions = orthogonal_instructions.values()
        
        if config.dataset != 'piqa':
            prompt_dir = f"{config.task_lm}_ICL_NSGA_{config.pareto_formulation}_compress_{config.prob_formulation}_{config.num_shots}_0_30_{config.sample_size}_0.2_Wheel_k_U_entropy" # f"{config.task_lm}_few_shot_natural"

            prompt_path = os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_instruct_top100_True_False_eval.csv")
        else:
            prompt_dir = f"globalEntropy_acc_marginprob_prob_true-global-balanced-acc_30_50_0.2_Wheel_k_U_entropy/" # f"{config.task_lm}_few_shot_natural"

            prompt_path = os.path.join(prompt_dir, f"final_{config.dataset}_{config.task_lm}_{config.index}_{config.prompt_type}_200_entropy_yahoo.csv")
   

        
        prompt_meta = pd.read_csv(prompt_path)
        #prompts = prompt_meta['prompt'].tolist()
        
        prompt = prompt_meta[prompt_meta['val_acc']==max(prompt_meta['val_acc'])]['prompt'].tolist()[0]
        #prompts[config.index]
        prompts = []
        for orthogonal_instruction in orthogonal_instructions:
            prompts.append(f"{orthogonal_instruction}\n{prompt}")        
        
        ans = {"prompt": prompts,
               #"performance": prompt_meta['performance'].tolist(),
               #"bin": prompt_meta['bin'].tolist(),
               "val_acc": [],
               "test_acc": []}
    if config.natural_pruned_flag:
        # collect pile instructions
        if config.dataset == 'subj':
            natural_instruction_path = "/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/prompt-waywardness/prompts/subj.json"
        elif config.dataset == 'snli':
            natural_instruction_path = "/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/prompt-waywardness/prompts/snli.json"
        elif config.dataset == 'agnews':
            natural_instruction_path = "/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/prompt-waywardness/prompts/agnews.json"
        elif config.dataset == 'piqa':
            natural_instruction_path = "/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/prompt-waywardness/prompts/piqa.json"
            
        with open(natural_instruction_path, 'r') as f:
            natural_instructions = json.load(f)#[json.loads(_) for _ in f] # [_.keys()[0]]
            natural_instructions = natural_instructions.values()
        
        if config.dataset != 'piqa':
            prompt_dir = f"{config.task_lm}_ICL_NSGA_{config.pareto_formulation}_compress_{config.prob_formulation}_{config.num_shots}_0_30_{config.sample_size}_0.2_Wheel_k_U_entropy" # f"{config.task_lm}_few_shot_natural"

            prompt_path = os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_instruct_top100_True_False_eval.csv")
        else:
            prompt_dir = f"globalEntropy_acc_marginprob_prob_true-global-balanced-acc_30_50_0.2_Wheel_k_U_entropy/" # f"{config.task_lm}_few_shot_natural"

            prompt_path = os.path.join(prompt_dir, f"final_{config.dataset}_{config.task_lm}_{config.index}_{config.prompt_type}_200_entropy_yahoo.csv")
   

        
        prompt_meta = pd.read_csv(prompt_path)
        #prompts = prompt_meta['prompt'].tolist()
        
        prompt = prompt_meta[prompt_meta['val_acc']==max(prompt_meta['val_acc'])]['prompt'].tolist()[0]
        #prompts[config.index]
        prompts = []
        for natural_instruction in natural_instructions:
            prompts.append(f"{natural_instruction}\n{prompt}")        
        
        ans = {"prompt": prompts,
               #"performance": prompt_meta['performance'].tolist(),
               #"bin": prompt_meta['bin'].tolist(),
               "val_acc": [],
               "test_acc": []}
        
    if config.natural_original_flag:
        # collect pile instructions
        if config.dataset == 'subj':
            natural_instruction_path = "/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/prompt-waywardness/prompts/subj.json"
        elif config.dataset == 'snli':
            natural_instruction_path = "/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/prompt-waywardness/prompts/snli.json"
        elif config.dataset == 'agnews':
            natural_instruction_path = "/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/prompt-waywardness/prompts/agnews.json"
        elif config.dataset == 'piqa':
            natural_instruction_path = "/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/prompt-waywardness/prompts/piqa.json"
            
        with open(natural_instruction_path, 'r') as f:
            natural_instructions = json.load(f)#[json.loads(_) for _ in f] # [_.keys()[0]]
            natural_instructions = natural_instructions.values()
        
        prompt_dir = f"{config.task_lm}_ICL_NSGA_{config.pareto_formulation}_compress_{config.prob_formulation}_{config.num_shots}_0_30_{config.sample_size}_0.2_Wheel_k_U_entropy"
        
        if args.prompt_type == "long":
            relative_templates_path = f"classification_prompts/{args.dataset}/few_shot_natural_prompts_8shot.jsonl"
        elif args.prompt_type == "2shot":  
            relative_templates_path = f"classification_prompts/{args.dataset}/few_shot_natural_prompts_2shot.jsonl"
        elif args.prompt_type == "4shot":  
            relative_templates_path = f"classification_prompts/{args.dataset}/few_shot_natural_prompts_4shot.jsonl"
        elif args.prompt_type == "short":  
            relative_templates_path = f"classification_prompts/{args.dataset}/few_shot_natural_prompts.jsonl"

        absolute_templates_path = "/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/" + relative_templates_path
        templates_dict_list = [] 
        with open(absolute_templates_path, 'r') as template_jsons:
            template_json_lists = list(template_jsons)#[json.loads(line) for line in template_jsons]# list(template_jsons)
        for template_json_str in template_json_lists:
            templates_dict_list.append(json.loads(template_json_str))

        if args.index != None:
            prompt = templates_dict_list[args.index]['prompt']
            
        prompts = []
        for natural_instruction in natural_instructions:
            prompts.append(f"{natural_instruction}\n{prompt}")        
        
        ans = {"prompt": prompts,
               #"performance": prompt_meta['performance'].tolist(),
               #"bin": prompt_meta['bin'].tolist(),
               "val_acc": [],
               "test_acc": []}
        
    if config.orthogonal_original_flag:
        # collect pile instructions
        orthogonal_instruction_path = "/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/prompt-waywardness/prompts/natural_prompts.json"
        with open(orthogonal_instruction_path, 'r') as f:
            orthogonal_instructions = json.load(f)#[json.loads(_) for _ in f] # [_.keys()[0]]
            orthogonal_instructions = orthogonal_instructions.values()
        
        prompt_dir = f"{config.task_lm}_ICL_NSGA_{config.pareto_formulation}_compress_{config.prob_formulation}_{config.num_shots}_0_30_{config.sample_size}_0.2_Wheel_k_U_entropy"
        
        if args.prompt_type == "long":
            relative_templates_path = f"classification_prompts/{args.dataset}/few_shot_natural_prompts_8shot.jsonl"
        elif args.prompt_type == "2shot":  
            relative_templates_path = f"classification_prompts/{args.dataset}/few_shot_natural_prompts_2shot.jsonl"
        elif args.prompt_type == "4shot":  
            relative_templates_path = f"classification_prompts/{args.dataset}/few_shot_natural_prompts_4shot.jsonl"
        elif args.prompt_type == "short":  
            relative_templates_path = f"classification_prompts/{args.dataset}/few_shot_natural_prompts.jsonl"

        absolute_templates_path = "/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/" + relative_templates_path
        templates_dict_list = [] 
        with open(absolute_templates_path, 'r') as template_jsons:
            template_json_lists = list(template_jsons)#[json.loads(line) for line in template_jsons]# list(template_jsons)
        for template_json_str in template_json_lists:
            templates_dict_list.append(json.loads(template_json_str))

        if args.index != None:
            prompt = templates_dict_list[args.index]['prompt']
            
        prompts = []
        for orthogonal_instruction in orthogonal_instructions:
            prompts.append(f"{orthogonal_instruction}\n{prompt}")        
        
        ans = {"prompt": prompts,
               #"performance": prompt_meta['performance'].tolist(),
               #"bin": prompt_meta['bin'].tolist(),
               "val_acc": [],
               "test_acc": []}
        
    if config.nolabel_GGA_flag or config.nolabelsignal_GGA_flag or config.noall_GGA_flag or config.randomlabel_flag:
        prompt_dir = f"{config.task_lm}_ICL_NSGA_{config.pareto_formulation}_compress_{config.prob_formulation}_{config.num_shots}_0_30_{config.sample_size}_0.2_Wheel_k_U_entropy" # f"{config.task_lm}_few_shot_natural"
        if not config.instruct_flag:
            prompt_path = os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_top100_True_False.csv")
                
        else:
            prompt_path = os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_instruct_top100_True_False.csv")
        if config.randomlabel_flag:
            prompt_path = os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_randomlabels_top100_True_False.csv")
        elif config.nolabel_GGA_flag:
            prompt_path = os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_nolabelGGA_top100_True_False.csv")
        elif config.nolabelsignal_GGA_flag:
            prompt_path = os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_nolabelsignalGGA_top100_True_False.csv")
        elif config.noall_GGA_flag:
        
            prompt_path = os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_noallGGA_top100_True_False.csv")
        print(prompt_dir)
        prompt_meta = pd.read_csv(prompt_path)
        prompts = prompt_meta['prompt'].tolist()
        ans = {"prompt": prompts,
               "performance": prompt_meta['performance'].tolist(),
               #"bin": prompt_meta['bin'].tolist(),
               "val_acc": [],
               "test_acc": []}
        if config.randomlabel_flag:
            designed_verbalizers = True
            
            absolute_templates_path = f"/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/classification_prompts/{config.dataset}/randomlabelwords_few_shot_natural_prompts.jsonl"
            with open(absolute_templates_path, "r") as f:
                prompt_ans = [json.loads(_) for _ in f]
            verbalizers = prompt_ans[config.index]['labels']
    if config.originalnolabel_GGA_flag:
        absolute_templates_path = f"/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/classification_prompts/{config.dataset}/few_shot_natural_prompts.jsonl"
        templates_dict_list = [] 
        with open(absolute_templates_path, 'r') as template_jsons:
            template_json_lists = list(template_jsons)#[json.loads(line) for line in template_jsons]# list(template_jsons)
        for template_json_str in template_json_lists:
            templates_dict_list.append(json.loads(template_json_str))

        prompt = templates_dict_list[config.index]['prompt']
        if config.dataset == 'sst-2':
            label_ids = [f" {correct_id}" for correct_id in list(sst2_label_mapping.values())]
        elif config.dataset == 'subj':
            label_ids = [f" {correct_id}" for correct_id in list(subj_label_mapping.values())]
        elif config.dataset == "yahoo":
            label_ids = [f" {correct_id}" for correct_id in list(yahoo_label_mapping.values())]
        elif config.dataset == "agnews":
            label_ids = [f" {correct_id}" for correct_id in list(agnews_label_mapping.values())]
        elif config.dataset == "sst-5":
            label_ids = [f" {correct_id}" for correct_id in list(sst5_label_mapping.values())]
        elif config.dataset == "yelp-5":
            label_ids = [f" {correct_id}" for correct_id in list(yelp5_label_mapping.values())]
        elif config.dataset == "snli":
            label_ids = [f" {correct_id}" for correct_id in list(snli_label_mapping.values())]

        for label_id in label_ids:
            prompt = prompt.replace(label_id, "")
        print(prompt)
        prompts = [prompt]
        ans = {"prompt": [prompt],
               #"performance": prompt_meta['performance'].tolist(),
               #"bin": prompt_meta['bin'].tolist(),
               "val_acc": [],
               "test_acc": []}
    if config.originalnolabelsignal_GGA_flag:
        signal_words = ICL_SIGNAL_WORDS[config.dataset]
        
        absolute_templates_path = f"/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/classification_prompts/{config.dataset}/few_shot_natural_prompts.jsonl"
        templates_dict_list = [] 
        with open(absolute_templates_path, 'r') as template_jsons:
            template_json_lists = list(template_jsons)#[json.loads(line) for line in template_jsons]# list(template_jsons)
        for template_json_str in template_json_lists:
            templates_dict_list.append(json.loads(template_json_str))

        prompt = templates_dict_list[config.index]['prompt']
        if config.dataset == 'sst-2':
            label_ids = [f" {correct_id}" for correct_id in list(sst2_label_mapping.values())]
        elif config.dataset == 'subj':
            label_ids = [f" {correct_id}" for correct_id in list(subj_label_mapping.values())]
        elif config.dataset == "yahoo":
            label_ids = [f" {correct_id}" for correct_id in list(yahoo_label_mapping.values())]
        elif config.dataset == "agnews":
            label_ids = [f" {correct_id}" for correct_id in list(agnews_label_mapping.values())]
        elif config.dataset == "sst-5":
            label_ids = [f" {correct_id}" for correct_id in list(sst5_label_mapping.values())]
        elif config.dataset == "yelp-5":
            label_ids = [f" {correct_id}" for correct_id in list(yelp5_label_mapping.values())]
        elif config.dataset == "snli":
            label_ids = [f" {correct_id}" for correct_id in list(snli_label_mapping.values())]
        prompt = prompt.replace(signal_words, "") 
        for label_id in label_ids:
            prompt = prompt.replace(label_id, "")
        prompts = [prompt]
        print(prompt)
        ans = {"prompt": [prompt],
               #"performance": prompt_meta['performance'].tolist(),
               #"bin": prompt_meta['bin'].tolist(),
               "val_acc": [],
               "test_acc": []}
    if config.originalnoall_GGA_flag:
        signal_words = "\n"+ICL_SIGNAL_WORDS[config.dataset]
        
        absolute_templates_path = f"/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/classification_prompts/{config.dataset}/few_shot_natural_prompts.jsonl"
        templates_dict_list = [] 
        with open(absolute_templates_path, 'r') as template_jsons:
            template_json_lists = list(template_jsons)#[json.loads(line) for line in template_jsons]# list(template_jsons)
        for template_json_str in template_json_lists:
            templates_dict_list.append(json.loads(template_json_str))

        prompt = templates_dict_list[config.index]['prompt']
        if config.dataset == 'sst-2':
            label_ids = [f" {correct_id}" for correct_id in list(sst2_label_mapping.values())]
        elif config.dataset == 'subj':
            label_ids = [f" {correct_id}" for correct_id in list(subj_label_mapping.values())]
        elif config.dataset == "yahoo":
            label_ids = [f" {correct_id}" for correct_id in list(yahoo_label_mapping.values())]
        elif config.dataset == "agnews":
            label_ids = [f" {correct_id}" for correct_id in list(agnews_label_mapping.values())]
        elif config.dataset == "sst-5":
            label_ids = [f" {correct_id}" for correct_id in list(sst5_label_mapping.values())]
        elif config.dataset == "yelp-5":
            label_ids = [f" {correct_id}" for correct_id in list(yelp5_label_mapping.values())]
        elif config.dataset == "snli":
            label_ids = [f" {correct_id}" for correct_id in list(snli_label_mapping.values())]
        prompt = prompt.replace(signal_words, "") 
        for label_id in label_ids:
            prompt = prompt.replace(label_id, "")
        print(prompt)
        prompts = [prompt]
        ans = {"prompt": prompts,
               #"performance": prompt_meta['performance'].tolist(),
               #"bin": prompt_meta['bin'].tolist(),
               "val_acc": [],
               "test_acc": []}
        
    if config.originalrandomlabel_flag:
        designed_verbalizers = True
            
        absolute_templates_path = f"/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/classification_prompts/{config.dataset}/randomlabelwords_few_shot_natural_prompts.jsonl"
        with open(absolute_templates_path, "r") as f:
            prompt_ans = [json.loads(_) for _ in f]
        verbalizers = prompt_ans[config.index]['labels']
        prompt = prompt_ans[config.index]['prompt']
        print(prompt)
        prompts = [prompt]
        ans = {"prompt": prompts,
               #"performance": prompt_meta['performance'].tolist(),
               #"bin": prompt_meta['bin'].tolist(),
               "val_acc": [],
               "test_acc": []}

    if config.random_flag:
        prompt_dir = f"./{config.task_lm}_few_shot_natural"
        # few_shot_natural_subj_13_llama3-it_random_0.96.csv
        if f"few_shot_natural_{config.dataset}_{config.pareto_seed}_{config.task_lm}_random_0.96.csv" in os.listdir(prompt_dir):
            prompt_path = f"{prompt_dir}/few_shot_natural_{config.dataset}_{config.pareto_seed}_{config.task_lm}_random_0.96.csv"
    
        prompt_meta = pd.read_csv(prompt_path)
        prompts = [prompt_meta[prompt_meta['acc']==max(prompt_meta['acc'])].iloc[-1]["prompt"]]
        val_acc = prompt_meta[prompt_meta['acc']==max(prompt_meta['acc'])].iloc[-1]["acc"]
        ans = {"prompt": prompts,
               "val_acc": val_acc, 
               "test_acc": []}
    
    if config.original_flag:
        
        if args.prompt_type == "long":
            relative_templates_path = f"classification_prompts/{args.dataset}/few_shot_natural_prompts_8shot.jsonl"
        elif args.prompt_type == "2shot":  
            relative_templates_path = f"classification_prompts/{args.dataset}/few_shot_natural_prompts_2shot.jsonl"
        elif args.prompt_type == "4shot":  
            relative_templates_path = f"classification_prompts/{args.dataset}/few_shot_natural_prompts_4shot.jsonl"
        elif args.prompt_type == "short":  
            relative_templates_path = f"classification_prompts/{args.dataset}/few_shot_natural_prompts.jsonl"

        absolute_templates_path = "/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/" + relative_templates_path
        templates_dict_list = [] 
        with open(absolute_templates_path, 'r') as template_jsons:
            template_json_lists = list(template_jsons)#[json.loads(line) for line in template_jsons]# list(template_jsons)
        for template_json_str in template_json_lists:
            templates_dict_list.append(json.loads(template_json_str))

        if args.index != None:
            prompt = templates_dict_list[args.index]['prompt']
        """
        prompt_dir = f"./{config.task_lm}_few_shot_natural"
        if not config.instruct_flag:
            if f"few_shot_natural_{config.dataset}_{config.index}_{config.task_lm}_mistral_instruct_0.96.csv" in os.listdir(prompt_dir):
                prompt_path = f"{prompt_dir}/few_shot_natural_{config.dataset}_{config.index}_{config.task_lm}_mistral_instruct_0.96.csv"
            elif f"few_shot_natural_{config.dataset}_{config.index}_{config.task_lm}_threshold-1_0.96-1.csv" in os.listdir(prompt_dir):
                prompt_path = f"{prompt_dir}/few_shot_natural_{config.dataset}_{config.index}_{config.task_lm}_threshold-1_0.96-1.csv"
        else:
            if f"few_shot_natural_{config.dataset}_{config.index}_{config.task_lm}_mistral_instruct_0.96-instruct.csv" in os.listdir(prompt_dir):
                prompt_path = f"{prompt_dir}/few_shot_natural_{config.dataset}_{config.index}_{config.task_lm}_mistral_instruct_0.96-instruct.csv"
            elif f"few_shot_natural_{config.dataset}_{config.index}_{config.task_lm}_threshold-1_0.96-1-instruct.csv" in os.listdir(prompt_dir):
                prompt_path = f"{prompt_dir}/few_shot_natural_{config.dataset}_{config.index}_{config.task_lm}_threshold-1_0.96-1-instruct.csv"
            

        prompt_meta = pd.read_csv(prompt_path)
        prompts = [prompt_meta.iloc[0]["prompt"]]
        """
        prompts = [prompt]
        #val_acc = prompt_meta.iloc[0]["acc"]
        ans = {"prompt": prompts,
               "val_acc": [],#val_acc, 
               "test_acc": []}
    
    if config.LLMLingua_flag:
        prompt_path = f"/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/classification_prompts/{config.dataset}/few_shot_natural_prompts_llmlingua_{config.task_lm}.jsonl"
        with open(prompt_path, "r") as f:
            prompts= [[json.loads(_) for _ in f][config.pareto_seed]["compressed_output"]['compressed_prompt']]
        ans = {"prompt": prompts,
               "test_acc": []}
        
    if config.LLMLingua2_flag:
        prompt_path = f"/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/classification_prompts/{config.dataset}/few_shot_natural_prompts_llmlingua2_xlm-roberta-large.jsonl"
        
        with open(prompt_path, "r") as f:
            prompts= [[json.loads(_) for _ in f][config.pareto_seed]["compressed_output"]['compressed_prompt']]
        if config.task_lm == 'roberta-large':
            prompts = [prompt+" <mask>" for prompt in prompts]
            
        ans = {"prompt": prompts,
               "test_acc": []}
        
    if config.RLPrompt_flag:
        prompt_dir = f"/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/classification_prompts/{config.dataset}"
        prompt_path = f"{prompt_dir}/rlprompt_{config.task_lm}.jsonl"
        with open(prompt_path, "r") as f:
            prompts= [json.loads(_)['prompt'] for _ in f]
        ans = {"prompt": prompts,
               "val_acc": [],
               "test_acc": []}
    
    if config.attribution_flag:
        prompt_path = f"/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/examples/few-shot-classification/evaluation/interpret-lm/{config.task_lm}_{config.dataset}_{config.index}_{config.attribution_type}_0.95_0.05_whole.csv"
        
        prompts= pd.read_csv(prompt_path)['prompt']#[[json.loads(_) for _ in f][config.pareto_seed]["compressed_output"]['compressed_prompt']]
        ans = {"prompt": prompts,
               "test_acc": []}
    
    if config.example_flag:
        prompts = [config.prompt]
        ans = {"prompt": prompts,
               "test_acc": []}
    if config.attention_first_flag:
        prompt_path = f"/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/examples/few-shot-classification/evaluation/interpret-lm/{config.task_lm}_{config.dataset}_{config.index}_attention_first_0.95_0.05.csv"
        
        prompts= pd.read_csv(prompt_path)['prompt']#[[json.loads(_) for _ in f][config.pareto_seed]["compressed_output"]['compressed_prompt']]
        ans = {"prompt": prompts,
               "test_acc": []}
    if config.attention_aggregate_flag:
        prompt_path = f"/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/examples/few-shot-classification/evaluation/interpret-lm/{config.task_lm}_{config.dataset}_{config.index}_attention_aggregate_0.95_0.05.csv"
        
        prompts= pd.read_csv(prompt_path)['prompt']#[[json.loads(_) for _ in f][config.pareto_seed]["compressed_output"]['compressed_prompt']]
        ans = {"prompt": prompts,
               "test_acc": []}
    if config.RS_flag:
        prompt_path = f"/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/examples/few-shot-classification/evaluation/interpret-lm/subj_1000_llama3-it_{config.index}.csv"
        
        prompts= pd.read_csv(prompt_path)['prompt']#[[json.loads(_) for _ in f][config.pareto_seed]["compressed_output"]['compressed_prompt']]
        ans = {"prompt": prompts,
               "test_acc": []}
    if config.example_shuffle_flag:
        prompt_dir = f"./{config.task_lm}_few_shot_natural"
        
        prompt_path = f"{prompt_dir}/few_shot_natural_{config.dataset}_{config.index}_{config.task_lm}_threshold-1_{config.prompt_type}_0.96-1-random-{config.shuffle_seed}.csv"
        print(prompt_path)
        prompt_meta = pd.read_csv(prompt_path)
        prompts = [prompt_meta[prompt_meta['acc']==max(prompt_meta['acc'])].iloc[-1]["prompt"]]
        val_acc = prompt_meta[prompt_meta['acc']==max(prompt_meta['acc'])].iloc[-1]["acc"]
        ans = {"prompt": prompts,
               "val_acc": val_acc, 
               "test_acc": []}
    if config.evoRS_shot_flag:
        prompt_dir = f"{config.task_lm}_ICL_NSGA_{config.pareto_formulation}_compress_{config.prob_formulation}_{config.num_shots}_0_30_{config.sample_size}_0.2_Wheel_k_U_entropy" 
        prompt_path = os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_ICLevolution_constrained_min_0_1shotwhole_100_sep_1000.csv")

            
        print(prompt_path)
        prompt_meta = pd.read_csv(prompt_path)
        prompts = prompt_meta['prompt'].tolist()#[prompt_meta[prompt_meta['acc']==max(prompt_meta['acc'])].iloc[-1]["prompt"]]
        lengths = prompt_meta['length'].tolist()
        #val_acc = #prompt_meta[prompt_meta['acc']==max(prompt_meta['acc'])].iloc[-1]["acc"]
        ans = {"prompt": prompts,
               "length": lengths,
               "val_acc": [], 
               "test_acc": []}
        
    if config.RS_shot_flag:
        prompt_dir = f"./{config.task_lm}_rs"
        
        prompt_path = f"{prompt_dir}/{config.dataset}_{config.prompt_type}_RS_1000.csv"
        print(prompt_path)
        prompt_meta = pd.read_csv(prompt_path)
        prompts = prompt_meta['prompt'].tolist()#[prompt_meta[prompt_meta['acc']==max(prompt_meta['acc'])].iloc[-1]["prompt"]]
        lengths = prompt_meta['length'].tolist()
        #val_acc = #prompt_meta[prompt_meta['acc']==max(prompt_meta['acc'])].iloc[-1]["acc"]
        ans = {"prompt": prompts,
               "length": lengths,
               "val_acc": [], 
               "test_acc": []}
    if config.compress_flag:
        prompt_dir = f"./globalEntropy_acc_marginprob_prob_true-global-balanced-acc_30_50_0.2_Wheel_k_U_entropy"
        
        prompt_path = f"Final_compression_{config.dataset}_{config.index}_llama3-it_0.98.csv" #f"{prompt_dir}/{config.dataset}_{config.prompt_type}_RS_1000.csv"
        print(prompt_path)
        prompt_meta = pd.read_csv(prompt_path)
        prompts = prompt_meta['prompt'].tolist()#[prompt_meta[prompt_meta['acc']==max(prompt_meta['acc'])].iloc[-1]["prompt"]]
        lengths = prompt_meta['length'].tolist()
        #val_acc = #prompt_meta[prompt_meta['acc']==max(prompt_meta['acc'])].iloc[-1]["acc"]
        ans = {"prompt": prompts,
               "length": lengths,
               "val_acc": [], 
               "test_acc": []}
        
    
    #print(prompt_path)
        
        
    """
    prompt_dir = "globalEntropy_acc_marginprob_prob_true-global-balanced-acc_30_50_0.2_Wheel_k_U"#f"{config.task_lm}_ICL_NSGA_{config.pareto_formulation}_compress_{config.prob_formulation}_{config.num_shots}_0_30_50_0.2_Wheel_k_U" # f"{config.task_lm}_few_shot_natural"
    prompt_path = os.path.join(prompt_dir, f"{config.dataset}_{config.prompt_type}.csv")#f"Turn1_{config.pareto_seed}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_top100.csv")#f"Turn1_{config.pareto_seed}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_ICLevolution_constrained_min_0_1shot_100_+.csv")#"turn1_1_0.8.csv")#f"Turn1_{config.pareto_seed}_0_{config.dataset}_ICLevolution_constrained_min_{config.evol_seed}_8shot.csv")#f"{config.dataset}_ICL_0_{config.num_shots}_turn1_{config.evol_seed}_top5.csv") #" f"{config.dataset}_{config.task_lm}_final.csv") #Turn1_0_0_subj_ICLevolution_constrained_min_0_8shot.csv
    #prompt_path = os.path.join(prompt_dir, f"{config.dataset}_ICL_{config.num_seed}_{config.num_shots}_turn1_constrain_min_0.csv") #"{config.dataset}_gpt2_final.csv"
    """ 
    
    """
    relative_templates_path = f"classification_prompts/{args.dataset}/few_shot_natural_prompts_masked.jsonl"
    absolute_templates_path = "/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/prompts/" + relative_templates_path
    
    templates_dict_list = [] 
    with open(absolute_templates_path, 'r') as template_jsons:
        template_json_lists = list(template_jsons)
    for template_json_str in template_json_lists:
        templates_dict_list.append(json.loads(template_json_str))


    prompts = [templates_dict_list[_]['prompt'] for _ in range(len(templates_dict_list))]
    """
    #compressed_prompts = prompt_meta['strong'].tolist() #prompt_meta['compressed_prompt'].tolist() # prompt_meta['compressed_prompt'].tolist()
    """
    prompt_indices = prompt_meta['index'].tolist()
    
    # prompts
    original_prompts = prompt_meta['original_prompt'].tolist() # prompt_meta['origin'].tolist()#prompt_meta['origin'].tolist()#
    strong_prompts =  prompt_meta['compressed_prompt'].tolist() # prompt_meta['strong'].tolist()#prompt_meta['strong'].tolist()#
    """
    
    # + prompts_file['final_val_acc'].tolist() + prompts_file['strong_val_acc'].tolist()
    tester = PromptedClassificationEvaluator(
            config, 
            task_lm=config.task_lm,
            is_mask_lm=None,#config.is_mask_lm,
            num_classes=num_classes,
            verbalizers=verbalizers,
            template=prompts[0], 
            prompt=prompts[0],#config.prompt
            dataset=config.dataset,
            designed_verbalizers=designed_verbalizers
        )
    '''
    for i,prompt in enumerate(strong_prompts):
        #tester = PromptedClassificationEvaluator(
        #    task_lm=config.task_lm,
        #    is_mask_lm=config.is_mask_lm,
        #    num_classes=num_classes,
        #    verbalizers=verbalizers,
        #    template=prompt, 
        #    prompt=prompt#config.prompt
        #)
        #tester.template = prompt
        print("index: ", i)
        #acc = tester.forward(val_loader, prompt)
        #print(acc)
        #ans['val_acc'].append(acc[0])

        acc = tester.forward(val_loader, prompt)

        if config.dataset == "subj":
            print(1-acc[0])
            ans['val_acc'].append(1-float(acc[0].cpu()))
        else:
            print(acc[0])
            ans['val_acc'].append(float(acc[0].cpu()))
    
    '''
    if False:#config.evo_parallel_flag:
        final_ans = {"prompt": [], "scheduled_val_acc": [], "val_acc": [], "test_acc": []}
        peak_scheduled_val_acc = 0
        
        sceduled_acc_collections = []
        
        for i,prompt in enumerate(prompts):
            print("index: ", i)

            acc = tester.forward(scheduled_val_loader, prompt)
            sceduled_acc_collections.append(float(acc[0].cpu()))
        
        indexed_prompt_acc = list(enumerate(sceduled_acc_collections))
        ranked_list = sorted(indexed_prompt_acc, key=lambda x: x[1], reverse=True)
        ranks = [index for index, value in ranked_list]
        
        for rank in ranks:
            print(peak_scheduled_val_acc, sceduled_acc_collections[rank])
            if sceduled_acc_collections[rank] > peak_scheduled_val_acc:

                final_ans['prompt'].append(prompts[rank])
                final_ans['scheduled_val_acc'].append(sceduled_acc_collections[rank])
                val_acc = tester.forward(val_loader, prompts[rank])
                final_ans['val_acc'].append(float(val_acc[0].cpu()))

                test_acc = tester.forward(test_loader, prompts[rank])         
                final_ans['test_acc'].append(float(test_acc[0].cpu()))
                if float(val_acc[0].cpu()) > peak_scheduled_val_acc:
                    peak_scheduled_val_acc = copy.deepcopy(float(val_acc[0].cpu()))

                #acc = tester.forward(test_loader, prompt)            
                #ans['test_acc'].append(float(acc[0].cpu()))
    else:
        max_test = 0
        for i,prompt in enumerate(prompts):
            #tester = PromptedClassificationEvaluator(
            #    task_lm=config.task_lm,
            #    is_mask_lm=config.is_mask_lm,
            #    num_classes=num_classes,
            #    verbalizers=verbalizers,
            #    template=prompt, 
            #    prompt=prompt#config.prompt
            #)
            #tester.template = prompt

            print("index: ", i)
            #if not config.whole_test_flag:
            if not config.ILPS_flag and not config.example_shuffle_flag:
                acc = tester.forward(val_loader, prompt)
                #    #print(acc)
                #    #ans['val_acc'].append(acc[0])
                #    #acc = tester.forward(val_loader, prompt)
                #    #print(acc)
                ans['val_acc'].append(float(acc[0].cpu()))
            #    """
            #    ans['val_marginprob'].append(float(acc[1].cpu()))
            #    ans["val_global_entropy"].append(float(acc[2]))
            #    ans["val_entropy"].append(float(acc[3]))
            #    ans["val_balanced_acc"].append(float(acc[4]))
            #    ans["val_balanced_acc_global_entropy"].append(float(acc[5]))
            #    """

            acc = tester.forward(test_loader, prompt)
            print(acc)
            if float(acc[0].cpu().item()) > max_test:
                max_test = float(acc[0].cpu().item())
            ans['test_acc'].append(float(acc[0].cpu()))
            print("Max test:", max_test)
    
    print(ans)
    '''
    if config.flag < 2:
        pd.DataFrame(ans).to_csv(f"./{config.final_type}/{config.final_type}_{config.task_lm}_{config.dataset}_llmlingua.csv")
    else:
        pd.DataFrame(ans).to_csv(f"./{config.final_type}/{config.final_type}_{config.task_lm}_{config.dataset}_llmlingua2.csv")
    '''
    if config.example_flag:
        print(ans)
        return
    
    if not os.path.exists(f"./{config.pareto_formulation}_{config.prob_formulation}_30_50_0.2_Wheel_k_U_entropy"):
        os.mkdir(f"./{config.pareto_formulation}_{config.prob_formulation}_30_50_0.2_Wheel_k_U_entropy")
    
    #pd.DataFrame(ans).to_csv(f"./{config.pareto_formulation}_{config.prob_formulation}_30_50_0.2_Wheel_k_U/final_{config.dataset}_{config.prompt_type}.csv")
    #"Turn1_{config.dataset}_{config.pareto_seed}_{config.evol_seed}_{config.prompt_type}.csv")#{config.dataset}_turn1_{config.num_seed}_{config.num_shots}_{config.task_lm}_{config.pareto_seed}_{config.evol_seed}0.7.csv")
    
    if not config.ILPS_flag and not config.random_flag and not config.original_flag and not config.LLMLingua_flag and not config.LLMLingua2_flag and not config.attribution_flag and not config.attention_first_flag and not config.attention_aggregate_flag and not config.RS_flag and not config.example_shuffle_flag and not config.RS_shot_flag and not config.evoRS_shot_flag and not config.evo_parallel_flag and not config.RLPrompt_flag and not config.ILPS_random_flag and not config.orthogonal_original_flag and not config.orthogonal_pruned_flag and not config.pile_pruned_flag and not config.pile_original_flag and not config.natural_original_flag and not config.natural_pruned_flag and not config.natural_original_flag and not config.natural_pruned_flag and not config.evo_parallel_mutate_flag and not config.nolabel_GGA_flag and not config.nolabelsignal_GGA_flag and not config.noall_GGA_flag and not config.randomlabel_flag and not config.compress_flag:
        if not config.whole_test_flag:
            pd.DataFrame(ans).to_csv(f"./{config.pareto_formulation}_{config.prob_formulation}_30_50_0.2_Wheel_k_U_entropy/Turn1_{config.dataset}_{config.task_lm}_{config.index}_{config.prompt_type}_200_entropy_yahoo.csv")
        else:
            if config.final_val_flag:
                pd.DataFrame(ans).to_csv(f"./{config.pareto_formulation}_{config.prob_formulation}_30_50_0.2_Wheel_k_U_entropy/final_valacc_{config.dataset}_{config.task_lm}_{config.index}_{config.prompt_type}_200_entropy_yahoo.csv")
            else:
                pd.DataFrame(ans).to_csv(f"./{config.pareto_formulation}_{config.prob_formulation}_30_50_0.2_Wheel_k_U_entropy/final_{config.dataset}_{config.task_lm}_{config.index}_{config.prompt_type}_200_entropy_yahoo.csv")
    else:
        if config.ILPS_flag:
            pd.DataFrame(ans).to_csv(f"./{config.pareto_formulation}_{config.prob_formulation}_30_50_0.2_Wheel_k_U_entropy/final_{config.dataset}_{config.task_lm}_{config.index}_ILPS.csv")
        elif config.random_flag:
            pd.DataFrame(ans).to_csv(f"./{config.pareto_formulation}_{config.prob_formulation}_30_50_0.2_Wheel_k_U_entropy/final_{config.dataset}_{config.task_lm}_{config.index}_ILPS_random.csv")
        elif config.original_flag:
            pd.DataFrame(ans).to_csv(f"./{config.pareto_formulation}_{config.prob_formulation}_30_50_0.2_Wheel_k_U_entropy/final_{config.dataset}_{config.task_lm}_{config.index}_{config.prompt_type}_original.csv")
        elif config.LLMLingua_flag:
            pd.DataFrame(ans).to_csv(f"./{config.pareto_formulation}_{config.prob_formulation}_30_50_0.2_Wheel_k_U_entropy/final_{config.dataset}_{config.task_lm}_{config.index}_LLMLingua.csv")
        elif config.LLMLingua2_flag:
            pd.DataFrame(ans).to_csv(f"./{config.pareto_formulation}_{config.prob_formulation}_30_50_0.2_Wheel_k_U_entropy/final_{config.dataset}_{config.task_lm}_{config.index}_LLMLingua2.csv")
        elif config.attribution_flag:
            pd.DataFrame(ans).to_csv(f"/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/examples/few-shot-classification/evaluation/interpret-lm/{config.task_lm}_{config.dataset}_{config.index}_{config.attribution_type}_0.95_0.05_eval.csv")
        elif config.attention_first_flag:
            pd.DataFrame(ans).to_csv(f"/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/examples/few-shot-classification/evaluation/interpret-lm/{config.task_lm}_{config.dataset}_{config.index}_attention_first_0.95_0.05_eval.csv")
        elif config.attention_aggregate_flag:
            pd.DataFrame(ans).to_csv(f"/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/examples/few-shot-classification/evaluation/interpret-lm/{config.task_lm}_{config.dataset}_{config.index}_attention_aggregate_0.95_0.05_eval.csv")
        elif config.RS_flag:
            pd.DataFrame(ans).to_csv(f"/mnt/workspace/workgroup_dev/jianyu/test/rl-prompt/examples/few-shot-classification/evaluation/interpret-lm/subj_1000_llama3-it_{config.index}_eval.csv")
        elif config.example_shuffle_flag:
            pd.DataFrame(ans).to_csv(f"./{config.pareto_formulation}_{config.prob_formulation}_30_50_0.2_Wheel_k_U_entropy/final_{config.dataset}_{config.task_lm}_{config.index}_ILPS_random_{config.shuffle_seed}_{config.prompt_type}.csv")
                                   
        elif config.RS_shot_flag:
            pd.DataFrame(ans).to_csv(f"./{config.task_lm}_rs/{config.dataset}_{config.prompt_type}_{config.task_lm}_RS_1000_eval.csv")
        elif config.evoRS_shot_flag:
            pd.DataFrame(ans).to_csv(os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_ICLevolution_constrained_min_0_1shotwhole_100_sep_1000_eval.csv") )
        elif config.evo_parallel_flag:
            pd.DataFrame(ans).to_csv(os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_instruct_top100_True_False_eval.csv"))
        elif config.evo_parallel_mutate_flag:
            pd.DataFrame(ans).to_csv(os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_instruct_top100_True_False_mutate_eval.csv"))
        
        elif config.RLPrompt_flag:
            pd.DataFrame(ans).to_csv(os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_RLPrompt_{config.task_lm}.csv") )
        elif config.ILPS_random_flag:
            pd.DataFrame(ans).to_csv(os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_{config.task_lm}_random{config.ILPS_random_index}.csv") )
        elif config.orthogonal_original_flag:
            pd.DataFrame(ans).to_csv(os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_{config.task_lm}_orthogonal_original.csv"))
        elif config.orthogonal_pruned_flag:
            pd.DataFrame(ans).to_csv(os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_{config.task_lm}_orthogonal_pruned.csv"))
        elif config.pile_pruned_flag:
            pd.DataFrame(ans).to_csv(os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_{config.task_lm}_pile_pruned.csv"))
        elif config.pile_original_flag:
            pd.DataFrame(ans).to_csv(os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_{config.task_lm}_pile_original.csv"))
        elif config.natural_original_flag:
            pd.DataFrame(ans).to_csv(os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_{config.task_lm}_natural_original.csv"))
        elif config.natural_pruned_flag:
            pd.DataFrame(ans).to_csv(os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_{config.task_lm}_natural_pruned.csv"))
        elif config.nolabel_GGA_flag:
            pd.DataFrame(ans).to_csv(os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_{config.task_lm}_nolabelGGA_eval.csv"))
        elif config.nolabelsignal_GGA_flag:
            pd.DataFrame(ans).to_csv(os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_{config.task_lm}_nolabelsignalGGA_eval.csv"))
        elif config.noall_GGA_flag:
            pd.DataFrame(ans).to_csv(os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_{config.task_lm}_noallGGA_eval.csv"))
        elif config.randomlabel_flag:
            pd.DataFrame(ans).to_csv(os.path.join(prompt_dir, f"Turn1_{config.index}_{config.evol_seed}_{config.dataset}_{config.prompt_type}_{config.task_lm}_randomlabel_eval.csv"))
        elif config.compress_flag:
            pd.DataFrame(ans).to_csv(os.path.join(prompt_dir, f"Final_compression_{config.dataset}_{config.index}_llama3-it_0.98_eval.csv"))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num_shots", type=int, default=16)
    parser.add_argument("--evolution_seed", type=int, default=0)
    parser.add_argument("--sample_shots", type=int, default=8)
    parser.add_argument("--base_path", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="sst-2")
    parser.add_argument("--dataset_seed", type=int, default=0)
    parser.add_argument("--flag", type=int, default=0)
    parser.add_argument("--task_lm", type=str, default="gpt2")
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--TAPruning_threshold", type=float, default=0.96)
    parser.add_argument("--is_mask_lm", type=bool, default=False)
    parser.add_argument("--prompt_shot", type=int, default=1, choices=[1, 2, 4, 8])
    parser.add_argument("--prompt_type", type=str, default="Unpruned", 
                        choices=["Unpruned", "TAPruning", "PromptQuine-SSGA", "PromptQuine-GGA"])
    parser.add_argument("--mode", type=str, default="test-only", choices=["valid-only", "test-only", "valid-test"])
    parser.add_argument("--randomlabel", type=bool, default=False)
    parser.add_argument("--prepend_pile_sentences", type=bool, default=False)
    parser.add_argument("--prepend_task_instructions", type=bool, default=False)
    parser.add_argument("--prepend_orthogonal_instructions", type=bool, default=False)
    parser.add_argument("--limit_samples", type=int, default=-1, help="The number of samples we used for dataset evaluation")
    parser.add_argument("--limit_prompts", type=int, default=-1, help="The number of prompts we limited (PromptQuine)")
    parser.add_argument("--prompt", type=str, default=0, help="Specify like: '{sentence_1} It was'")
    parser.add_argument("--src_prompt_file_path", type=str, default=0, help="An input file path")
    parser.add_argument("--tgt_prompt_file_path", type=str, default=0, help="An output file path")
    args = parser.parse_args()
    main(args)

