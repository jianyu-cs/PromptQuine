# Prompt Pruning for Few-shot Classification
We provide commands for running both our *TAPruning* and **PromptQuine** for standard ICL prompts here. 

Please check `pruner_config.yaml` for the relevant arguments and specify their values when running the commands below.

## 1. TAPruning for Few-shot Classification
### Guideline (TAPruning)
You can conduct TAPruning over (ICL) prompts on a given dataset with the following commands:

```
python prompt_prune.py \
  model.name="FacebookAI/roberta-large" \
  model.inference_engine="HF" \
  model.num_devices=1 \
  data.dataset=snli \
  pruning.reward_driven=False \
  pruning.algorithm="TAPruning" \
  pruning.fix_prune_order=True \
  pruning.TAPruning_threshold=0.96 \
  model.ICL_index=0 \
  model.ICL_shots=1 \
  data.dataset_seed=0
```

Please refer to `pruner_config.yaml` for more details!

You will obtain the full pruning traces in the same `PrunedPrompts_by_TAPruning/**/Train` folder.

### Evaluation (TAPruning)
After obtaining the prompt pruning traces, please run the following commands (`prompt_eval.py`) to obtain corresponding full evaluation (e.g., validation + testing) results.

```
python prompt_eval.py \
  model.name="FacebookAI/roberta-large" \
  model.inference_engine="HF" \
  model.num_devices=1 \
  data.dataset=snli \
  pruning.reward_driven=False \
  pruning.algorithm="TAPruning" \
  pruning.fix_prune_order=True \
  pruning.TAPruning_threshold=0.96 \
  model.ICL_index=0 \
  model.ICL_shots=1 \
  data.dataset_seed=0 \
  prompt.is_pruned_prompt = True 
```

You will obtain the full evaluation results in the same `PrunedPrompts_by_TAPruning/**/Eval` folder.

* Change `prompt.is_pruned_prompt` to evaluate unpruned original ICL prompts. 
* Note that you shall **specify the detailed hyperparameters** you have set for training. This ensures that our script will load correct prompt csv file.

## 2. PromptQuine for Few-shot Classification
If computational resources allow, in order to improve evolution's performance, please consider:
1. Increase the `prompt_quine.reproduction_size`;
2. Increase the `data.num_shots` and set `data.split=False`;
3. Increase the `prompt_quine.population_size`;

This principle applies to most of our experiments where suboptimal results are observed. For details, please refer to **our paper**; note that we slightly adjusted the settings for some challenging higher-shot ICL tasks.

### Guideline (PromptQuine)
You can conduct PromptQuine over (ICL) prompts on a given dataset with the following commands. For example,

```
python prompt_prune.py \
  model.name="meta-llama/Meta-Llama-3-8B-Instruct" \
  model.inference_engine="vLLM" \
  model.num_devices=1 \
  data.dataset=subj \
  pruning.algorithm="PromptQuine" \
  pruning.reward_driven=True \
  prompt_quine.algorithm_mode=SSGA \
  prompt_quine.initialize_duplicate=True \
  model.ICL_index=0 \
  model.ICL_shots=1 \
  data.dataset_seed=0
```

You will obtain the full pruning traces in the same `PrunedPrompts_by_PromptQuine/**/Train` folder.

Please refer to `pruner_config.yaml` for more details!

### Evaluation (PromptQuine)
After obtaining the prompt pruning traces, please run the following commands (`prompt_eval.py`) to obtain corresponding full evaluation (e.g., validation + testing) results.

```
python prompt_prune.py \
  model.name="meta-llama/Meta-Llama-3-8B-Instruct" \
  model.inference_engine="vLLM" \
  model.num_devices=1 \
  data.dataset=subj \
  pruning.algorithm="PromptQuine" \
  pruning.reward_driven=True \
  prompt_quine.algorithm_mode=SSGA \
  prompt_quine.initialize_duplicate=True \
  model.ICL_index=0 \
  model.ICL_shots=1 \
  data.dataset_seed=0 \
  prompt.is_pruned_prompt = True 
```

You will obtain the full evaluation results in the same `PrunedPrompts_by_PromptQuine/**/Eval` folder.

* Change `prompt.is_pruned_prompt` to evaluate unpruned original ICL prompts. 
* Note that you shall **specify the detailed hyperparameters** you have set for training. This ensures that our script will load correct prompt csv file.




