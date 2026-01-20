# Prompt Pruning for Jailbreaking
We provide commands for running both our *TAPruning* and **PromptQuine** for standard ICL prompts here. 

Please check `pruner_config.yaml` for the relevant arguments and specify their values when running the commands below.

Note:
* The current scripts require at least two GPUs for inference, e.g., if `pruning.pruning_metric` in `["ASR-EM", "ASR-LLM"]`. Obviously, one GPU is required to support prompted inference model, and another GPU is required to support Llama-Guard to support ASR-LLM calculation;
* If `pruning.pruning_metric` is `"ASR-SV`, we have to use at least three GPUs for now. Please try to modify our ray configurations or our entire script for more flexible adjustments; 

## 1. TAPruning for Jailbreaking
### Guideline (TAPruning)
You can conduct TAPruning over (ICL) prompts on `Advbench` (w/ `greedy decoding`) with the following commands:

```
python prompt_prune.py \
  model.name="lmsys/vicuna-7b-v1.5" \
  pruning.algorithm="TAPruning" \
  pruning.fix_prune_order=True \
  pruning.TAPruning_threshold=0.96 \
  model.ICL_index=0 \
  model.ICL_shots=1 \
  model.priming=True \
  model.gpus_per_bundle=1 \
  model.cpus_per_bundle=8 \
  pruning.pruning_metric="ASR-EM"
```

You will obtain the full pruning traces in the same `PrunedPrompts_by_TAPruning/**/Train` folder.

Please refer to `pruner_config.yaml` for more details!

Please consider to adjust `model.priming=False` and `pruning.pruning_metric` to support `in-context attacks` with more varied settings.

### Evaluation (TAPruning)
After obtaining the prompt pruning traces, please run the following commands (`prompt_eval.py`) to obtain corresponding full evaluation (e.g., validation + testing) results.

```
python prompt_eval.py \
  model.name="lmsys/vicuna-7b-v1.5" \
  pruning.algorithm="TAPruning" \
  pruning.fix_prune_order=True \
  pruning.TAPruning_threshold=0.96 \
  model.ICL_index=0 \
  model.ICL_shots=1 \
  model.priming=True \
  model.gpus_per_bundle=1 \
  model.cpus_per_bundle=8 \
  pruning.pruning_metric="ASR-EM" \
  prompt.is_pruned_prompt = True 
```

You will obtain the full evaluation results in the same `PrunedPrompts_by_TAPruning/**/Eval` folder.

* Change `prompt.is_pruned_prompt` to evaluate unpruned original ICL prompts. 
* Note that you shall **specify the detailed hyperparameters** you have set for training. This ensures that our script will load correct prompt csv file.

## 2. PromptQuine for Jailbreaking Tasks
If computational resources allow, in order to improve evolution's performance, please consider:
1. Increase the `prompt_quine.reproduction_size`;
2. Increase the `data.max_size`;
3. Increase the `prompt_quine.population_size`;

This principle applies to most of our experiments where suboptimal results are observed. For details, please refer to **our paper**; note that we slightly adjusted the settings for some challenging higher-shot ICL tasks.

### Guideline (PromptQuine)
You can conduct PromptQuine over (ICL) prompts on a given dataset (w/ `greedy decoding`) with the following commands. For example,

```
python prompt_prune.py \
  model.name="lmsys/vicuna-7b-v1.5" \
  pruning.algorithm="PromptQuine" \
  prompt_quine.algorithm_mode=SSGA \
  prompt_quine.initialize_duplicate=True \
  model.ICL_index=0 \
  model.ICL_shots=1 \
  model.priming=True \
  model.gpus_per_bundle=1 \
  model.cpus_per_bundle=8 \
  pruning.pruning_metric="ASR-EM"
```

You will obtain the full pruning traces in the same `PrunedPrompts_by_PromptQuine/**/Train` folder.

Please refer to `pruner_config.yaml` for more details!

Please consider to adjust `model.priming=False` and `pruning.pruning_metric` to support `in-context attacks` with more varied settings.

### Evaluation (PromptQuine)
After obtaining the prompt pruning traces, please run the following commands (`prompt_eval.py`) to obtain corresponding full evaluation (e.g., validation + testing) results.

```
python prompt_eval.py \
  model.name="lmsys/vicuna-7b-v1.5" \
  pruning.algorithm="PromptQuine" \
  prompt_quine.algorithm_mode=SSGA \
  prompt_quine.initialize_duplicate=True \
  model.ICL_index=0 \
  model.ICL_shots=1 \
  model.priming=True \
  model.gpus_per_bundle=1 \
  model.cpus_per_bundle=8 \
  pruning.pruning_metric="ASR-EM"
  prompt.is_pruned_prompt = True
```

You will obtain the full evaluation results in the same `PrunedPrompts_by_PromptQuine/**/Eval` folder.

* Change `prompt.is_pruned_prompt` to evaluate unpruned original ICL prompts. 
* Note that you shall **specify the detailed hyperparameters** you have set for training. This ensures that our script will load correct prompt csv file.




