# Prompt Pruning for Unsupervised Text Style Transfer
We provide commands for running both our *TAPruning* and **PromptQuine** for standard ICL prompts here. 

Please check `pruner_config.yaml` for the relevant arguments and specify their values when running the commands below.

Please note that we do not explicitly optimize for inference throughput. Consequently, there is **room for further improvement** through additional techniques. This is why more sophisticated generation tasks (e.g., using Best-of-N sampling, etc) can still be quite time-consuming (e.g., taking hours), comparable to the runtime of other algorithms such as RLPrompt.

## Style Classifier
Please refer to [RLPrompt Tutorial](https://github.com/mingkaid/rl-prompt/tree/main/examples/text-style-transfer) to download the Style Classifiers into our `style_classifiers` folder.

## 1. TAPruning for Unsupervised Text Style Transfer
### Guideline (TAPruning)
You can conduct TAPruning over (ICL) prompts on a given dataset (w/ `greedy decoding`) with the following commands:

```
python prompt_prune.py \
  model.name="openai-community/gpt2" \
  model.num_devices=1 \
  data.direction="1_to_0" \
  data.max_size=200 \
  pruning.algorithm="TAPruning" \
  pruning.fix_prune_order=True \
  pruning.TAPruning_threshold=0.96 \
  model.ICL_index=0 \
  model.ICL_shots=1 \
  model.task_top_k=1 \
  model.num_samples=1
```

Please refer to `pruner_config.yaml` for more details!

You will obtain the full pruning traces in the same `PrunedPrompts_by_TAPruning/**/Train` folder.

### Evaluation (TAPruning)
After obtaining the prompt pruning traces, please run the following commands (`prompt_eval.py`) to obtain corresponding full evaluation (e.g., validation + testing) results.

```
python prompt_eval.py \
  model.name="openai-community/gpt2" \
  model.num_devices=1 \
  data.direction="1_to_0" \
  data.max_size=200 \
  pruning.algorithm="TAPruning" \
  pruning.fix_prune_order=True \
  pruning.TAPruning_threshold=0.96 \
  model.ICL_index=0 \
  model.ICL_shots=1 \
  model.task_top_k=1 \
  model.num_samples=1 \
  prompt.is_pruned_prompt = True 
```

You will obtain the full evaluation results in the same `PrunedPrompts_by_TAPruning/**/Eval` folder.

* Change `prompt.is_pruned_prompt` to evaluate unpruned original ICL prompts. 
* Note that you shall **specify the detailed hyperparameters** you have set for training. This ensures that our script will load correct prompt csv file.

## 2. PromptQuine for Unsupervised Text Style Transfer
If computational resources allow, in order to improve evolution's performance, please consider:
1. Increase the `prompt_quine.reproduction_size`;
2. Increase the `data.max_size`;
3. Increase the `prompt_quine.population_size`;

This principle applies to most of our experiments where suboptimal results are observed. For details, please refer to **our paper**; note that we slightly adjusted the settings for some challenging higher-shot ICL tasks.

### Guideline (PromptQuine)
You can conduct PromptQuine over (ICL) prompts on a given dataset (w/ `greedy decoding`) with the following commands. For example,

```
python prompt_prune.py \
  model.name="openai-community/gpt2" \
  model.num_devices=1 \
  data.direction="1_to_0" \
  data.max_size=100 \
  pruning.algorithm="PromptQuine" \
  prompt_quine.algorithm_mode=SSGA \
  prompt_quine.initialize_duplicate=True \
  prompt_quine.successive_halving=True \
  model.ICL_index=0 \
  model.ICL_shots=1 \
  model.task_top_k=1 \
  model.num_samples=1
```

You will obtain the full pruning traces in the same `PrunedPrompts_by_PromptQuine/**/Train` folder.

Please refer to `pruner_config.yaml` for more details!

### Evaluation (PromptQuine)
After obtaining the prompt pruning traces, please run the following commands (`prompt_eval.py`) to obtain corresponding full evaluation (e.g., validation + testing) results.

```
python prompt_prune.py \
  model.name="openai-community/gpt2" \
  model.num_devices=1 \
  data.direction="1_to_0" \
  data.max_size=100 \
  pruning.algorithm="PromptQuine" \
  prompt_quine.algorithm_mode=SSGA \
  prompt_quine.initialize_duplicate=True \
  prompt_quine.successive_halving=True \
  model.ICL_index=0 \
  model.ICL_shots=1 \
  model.task_top_k=1 \
  model.num_samples=1 \
  prompt.is_pruned_prompt = True 
```

You will obtain the full evaluation results in the same `PrunedPrompts_by_PromptQuine/**/Eval` folder.

* Change `prompt.is_pruned_prompt` to evaluate unpruned original ICL prompts. 
* Note that you shall **specify the detailed hyperparameters** you have set for training. This ensures that our script will load correct prompt csv file.




