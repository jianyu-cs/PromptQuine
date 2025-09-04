# Prompt Pruning for Few-shot Classification
We provide commands for running both our *TAPruning* and **PromptQuine** for traditional ICL prompts here. 

## TAPruning for Few-shot Classification
You can conduct TAPruning over (ICL) prompts on a given dataset with the following commands:
```
python prompt_prune.py \
    --model=FacebookAI/roberta-large \
    --is_mask_lm=True \
    --dataset=[sst-2, subj, agnews, snli, yelp-5, yahoo, piqa] \
    --pruner=TAPruning \
    --fix_prune_order=True \
    --TAPruning_threshold=0.96 \
    --ICL_shots=1 \
    --ICL_index=0
```

* Note that you can adapt your own prompts, dataset and verbalizers by modifying our code.

After obtaining the prompt pruning traces, please run the following command to obtain corresponding full validation (maybe with testing) results:
* We provide options of both testing the entire prompts from the pool and testing only the selected prompt (maybe biased): 
