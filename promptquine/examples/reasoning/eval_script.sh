CUDA_VISIBLE_DEVICES=3 python prompt_eval.py \
  data.dataset="gsm8k" \
  data.max_size=100 \
  model.name="meta-llama/Meta-Llama-3-8B-Instruct" \
  model.num_devices=1 \
  pruning.algorithm="PromptQuine" \
  prompt_quine.algorithm_mode=GGA \
  prompt_quine.initialize_duplicate=True \
  prompt_quine.successive_halving=True \
  model.ICL_index=0 \
  model.ICL_shots=1 \

CUDA_VISIBLE_DEVICES=3 python prompt_eval.py \
  data.dataset="mawps" \
  data.max_size=100 \
  model.name="meta-llama/Meta-Llama-3-8B-Instruct" \
  model.num_devices=1 \
  pruning.algorithm="PromptQuine" \
  prompt_quine.algorithm_mode=GGA \
  prompt_quine.initialize_duplicate=True \
  prompt_quine.successive_halving=True \
  model.ICL_index=0 \
  model.ICL_shots=1 \

