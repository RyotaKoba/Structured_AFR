# Set common variables
# model="lmsys/vicuna-13b-v1.5"
model="meta-llama/Meta-Llama-3-8B"
#deepseek-ai/DeepSeek-R1-Distill-Llama-8B
#meta-llama/Meta-Llama-3-8B

python main.py \
--model $model \
--prune_method "done" \
--pruning_ratio 0.5 \
--nsamples 128 \
--eval