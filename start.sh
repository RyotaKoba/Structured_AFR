# Set common variables
model="meta-llama/Meta-Llama-3-8B"
#deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# meta-llama/Meta-Llama-3-8B
# model="lmsys/vicuna-13b-v1.5"
# CUDA_LAUNCH_BLOCKING=1 python analyzer.py

CUDA_LAUNCH_BLOCKING=1 python main.py \
--model $model \
--prune_method "structured_afr" \
--pruning_ratio 0.5 \
--nsamples 128 \
--a 1  \
--b 1  \
--c 1  \
--cuda \
--global_metrics angular \
--local_metrics three_w_one_wa \
--save_model "./pruned_model/trash" \
# --protect_sw
# --pruning_ration : 枝刈り率
