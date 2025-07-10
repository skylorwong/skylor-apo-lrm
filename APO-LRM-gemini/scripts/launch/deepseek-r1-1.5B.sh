set -e
set -x
#!/bin/bash

export HF_HOME="/nfs/data/sohyun/huggingface"

model_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
tokenizer_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
dtype=bfloat16
mem_fraction_static=0.87

gpu=$1
port=30000

IFS=',' read -ra GPU_ARRAY <<< "$gpu"

tensor_parallel_size=${#GPU_ARRAY[@]}
echo $tensor_parallel_size

CUDA_VISIBLE_DEVICES=$gpu python3 -m sglang.launch_server --model-path $model_path --tokenizer-path $tokenizer_path \
    --port $port --tp-size $tensor_parallel_size --trust-remote-code --dtype $dtype --mem-fraction-static $mem_fraction_static
