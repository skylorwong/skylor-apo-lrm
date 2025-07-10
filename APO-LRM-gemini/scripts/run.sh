#engine=sglang_deepseek_r1_1.5b
#gradient_engine=gpt-4.1-mini-2025-04-14gemini-2.5-flash
host=http://localhost:30000
engine=gemini-1.5-flash
gradient_engine=gemini-1.5-flash
n_train_exs=1024
minibatch_size=32
samples_per_eval=64

out=exps/math/deepseek1.5_gpt-4.1-mini_mb=${minibatch_size}_se=${samples_per_eval}_ntr=${n_train_exs}.txt

python main.py --task math \
    --prompts prompts/math.md \
    --data_dir data/math \
    --out $out \
    --evaluator ucb \
    --max_threads 96 \
    --engine $engine \
    --n_train_exs $n_train_exs \
    --n_test_exs 100 \
    --minibatch_size $minibatch_size \
    --gradient_engine $gradient_engine \
    --num_rollout 1 \
    --engine_temperature 0.6 \
    --scorer math_verify \
    --samples_per_eval $samples_per_eval \
    --host $host