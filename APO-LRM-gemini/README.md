
# Introduction

This is code is based on the following paper [Automatic Prompt Optimization with "Gradient Descent" and Beam Search](https://arxiv.org/abs/2305.03495) (EMNLP 2023).

The main entrypoint is `main.py`

# Quickstart:

## Env setup
Before running the following script, please modify the `prefix` in the `env.yaml` file to match your local setup.
```
conda env create -f env.yaml
```

## Launch a model
We use the SGLang framework to launch a model inference server locally. If you encounter any issues during installation, please refer to the [official GitHub repository](https://github.com/sgl-project/sglang).
```
bash scripts/launch/deepseek-r1-1.5B.sh $GPUS
```

## Run experiments
```
bash scripts/run.sh
```

This will run an experiment with UCB bandits for candidate selection. The program will print configuration settings and provide progress updates with each optimization round. The results, including candidate prompts and their associated scores, will be written to the specified output file.


For usage instructions. Some of the arguments include:

* `--task`: Task name like 'ethos', 'jailbreak', etc.
* `--data_dir`: Directory where the task data resides.
* `--prompts`: Path to the prompt markdown file.
* `--out`: Output file name.
* `--max_threads`: Maximum number of threads to be used.
* `--gradient_engine`: The model performing gradient & modification steps.
* `--engine`: The model making predictions.
* `...`: Various other parameters related to optimization and evaluation.