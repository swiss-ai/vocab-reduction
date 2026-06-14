# Evaluation pipeline

This file explains how to launch the evaluation pipeline.
Quick usage:
```
sbatch scripts/evaluate.sbatch <model> <iteration> <tokens-per-iter> <name>
```
This will evaluate `<model>` at iteration `<iteration>`, and save it under the `<name>` logs.
The `<model>` can be either the path of a megatron checkpoint or the name/path of a huggingface model.
Once this slurm job finishes you can push the results to wandb using:
```
python scripts/update_wandb.py $LOGS_ROOT
```
By default, `LOGS_ROOT=$SCRATCH/eval-logs`, but you can customize `evaluate.sbatch` to save the logs somewhere else (see next section).
Look at `examples/` for a bunch of examples used to create: https://wandb.ai/epflmlo-epfl/swissai-eval-main-v1.5/workspace?nw=nwuserepflmlo.

## Customizing `evaluate.sbatch`

The number of arguments needed to run this `evaluate.sbatch` will always be four positional arguments:
1. `<model>`: The model to be evaluated.
   If it is a megatron model, it is expected to be the path where the checkpoint is located.
   For huggingface checkpoints, it can be either the model name (in the hub) or path.
   Important for megatron models: Do not specify the `iter_xyz` in the path, e.g. `path/to/checkpoint/iter_000010` is an incorrect path and you should use `path/to/checkpoint` instead.
1. `<iteration>`: Integer specifying the iteration of the model you want to evaluate.
   In the case of megatron models, loading this iteration is enforced internally via the `--ckpt-step` when converting the model.
   In the case of huggingface models, it is only used to organize the `LOGS_ROOT` hierarchy and to calculate the `CONSUMED_TOKENS` internally (see next argument).
1. `<tokens-per-iter>`: Either an integer or a "tok_per_it specification" to determine how many tokens are consumed per iteration.
   If it is an integer, then the calculation simplifies to `CONSUMED_TOKENS = iteration * tokens_per_iter`.
   The other specification must be a set of `tokens_per_iter_on_this_stage:max_iteration_on_this_stage` strings concatenated by `,` where (optionally) the last string might not need the `max_iteration_on_this_stage`, in which case it is assumed that the last stage lasts forever.
   For instance the specification `256:1000,512:3000,1024:` will have three stages where iterations 1 to 1000 have 256 tokens_per_iteration, 1001 to 3000 have 512 tokens_per_iteration and iterations from 3001 on will have 1024 tokens per iteration.
   If under this specification the iteration 3050 is evaluated for instance, the total consumed tokens will be `256*1000 + 512*(3000 - 1000) + 1024*(3050 - 3000)`.
1. `<name>`: The unique model name you associate `<model>` with (e.g. "Apertus-70B" or "SmolLM2-1.7B").
   We have this option because sometimes different `<model>` paths reference the same model family/name (e.g. when Apertus training paths change).
   All runs (regardless of `<model>` value) with the same `<model>` tag will be associated together in the `LOGS_ROOT` directory.

In addition of the four needed positional arguments, you can specify the following environmental variables (all are optional unless explicitly mentioned otherwise):
- `LOGS_ROOT`: Path where all the logs will be saved.
  The (wandb & harness) information for each run will be saved to `LOGS_ROOT/<name>/iter_<iteration>`.
  Using this structure `update_wandb.py` can aggregate various harness runs for the same `<name>` and `<iteration>` and push them to wandb.
- `BS`: Batch size.
- `REVISION`: Only used with huggingface models.
- `TOKENIZER`: A huggingface tokenizer path/name.
   Needed for megatron checkpoint, otherwise the default will be set as the same value as `<model>`.
- `HF_TEMP_DIR`: If set, the converted megatron checkpoints will be saved here, otherwise they will not be saved anywhere.
- `MEGATRON_BRANCH`: Only used in megatron checkpoints.
   The megatron->huggingface conversion scripts will be looked at this branch.
- `HARNESS_BRANCH`: lm-evaluation-harness branch to install."
- `TRANSFORMERS_BRANCH`: transformers branch to install."
- `SIZE`: The (approximate) size of the model in billions of parameters.
   Used to set model parallelism (needs to be set in larger models, otherwise you will run out of CUDA memory when evaluating).
- `LIMIT`: The --limit argument to pass to lm-evaluation-harness.
- `BOS`: Set this to `true` if you wish to prepend the BOS token when evaluating models.
- `TASKS`: Tasks to run with lm eval harness.

## Customizing `update_wandb.py`

Whenever you are running `update_wandb.py` make sure you have your `WANDB_API_KEY`, `WANDB_ENTITY` and `WANDB_PROJECT` variables enabled.
By default, the script will look for all models and iterations in the specified `LOGS_ROOT`, aggregate all benchmarks result and push them to the specified project.
This can take very long when you have many evaluations in `LOGS_ROOT`.
You can choose to update only a subset of results using `--name` and `--it`.
When specified, only the results that match the value given will be aggregated and updated in wandb.

## End-to-End Example

The following snippet will evaluate an intermediate checkpoint of OLMo2-7B and push it to the wandb.
```bash
NAME=OLMo2-7B
TOK_PER_IT=4194304 
IT=50000
TOKENS_IN_B=$(( ( IT*TOK_PER_IT + 1000000000 )/1000000000 ))B  # Needed for the REVISION, but not directly for the `evaluate.sbatch`

export REVISION=stage1-step$IT-tokens${TOKENS_IN_B}B
export LOGS_ROOT=$SCRATCH/eval-logs  # Set explicitly as an example, this is the default value in the script.
sbatch scripts/evaluate.sbatch allenai/OLMo-2-1124-7B $IT $NAME

# Wait for the slurm job to finish...

# Don't forget to set your WANDB_API_KEY, then run:
WANDB_PROJECT=swissai-eval-main-v1.5 WANDB_ENTITY=epflmlo-epfl python3 scripts/update_wandb.py --it $IT --name $NAME
```

In case you want to evaluate larger models, you will need to (1) specify `SIZE` to use a correct model parallel value and not run out of memory, and (2) split the evaluation tasks in two (or more) so the job finishes in under 12h.
Example:
```bash
NAME=Apertus-3B
TOK_PER_IT=8388608:523519,16777216:  # The Apertus70B model doubled GBS around iteration 523k
IT=750000
PATH=/capstor/scratch/cscs/asolergi/main_run_70B_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-70b-512-nodes-1e-5lr/checkpoints-512-noOverlap/

export BOS=true  # Needed to correctly evaluate Apertus models.
export SIZE=70  # Needed to set model parallelism internally, otherwise you will run out of CUDA memory.
export LOGS_ROOT=$SCRATCH/eval-logs  # Set explicitly as an example, this is the default value in the script.

# Run both tasks separately.
TASKS=english sbatch scripts $PATH $IT $NAME
TASKS=multilingual_pt1 sbatch scripts $PATH $IT $NAME
TASKS=multilingual_pt2 sbatch scripts $PATH $IT $NAME

# Once both jobs finish you can aggregate the results and push them to the wandb by using:
WANDB_PROJECT=swissai-eval-main-v1.5 WANDB_ENTITY=epflmlo-epfl python3 scripts/update_wandb.py --it $IT --name $NAME
```

## Regarding task separation

This is the hierarchy that defines the default `swissai_eval`:
- swissai_eval (100%)
  - english  (~46%)
    - english_pt1 (~10%)
    - english_pt2 (~36%)
  - multilingual (~54%)
    - multilingual_pt1 (~8%)
    - multilingual_pt2 (~46%)

The percentages shown represent the approximate time taken for each task group to finish (ignoring data loading).
Depending on the size of the model you are evaluating, you might need to split your evaluation run into different jobs in parallel in order to finish within the 12h limit.
A good rule of thumb is to make sure that the task (sub)groups you select finish on time is to ensure `2.5 * percentage * model_size < 100`, where `model_size` is in billions.
Examples:
- A 30B model can complete all tasks in one go: `2.5 * 1.0 * 30 = 75`.
- A 70B model should reliably finish when splitting into two: `english`, `multilingual` with heuristic scores of `80` and `94.5`, respectively.

More information on the task hierarchy refer to https://github.com/swiss-ai/lm-evaluation-harness/tree/main/lm_eval/tasks/swissai_eval.

## About VLLM

By default, vllm is enabled by default for faster inference.
To recover the old behaviour using transformers inference, you can run `export BACKEND=hf`.

>[!NOTE]
> Results of generation tasks (such as gsm8k and squadv2) change drastically between inference engines.
> Make sure  you only compare results of such tasks when all models use the same backend.
> Results of likelihood tasks (e.g. hellaswag) may also change when switching inference engines, but not as much.
> Currently, all generation tasks in the [main dashboard](https://wandb.ai/epflmlo-epfl/swissai-eval-main-v1.6) are evaluated with VLLM.

## Using `scripts/automate.py`

The easiest way to evaluate many models is to use the `scripts/automate.py` script.
This will perform the following:
1. Look for the automation configuration file (`configs/automation.json` by default, specify a different one with `--config-path`) to get the general settings (e.g. evaluation logsroot, HF home, etc) and the list of models to evaluate.
2. It will then request `src/evals/tasks.py:get_all_tasks` for a list of tasks to evaluate.
  This list is inferred from `configs/all_tasks.json`.
3. For all models specified in `configs/automation.json`, it will attempt to locate the set of tasks that have been already evaluated by looking at the `logs_root`.
4. If there are tasks obtained in step 2 that haven't been evalauted by a model, it will launch (one or more) job(s) with all missing tasks using `scripts/evaluate.sh`.
5. If the script was launched with `--sync`, it will also sync the wandb project in the end.

Make sure you run this script with `HF_TOKEN` and `WANDB_API_KEY` defined.
You will also need a few not-so-common python dependencies like `iso639-lang` and `prtpy` (check the `pyproject.toml`), which can be easily resolved by using `uv`.

IMPORTANT!
If you want to run newer models not compatible with the default version of vllm, you can specify the `--use-official-vllm` flag.
For instance, the fp8-quantized Apertus can be evaluated running
```
uv run scripts/automate.py --use-official-vllm --config-path=configs/automation-fp8.json
```

## About `evaluate_hf.sbatch`
This file is here to keep things self contained, it will work great with the `Dockerfile` and `env.toml ` specified in the `containers` folder, only for huggingface models though. Use this to ensure stable HF evals if the other `evaluate.sbatch` fails. Supports `vllm`.
```bash
TASKS=configs/alignment/tasks_english.txt bash examples/eval_apertus_8b_hf.sh
```