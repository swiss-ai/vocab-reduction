"""Automatic evaluations
Usage (in a tmux session that never ends):
```
while true; do
    python scripts/automate.py
    sleep $(( 60*60 ))
done
```
Make sure to export your WANDB_API_KEY and LOGS_ROOT.
"""
from __future__ import annotations

import argparse
import collections
import re
import os
import math
import json
import subprocess
import shutil
from pathlib import Path

from evals.tasks import Task, get_all_tasks, get_partition


def get_max_samples(size: int) -> int:
    return max(filter(lambda t: int(t[0]) <= size, CFG["max_samples"].items()), key=lambda t: int(t[0]))[1]

def get_running(as_jobname: bool = False) -> dict[str, dict[int, list[str]] | list[str]]:
    proc = subprocess.run(["squeue", "--me", '--format="%j"', "--noheader"],
                          capture_output=True, text=True)
    assert proc.returncode == 0

    jobnames = proc.stdout.strip().split("\n")
    if jobnames == [""]:
        return [] if as_jobname else collections.defaultdict(lambda: collections.defaultdict(list))
    jobnames = [re.match('^"(.*)"$', jobname).group(1) for jobname in jobnames]

    running = [] if as_jobname else collections.defaultdict(lambda: collections.defaultdict(list))
    for jobname in jobnames:
        rmatch = re.match(r"^eval_(.*)_([a-zA-Z0-9]+)_([0-9]+)$", jobname)
        if rmatch is not None:
            if as_jobname:
                running.append(jobname)
            else:
                name, group, it = rmatch.groups()
                running[name][int(it)].append(group)
    return running


def get_evaluated(model: str) -> dict[int, list[str]]:
    status = collections.defaultdict(list)
    for path in Path(CFG["logs_root"]).glob(f"{model}/iter_*/harness/eval_*/*/results*.json"):
        it = int(re.match("^iter_([0-9]+)$", path.parent.parent.parent.parent.name).group(1))
        with open(path) as f:
            info = json.load(f)
        for taskname in info["results"]:
            status[it].append(taskname)
    return status


def get_available(model_dirs: list[Path]) -> list[int]:
    available = []
    for model_dir in model_dirs:
        for path in filter(Path.is_dir, Path(model_dir).iterdir()):
            available.append(int(re.match("^iter_([0-9]+)$", path.name).group(1)))
    return available


def submit(name: str, model: dict, it: int, tasks: list[Task],
           model_path: str, extra_env: dict[str, str] = {},
           use_official_vllm: bool = False):

    # Get partition of tasks.
    total_size = sum(task.size for task in ALL_TASKS)
    max_samples = get_max_samples(model.get("size", 1))
    n_def_shards = math.ceil(total_size/max_samples)
    default_partition = get_partition(tasks=ALL_TASKS, shards=n_def_shards)

    # Check all parts of the default partition, if any of them is 
    # completely contained in the tasks requested, launch that shard.
    shards_to_launch = []
    for shard_i, part in enumerate(default_partition):
        if set(part) <= set(tasks):
            shards_to_launch.append((shard_i, part))
            tasks = [task for task in tasks if task not in part]

    # For all remaining tasks that don't match perfectly a default part,
    # create a "mixed" job submission.
    total_size = sum(task.size for task in tasks)
    n_shards = math.ceil(total_size/max_samples)
    partition = get_partition(tasks=tasks, shards=n_shards)
    for part in partition:
        shards_to_launch.append(("mixed", part))

    # Schedule all tasks requested.
    for shard_i_or_mixed, tasks_to_launch in shards_to_launch:
        if shard_i_or_mixed == "mixed":
            jobname = "mixed"
        else:
            jobname = f"shard{shard_i_or_mixed}of{n_def_shards}"
        jobname = f"eval_{name}_{jobname}_{it}"

        if use_official_vllm:
            cmd = ["sbatch", "--environment=./containers/env-official.toml"]
        else:
            cmd = ["sbatch", "--environment=./containers/env.toml"]

        cmd += [f"--job-name={jobname}", "scripts/evaluate.sbatch", model_path,
                str(it), model["tokens_per_iter"], name] 

        env = {**os.environ,
               "LOGS_ROOT": CFG["logs_root"],
               "SIZE": str(model.get("size", 1)),
               "TASKS": ",".join(task.name for task in tasks_to_launch)}
        env.update(extra_env)
        if use_official_vllm:
            env.update({
                "HARNESS_FORK": "https://github.com/EleutherAI/lm-evaluation-harness.git",
                "HARNESS_BRANCH": "main"
            })

        maybe_show = [task.name for task in tasks_to_launch]
        if len(maybe_show) > 32 or "mixed" not in jobname:
            maybe_show = ""
        print("Launching", jobname, maybe_show)
        subprocess.run(cmd, env=env, stdout=subprocess.PIPE)


def submit_needed(force_tasks: list[str], use_official_vllm: bool):
    def get_missing(it: int) -> list[Task]:
        handled = status.get(it, [])
        missing = []
        for task in ALL_TASKS:
            if len(task.alias) > 0 and any(actual_name not in handled for actual_name in task.alias):
                missing.append(task)
            elif len(task.alias) == 0 and task.name not in handled:
                missing.append(task)
            elif task.name in force_tasks:
                missing.append(task)
        return missing

    running = get_running()
    for name, model in CFG["models"].items():
        total_size = sum(task.size for task in ALL_TASKS)
        max_samples = get_max_samples(model.get("size", 1))
        n_shards = math.ceil(total_size/max_samples)

        # Get tasks alredy evaluated (reading them from the `results.json`).
        status = get_evaluated(name)
        default_partition = get_partition(tasks=ALL_TASKS, shards=n_shards)

        # Handle already evaluated: if a "mixed" group is running, assume it will
        # contain all missing tasks because we don't know which one does it contain in reality,
        # otherwise obtain the correct shard.
        for it, groups in running[name].items():
            for group in groups:
                if group == "mixed":
                    actual_tasks = ALL_TASKS
                else:
                    shard_i, total_shards = re.match("^shard([0-9]+)of([0-9]+)$", group).groups()
                    assert int(total_shards) == n_shards
                    actual_tasks = default_partition[int(shard_i)]

                expected_names = []
                for task in actual_tasks:
                    expected_names += [task.name] if len(task.alias) == 0 else list(task.alias)

                if it in status:
                    status[it] += expected_names
                else:
                    status[it] = expected_names

        if "model_dirs" in model:  # Megatron checkpoints where iterations are taken on the fly.
            available = get_available(model["model_dirs"])
            for it in available:
                if (it - model["start_eval_from"]) % model["frequency"] == 0 and it >= model["start_eval_from"] or it in model.get("force_iters", []):
                    missing = get_missing(it)
                    if len(missing) > 0:
                        paths = [model_dir for model_dir in model["model_dirs"]
                                 if Path(f"{model_dir}/iter_{it:07d}").exists()]
                        if len(paths) != 1:
                            raise ValueError(f"Model {name} has {len(paths)} paths for iter {it} (should be =1): {paths}")
                        path, = paths
                        extra_env = {"EXTRA_PIPS": "nvidia-modelopt==0.27.0", "TOKENIZER": "alehc/swissai-tokenizer",
                                     "BOS": "true", "HF_TEMP_DIR": CFG["hf_temp_dir"]}
                        submit(name, model, it, missing, str(path), extra_env)
        else:  # Huggingface checkpoints.
            for i in range(len(model["iters"])):
                extra = model.get("extra_env", {})
                it = model["iters"][i]
                if "revisions" in model and model["revisions"][i] is not None:
                    extra["REVISION"] = model["revisions"][i]
                # Determine missing set.
                missing = get_missing(it)
                if len(missing) > 0:
                    submit(name, model, it, missing, model["name"], extra, use_official_vllm)


def update_hf_checkpoints():
    jobnames = get_running(as_jobname=True)
    for path in Path(CFG["hf_temp_dir"]).iterdir():
        if path.name in jobnames:  # Don't touch hf checkpoints of unfinished runs.
            continue
        name, it = re.match("^eval_(.*)_.*_([0-9]+)$", path.name).groups()
        it = int(it)
        dest = Path(CFG["hf_storage_dir"])/f"{name}_it{it}"
        if dest.exists():  # Checkpoint is already stored, probably from a job with different tasks that finished earlier.
            print("Removing", path)
            shutil.rmtree(path)
        else:
            print("Moving", path, "to", dest)
            shutil.move(path, dest)


def cleanup_hf_checkpoints():
    # Get model=>[(it, path)] mapping.
    stored = collections.defaultdict(list)
    for path in Path(CFG["hf_storage_dir"]).iterdir():
        rmatch = re.match("^(.*)_it([0-9]+)$", path.name)
        if rmatch is not None and rmatch.group(1) in CFG["models"]:
            name, it = rmatch.groups()
            stored[name].append((int(it), path))

    # Remove old checkpoints.
    keep = CFG["num_hf_checkpoints_to_keep"]
    for saved in stored.values():
        remove = sorted(saved, key=lambda t: t[0])[:-keep]
        for _, path in remove:
            print("Removing", path)
            shutil.rmtree(path)


def sync_wandb():
    print("Syncing wandb...")
    env = {**os.environ,
           "WANDB_SILENT": "true",
           "WANDB_RESUME": "allow",
           "WANDB_ENTITY": CFG["wandb_entity"],
           "WANDB_PROJECT": CFG["wandb_project"]}
    cmd = ["python3", "scripts/update_wandb.py", str(CFG["logs_root"]), "--names"]
    cmd += sorted(CFG["models"])
    subprocess.run(cmd, env=env)


def main(force_tasks: list[str], use_official_vllm: bool, sync: bool):
    submit_needed(force_tasks, use_official_vllm)
    #update_hf_checkpoints()
    #cleanup_hf_checkpoints()
    if sync:
        sync_wandb()


if __name__ == "__main__":
    # Argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=Path, default=Path("configs/automation.json"))
    parser.add_argument("--force-tasks", nargs="*", default=[])
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--use-official-vllm", action="store_true")
    args = parser.parse_args()
    
    # Get general config and launch.
    ALL_TASKS = get_all_tasks()
    with open(args.config_path) as f:
        CFG = json.load(f)
    del args.config_path

    # Remove non-official tassks when official vllm container was requested.
    if args.use_official_vllm:
        noadd = ["blend", "switzerland_qa", "include_base_new_45", "cultural_bench"]
        ALL_TASKS = list(filter(lambda task: all(no not in task.name for no in noadd), ALL_TASKS))
    main(**vars(args))
