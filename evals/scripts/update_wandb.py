import collections
import statistics
import re
import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import wandb
import iso639

from evals.tasks import get_all_tasks, Task

INVALID_NUM = -1.0  # or -float("inf")

def get_log(infos: List[dict], tasks_cfg: dict, all_tasks: list[Task]) -> dict[str, float]:
    def agg(log: dict[str, dict[str, float]], prefix: str, tasks_to_agg: list[str], warn: bool = True,
            macro: bool = True, micro: bool = True, no_micro_suffix: bool = False,
            no_macro_suffix: bool = False, metrics: Optional[list[str]] = None):

        missing = set(tasks_to_agg) - set(log)
        macro_name = prefix if no_macro_suffix else f"{prefix}.macro"
        micro_name = prefix if no_micro_suffix else f"{prefix}.micro"
        if len(missing) > 0:
            if warn:
                print("WARNING! Aggregation for", prefix, "not available. Missing:", sorted(missing))
            for metric in metrics if metrics is not None else ["acc"]:
                if macro:
                    log[macro_name][metric] = INVALID_NUM
                if micro:
                    log[micro_name][metric] = INVALID_NUM
            return

        for metric in filter(lambda metric: "stderr" not in metric and (metrics is None or metric in metrics), all_metrics):
            values = [log[taskname][metric] for taskname in tasks_to_agg if metric in log[taskname]]
            sizes = [true_sizes[taskname] for taskname in tasks_to_agg if metric in log[taskname]]
            if len(values) > 0:
                if macro:
                    log[macro_name][metric] = statistics.mean(values)
                if micro:
                    log[micro_name][metric] = sum(value*size for value, size in zip(values, sizes))/sum(sizes)

    def pretty_dim(dim: str) -> str:
        return " ".join(dim.split("_")).title()


    # Aggregate raw info.
    groups = {}
    results = {}
    true_sizes = {}
    all_metrics = set()
    for info in infos:
        groups.update(info["group_subtasks"])
        results.update(info["results"])
        true_sizes.update({name: details["effective"] for name, details in info["n-samples"].items()})

    # Aggregate true_sizes if needed.
    for taskname in results:
        if taskname not in true_sizes:
            true_sizes[taskname] = sum(size for other_taskname, size in true_sizes.items()
                                       if other_taskname.startswith(taskname))

    # Prepare final logs.
    log = collections.defaultdict(dict)
    for dataname, details in results.items():
        for metricname, val in details.items():
            if metricname == "alias" or val in ["N/A", " "]:
                continue

            # Handle nested dict case: {'acc,none': 551} -> 551
            if isinstance(val, dict):
                # Extract the first value from the dict (assuming it's the metric)
                if len(val) > 0:
                    val = list(val.values())[0]
                else:
                    continue  # skip empty dicts

            # Skip non-numeric values (strings like task names)
            if not isinstance(val, (int, float)):
                print(f"Skipping non-numeric: {dataname}.{metricname} = {val} ({type(val)})")
                continue

            # Convert int to float for consistency
            val = float(val)

            metricname = metricname.split(",")[0]
            all_metrics.add(metricname)
            log[dataname][metricname] = val

    # Fix some aggregations.
    cultural_bench_parts = ["cultural_bench", "cultural_bench_easy", "cultural_bench_hard"]
    aggs = {
        "cultural_bench": (90, lambda taskname: taskname.startswith("cultural_bench") and taskname not in cultural_bench_parts),
        "cultural_bench_easy": (45, lambda taskname: taskname.startswith("cultural_bench_easy") and taskname not in cultural_bench_parts),
        "cultural_bench_hard": (45, lambda taskname: taskname.startswith("cultural_bench_hard") and taskname not in cultural_bench_parts),
        "m_hellaswag": (30, lambda taskname: taskname.startswith("hellaswag_")),
        "arc": (2, lambda taskname: taskname in ["arc_easy", "arc_challenge"]),
        "m_arc": (31, lambda taskname: taskname.startswith("arc_") and taskname not in ["arc_easy", "arc_challenge"]),
        "global_mmlu": (15, lambda taskname: re.match("^global_mmlu_[a-z]{2}$", taskname)),
        "include_base_44": (44, lambda taskname: re.match("^include_base_44_[a-z ]+$", taskname)),
        "xcopa": (11, lambda taskname: taskname.startswith("xcopa_")),
        "xnli": (15, lambda taskname: taskname.startswith("xnli_")),
        "xwinograd": (6, lambda taskname: taskname.startswith("xwinograd_")),
        "switzerland_qa": (5, lambda taskname: re.match("^switzerland_qa_[a-z]{2}$", taskname)),
        "blend": (16, lambda taskname: taskname.startswith("blend_")),
        "include_base_45": (45, lambda taskname: taskname.startswith("include_base_new_45_")),
    }
    for part, (target, filter_fn) in aggs.items():
        tasks_to_agg = list(filter(filter_fn, results))
        if len(tasks_to_agg) == target:
            agg(log, part, tasks_to_agg)
            if part == "cultural_bench":  # Legacy.
                agg(log, part, tasks_to_agg, macro=False, no_micro_suffix=True)
        else:
            print(f"WARNING! Couldn't fix {part} because all its parts haven't been evaluated (have {len(tasks_to_agg)}; expect {target})")

    # Fix cultural bench microaggregation.
    #for prefix in cultural_bench_parts:
    #    tasks_to_agg = [taskname for taskname in results
    #                    if taskname.startswith(prefix) and taskname not in cultural_bench_parts]
    #    target = 2*45 if prefix == "cultural_bench" else 45
    #    if len(tasks_to_agg) == target:
    #        agg(log, prefix, tasks_to_agg)
    #    else:
    #        print("WARNING! Couldn't fix", prefix, "because it hasn't been evaluated", tasks_to_agg)

    # Now that we have all the "leaf task groups" we can do four aggregations:
    # Let's start with the {language_group} agg.
    for lang_group_name, langs in tasks_cfg["language_groups"].items():
        tasks_to_agg = [task.name for task in all_tasks
                        if task.language.pt3 in langs]
        agg(log, f"All Tasks/{lang_group_name}", list(tasks_to_agg), micro=False, metrics=["acc"])

    # Aggregate start with the {dimension} agg.
    all_dims = sorted({task.dimension for task in all_tasks})
    for dim in all_dims:
        tasks_to_agg = [task.name for task in all_tasks
                        if task.dimension == dim]
        agg(log, f"All Dimensions/{pretty_dim(dim)}", list(tasks_to_agg), micro=False, metrics=["acc"])

    # Aggregate {dimension}.{language_group}.macro.
    for lang_group_name, langs in tasks_cfg["language_groups"].items():
        for dim in all_dims:
            tasks_to_agg = [task.name for task in all_tasks
                            if task.language.pt3 in langs and task.dimension == dim]
            agg(log, f"{pretty_dim(dim)}/{lang_group_name}", list(tasks_to_agg), micro=False, metrics=["acc"])


    # Finally, prepare wandb format.
    wandb_log = {}
    for dataname, details in log.items():
        for metric, value in details.items():
            wandb_log[f"{dataname}/{metric}"] = value
    return wandb_log


def get_history(name: str) -> Dict[int, Dict[str, float]]:
    api = wandb.Api()
    try:
        run = api.run(f"{api.default_entity}/{os.environ['WANDB_PROJECT']}/{name}")
    except wandb.errors.errors.CommError:  # Run not found.
        return {}
    history = collections.defaultdict(dict)
    for row in run.scan_history():
        row = {key: value for key, value in row.items()
               if not key.startswith("_") and key != "eval_table"}
        history[row["ConsumedTokens"]].update(row)
    return history


def repair(all_tasks: list[Task]) -> list[Task]:
    repaired = []
    for task in all_tasks:
        if task.name == "ai2_arc":
            repaired += [Task("arc_easy", (), 0, iso639.Lang("en"), task.dimension),
                         Task("arc_challenge", (), 0, iso639.Lang("en"), task.dimension)]
        else:
            repaired.append(task)
    return repaired


def main(logs_root: Path, names: list[str], it: Optional[int], cfg: Path):

    all_tasks = get_all_tasks(all_tasks_json=cfg/"all_tasks.json")
    all_tasks = repair(all_tasks)
    with open(cfg/"tasks.json") as f:
        tasks_cfg = json.load(f)
    all_languages = {task.language.pt3 for task in all_tasks}

    for lang_group in tasks_cfg["language_groups"].values():
        for lang in lang_group:
            assert lang in all_languages

    tasks_cfg["language_groups"]["global"] = list(all_languages)

    # Grab each possible log and update wandb run.
    # First, iterate model names.
    latest_logs = {}
    for p1 in filter(lambda p: names == [] or p.name in names, logs_root.iterdir()):
        print("Updating path", p1)
        history = get_history(p1.name)  # Get already pushed information.
        with wandb.init(id=p1.name, name=p1.name) as run:
            run.define_metric("ConsumedTokens")
            run.define_metric("*", step_metric="ConsumedTokens")
            # Now iterate all iterations for this name.
            for p2 in p1.iterdir():
                current_it = int(re.match("^iter_([0-9]+)$", p2.name).group(1))
                # Skip if specified --it doesn't match currently iterated it.
                if it is not None and it != current_it:
                    continue
                print("Updating iteration", current_it)

                with open(p2/"consumed_tokens.txt") as f:
                    consumed_tokens = int("".join(f).strip())

                # Get all results.json harness logs.
                results = []
                for path in sorted(p2.glob("harness/eval_*/*/results*.json")):
                    with open(path) as f:
                        results.append(json.load(f))

                if len(results) > 0:
                    log = get_log(results, tasks_cfg, all_tasks)
                    log.update({"ConsumedTokens": consumed_tokens, "OptStep": current_it})

                    # Update log if needed.
                    if consumed_tokens in history:
                        for key in set(history[consumed_tokens]) - set(log):
                            log[key] = INVALID_NUM

                        if log == history[consumed_tokens]:
                            print("Exact log already matches wandb! Ignoring entry to avoid pushing duplicates")
                        else:
                            print("Important! wandb log at current iteration already found, but differs. Updating")
                            run.log(log)
                    else:
                        run.log(log)
                        print("Logging")

                    # Update all_logs so we can build the table after this big loop.
                    if p1.name not in latest_logs or latest_logs[p1.name]["ConsumedTokens"] < consumed_tokens:
                        latest_logs[p1.name] = log
                else:
                    print("No logs found!")
                print()

    # Build and push the table.
    # We need `it` to be None to ensure that the logs on `latest_logs` actually
    # belong to the latest known iteration.
    show_in_table = tasks_cfg["show_in_table"]
    if it is None:
        for name, log in filter(lambda t: set(show_in_table) <= set(t[1]),
                                latest_logs.items()):
            print("Updating table for model", name)
            sublog = {"Model": name}
            sublog.update({task: log[task] for task in ["ConsumedTokens"] + show_in_table})
            df = pd.DataFrame([sublog])
            with wandb.init(id=name, name=name) as run:
                run.log({"eval_table": wandb.Table(dataframe=df), "ConsumedTokens": log["ConsumedTokens"]})

    # Update text description.
    print("Goodbye")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("logs_root", type=Path)
    parser.add_argument("--names", nargs="*", default=[])
    parser.add_argument("--it", type=int)
    parser.add_argument("--cfg", type=Path, default=Path("configs"))
    args = parser.parse_args()
    main(**vars(args))
