import argparse
from typing import Optional

from evals.tasks import TaskKind, get_all_tasks


def main(kind: Optional[TaskKind]):
    tasks = get_all_tasks()
    if kind is not None:
        tasks = list(filter(lambda task: kind in task.kind, tasks))
    print(",".join(task.name for task in tasks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", type=TaskKind)
    main(**vars(parser.parse_args()))
