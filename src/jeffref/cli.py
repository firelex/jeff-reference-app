"""The Jeff reference app's command line.

    jeffref fetch --base --tasks all --big     download Jeff's base model, every adapter with its test sets, and the 27B
    jeffref bench-27b --task guard             the 27B on the task's fixed test and calibration samples (resumable)
    jeffref bench-jeff --task guard            Jeff alone and Jeff + adapter on the same rows
    jeffref cascade                            the report (results/cascade.md and .json) from the recorded results

Only one model is in memory at a time: each bench command loads one model, holds a lock while it runs, and frees the
model when it exits. Run the bench commands one after the other, never side by side."""

import argparse
from collections.abc import Sequence

from jeffref.tasks import TASKS


def task_list(value: str) -> list[str]:
    if value == "all":
        return list(TASKS)
    names = [name for name in value.split(",") if name]
    unknown = [name for name in names if name not in TASKS]
    if unknown or not names:
        raise argparse.ArgumentTypeError(f"unknown task(s) {unknown or value!r}; known: {', '.join(TASKS)} (or all)")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jeffref", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download models and test sets from Hugging Face")
    fetch.add_argument("--base", action="store_true", help="Jeff's base model (mstrasser/Jeff-Qwen3.5-0.8B, v1.2)")
    fetch.add_argument("--tasks", type=task_list, default=[],
                       help="comma-separated tasks whose adapter and test sets to download, or all")
    fetch.add_argument("--big", action="store_true", help="Qwen3.8-27B at 8 bits (about 29 GB)")
    big = sub.add_parser("bench-27b", help="the 27B on one task's fixed samples")
    big.add_argument("--task", required=True, choices=list(TASKS))
    big.add_argument("--limit", type=int, help="answer at most this many new rows per split (for a smoke test)")
    jeff = sub.add_parser("bench-jeff", help="Jeff and Jeff + adapter on one task's fixed samples")
    jeff.add_argument("--task", required=True, choices=list(TASKS))
    sub.add_parser("cascade", help="the cascade report from the recorded results")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fetch":
        if not (args.base or args.tasks or args.big):
            parser.error("fetch needs at least one of --base, --tasks, --big")
        from jeffref.download import fetch_adapter, fetch_base, fetch_big
        from jeffref.tasks import get_task

        if args.base:
            fetch_base()
        for name in args.tasks:
            fetch_adapter(get_task(name))
        if args.big:
            fetch_big()
    elif args.command == "bench-27b":
        from jeffref.bench import run, summary
        from jeffref.lock import one_model

        with one_model(f"bench-27b --task {args.task}"):
            run(args.task, args.limit)
        summary()
    elif args.command == "bench-jeff":
        from jeffref.jeff_bench import run as run_jeff
        from jeffref.lock import one_model

        with one_model(f"bench-jeff --task {args.task}"):
            run_jeff(args.task)
    else:
        from jeffref.cascade import main as cascade

        cascade()


if __name__ == "__main__":
    main()
