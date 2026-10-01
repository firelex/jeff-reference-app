"""The cascade against the 27B alone, per task, each task with its own confidence threshold.

Two routes over the same fixed sample of rows:
- 27B alone: every query is answered by the 27B.
- Cascade: every query is answered by Jeff + the task's adapter first; when Jeff's confidence (the calibrated probability
  of its chosen option) is below X, the query is also sent to the 27B and the 27B's answer is used. The query's time is
  then Jeff's time plus the 27B's.

Both are replayed from the recorded answers and times (results/big-8bit/, results/jeff-mlx/) and the recorded samples
(results/samples/: each sampled row's options and gold label); nothing is re-run or estimated, and nothing is
downloaded. Each task's X is chosen on its own calibration rows: the lowest X (fewest queries sent to the 27B, so the
fastest) whose accuracy beats the 27B alone by at least MARGIN; then it is reported on the test rows. The headline is the
mean over the five agent decisions; the other tasks are reported beside it.

    uv run jeffref cascade        (writes results/cascade.json and results/cascade.md)

A task is included only when the 27B has answered all its test and calibration rows and Jeff + adapter has been scored
on the same test file, calibration rows included."""

import json
import math
from pathlib import Path
from typing import Any

from jeffref.bench import records_path
from jeffref.scoring import big_prediction, read_records
from jeffref.tasks import AGENT_TASKS, ROOT, TASKS, Task, accepted_sha256s, recorded_sample

RESULTS = ROOT / "results"
QUANT = "8bit"
THRESHOLDS = [round(step / 100, 2) for step in range(0, 101)]
EXCLUDED = {"emotion": ("picking the single strongest of 27 emotions (or neutral) in short Reddit comments is hard even for "
                         "people, and the human labels often disagree")}  # reported, but not averaged into the headline
MARGIN = 0.01  # a task's threshold must beat the 27B alone by at least this much accuracy on its calibration rows


def rows_for(task: Task, split: str, results: Path = RESULTS) -> list[dict[str, Any]] | None:
    """Per row: Jeff's confidence, correctness and seconds, and the 27B's correctness and seconds. None when either
    side is incomplete (for example a task still being measured)."""
    sample = recorded_sample(task, split)
    big_path = results / f"big-{QUANT}" / records_path(QUANT, task.name, split).name
    big = read_records(big_path)
    if len(big) != len(sample):
        return None
    jeff_path = results / "jeff-mlx" / f"{task.name}-predictions.jsonl"
    summary_path = results / "jeff-mlx" / f"{task.name}.json"
    if not jeff_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    if summary["test_sha256"] not in accepted_sha256s(task) or "adapter_calibration" not in summary:
        return None
    jeff = {}
    for line in jeff_path.read_text().splitlines():
        record = json.loads(line)
        if record["model"] == "adapter" and record["split"] == split:
            jeff[record["id"]] = record
    if set(jeff) != {row["id"] for row in sample}:
        raise ValueError(f"{jeff_path} ({split}) does not cover exactly the sampled rows")
    out = []
    for row in sample:
        mine, theirs = jeff[row["id"]], big[row["id"]]
        out.append({"confidence": max(mine["probabilities"]), "jeff_correct": mine["correct"],
                    "jeff_s": mine["seconds"], "big_correct": big_prediction(row, theirs)["correct"],
                    "big_s": theirs["total_s"]})
    return out


def route(rows: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    correct = seconds = sent = 0.0
    for row in rows:
        seconds += row["jeff_s"]
        if row["confidence"] < threshold:
            sent += 1
            seconds += row["big_s"]
            correct += row["big_correct"]
        else:
            correct += row["jeff_correct"]
    n = len(rows)
    return {"accuracy": correct / n, "mean_s": seconds / n, "to_27b": sent / n}


def big_alone(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows)
    return {"accuracy": sum(r["big_correct"] for r in rows) / n, "mean_s": sum(r["big_s"] for r in rows) / n}


def report(results: Path = RESULTS) -> tuple[dict[str, Any], str]:
    """The cascade report from the recorded results in `results`: (cascade.json content, cascade.md text)."""
    tasks: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, task in TASKS.items():
        splits = {split: rows_for(task, split, results) for split in ("test", "calibration")}
        if all(splits.values()):
            tasks[name] = splits  # type: ignore[assignment]
    if not tasks:
        raise SystemExit("No task has both the 27B and Jeff + adapter scored on test and calibration rows yet")

    # The headline and the threshold come from the five agent decisions only; the other tasks are reported beside
    # them, with the same threshold, but are not averaged in.
    absent = [name for name in AGENT_TASKS if name not in tasks]
    if absent:
        raise SystemExit(f"The headline needs all five agent tasks; not yet scored on both sides: {absent}")
    headline = {name: tasks[name] for name in AGENT_TASKS}

    def mean_accuracy(split: str, threshold: float) -> float:
        return sum(route(t[split], threshold)["accuracy"] for t in headline.values()) / len(headline)

    chosen = max(THRESHOLDS, key=lambda x: (round(mean_accuracy("calibration", x), 12), -x))
    def own_threshold(name: str, t: dict[str, list[dict[str, Any]]]) -> float:
        """Every task gets its own threshold, chosen on its own calibration rows: the lowest threshold (fewest queries
        sent to the 27B, so the fastest) whose accuracy beats the 27B alone by at least MARGIN. One shared threshold does
        not fit: emotion (28 options) has lower confidences, and ground is the task where the 27B is strong enough to
        be worth asking."""
        big = big_alone(t["calibration"])["accuracy"]
        for x in THRESHOLDS:
            if route(t["calibration"], x)["accuracy"] >= big + MARGIN:
                return x
        raise SystemExit(f"{name}: no threshold beats the 27B by {MARGIN:.0%} on the calibration rows")

    per_task = {}
    for name, t in tasks.items():
        threshold = own_threshold(name, t)
        base, cascade = big_alone(t["test"]), route(t["test"], threshold)
        per_task[name] = {"group": "agent" if name in AGENT_TASKS else "other", "threshold": threshold, "rows": len(t["test"]),
                          "big_alone": base, "cascade": cascade,
                          "speed_up": base["mean_s"] / cascade["mean_s"],
                          "curve": {str(x): route(t["test"], x) for x in THRESHOLDS}}
    def summary(names: list[str]) -> dict[str, Any]:
        rows = [per_task[name] for name in names]
        return {"tasks": names,
                "big_alone_accuracy": sum(p["big_alone"]["accuracy"] for p in rows) / len(rows),
                "cascade_accuracy": sum(p["cascade"]["accuracy"] for p in rows) / len(rows),
                "big_alone_mean_s": sum(p["big_alone"]["mean_s"] for p in rows) / len(rows),
                "cascade_mean_s": sum(p["cascade"]["mean_s"] for p in rows) / len(rows),
                "speed_up_geometric_mean": math.exp(sum(math.log(p["speed_up"]) for p in rows) / len(rows))}

    # The headline: every task measured on both sides except those in EXCLUDED (reported separately, with the reason
    # and the average including them). The inbox agent's five decisions are a second figure.
    headline_names = [name for name in per_task if name not in EXCLUDED]
    for name, p in per_task.items():
        p["in_headline"] = name not in EXCLUDED
    overall = summary(headline_names)
    inbox_agent = summary(list(AGENT_TASKS))
    all_tasks = summary(list(per_task))
    curve = []
    for x in THRESHOLDS:
        points = [route(t["test"], x) for t in headline.values()]
        ups = [big_alone(t["test"])["mean_s"] / p["mean_s"] for t, p in zip(headline.values(), points)]
        curve.append({"threshold": x, "calibration_accuracy": mean_accuracy("calibration", x),
                      "test_accuracy": sum(p["accuracy"] for p in points) / len(points),
                      "to_27b": sum(p["to_27b"] for p in points) / len(points),
                      "speed_up": math.exp(sum(math.log(u) for u in ups) / len(ups))})
    result = {"threshold": "per task", "shared_threshold": chosen,
              "chosen_on": "each task's own calibration rows: the lowest threshold (fastest) whose accuracy beats the 27B alone by at least 1 point",
              "overall": overall, "inbox_agent": inbox_agent, "all_tasks": all_tasks,
              "excluded_from_headline": {name: reason for name, reason in EXCLUDED.items() if name in per_task},
              "tasks": per_task, "curve": curve}

    header = ["| Task | rows | 27B alone: accuracy | 27B alone: mean time | Cascade: accuracy | Cascade: mean time "
              "| sent to 27B | faster |", "|---|---:|---:|---:|---:|---:|---:|---:|"]

    def row(name: str) -> str:
        p = per_task[name]
        b, c = p["big_alone"], p["cascade"]
        return (f"| {name} | {p['rows']} | {b['accuracy']:.1%} | {b['mean_s']:.2f} s | {c['accuracy']:.1%} "
                f"| {c['mean_s']:.2f} s | {c['to_27b']:.1%} | {p['speed_up']:.1f}x |")

    def mean_row(label: str, m: dict[str, Any]) -> str:
        return (f"| **{label}** | | {m['big_alone_accuracy']:.1%} | {m['big_alone_mean_s']:.2f} s | "
                f"{m['cascade_accuracy']:.1%} | {m['cascade_mean_s']:.2f} s | | {m['speed_up_geometric_mean']:.1f}x (geometric) | |")

    lines = ["Each task has its own threshold, chosen on its own calibration rows: the lowest (fastest) whose accuracy "
             f"beats the 27B alone by at least {MARGIN:.0%}. Test rows below.",
             "", f"## The headline: {len(headline_names)} adapters (every one measured, except "
             f"{', '.join(n for n in EXCLUDED if n in per_task) or 'none'})", "",
             header[0] + " threshold |", header[1] + "---:|"]
    lines += [row(name) + f" {per_task[name]['threshold']:.2f} |" for name in headline_names]
    lines.append(mean_row(f"mean of the {len(headline_names)}", overall))
    lines += ["", "## The inbox agent's five decisions (guard, triage, support-intents, tools, ground)", "",
              header[0] + " threshold |", header[1] + "---:|", mean_row("mean of the five", inbox_agent)]
    for name, reason in EXCLUDED.items():
        if name in per_task:
            p = per_task[name]
            lines += ["", f"## Left out of the headline: {name}", "", header[0] + " threshold |", header[1] + "---:|",
                      row(name) + f" {p['threshold']:.2f} |", "",
                      f"Left out because {reason}. Including it, the mean over all {len(per_task)} measured adapters is "
                      f"{all_tasks['cascade_accuracy']:.1%} for the cascade against {all_tasks['big_alone_accuracy']:.1%} "
                      f"for the 27B alone, so leaving it out makes the headline gain smaller, not larger."]
    lines += ["", f"For comparison, one shared threshold for the five agent tasks (best: {chosen:.2f}); mean; faster = "
              "geometric mean of per-task speed-ups:", "",
              "| X | calibration accuracy | test accuracy | sent to 27B | faster |", "|---:|---:|---:|---:|---:|"]
    for point in curve:
        if round(point["threshold"] * 100) % 5 == 0 or point["threshold"] == chosen:
            lines.append(f"| {point['threshold']:.2f} | {point['calibration_accuracy']:.1%} | "
                         f"{point['test_accuracy']:.1%} | {point['to_27b']:.1%} | {point['speed_up']:.1f}x |")
    return result, "\n".join(lines) + "\n"


def main(results: Path = RESULTS) -> None:
    result, text = report(results)
    (results / "cascade.json").write_text(json.dumps(result, indent=1) + "\n")
    (results / "cascade.md").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
