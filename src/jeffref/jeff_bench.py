"""Score Jeff v1.2 (alone and with the task's adapter) with the MLX backend on this Mac, on exactly the rows the 27B
answers (the same fixed sample).

    jeffref bench-jeff --task emotion        (the task's own sample sizes, jeffref.tasks.TASKS)

One task per process: Jeff and that one adapter are loaded, scored, and freed when the process exits. MLX's buffer
cache is capped as for the 27B.
Test rows are scored by Jeff alone and by Jeff + adapter; the calibration rows by Jeff + adapter only, to choose the
confidence threshold of the cascade (jeffref.cascade) on rows the test figures do not use.
Writes results/jeff-mlx/<task>.json (metrics and time per decision) and <task>-predictions.jsonl (one line per
model, split and row)."""

import json
import time
from typing import Any

from jeff.evaluate import make_prediction

from jeffref.scoring import latency_summary, summarize
from jeffref.tasks import (JEFF_CHECKPOINT, JEFF_REPO, JEFF_REVISION, ROOT, SEED, adapter_path, check_layout, get_task,
                           recorded_sha256, task_sample)

OUT = ROOT / "results" / "jeff-mlx"


def run(name: str) -> None:
    import mlx.core as mx
    from jeff.mlx_backend import MlxDecisionModel

    from jeffref.big import MLX_CACHE_LIMIT

    mx.set_cache_limit(MLX_CACHE_LIMIT)

    task = get_task(name)
    check_layout(task)
    rows = task_sample(task, "test")  # checks the downloads against results/samples before the model loads
    calibration = task_sample(task, "calibration")
    if not (JEFF_CHECKPOINT / "decision_config.json").exists():
        raise FileNotFoundError(f"Jeff's base model is not in {JEFF_CHECKPOINT}; download it first: jeffref fetch --base")
    started = time.perf_counter()
    model = MlxDecisionModel(JEFF_CHECKPOINT, {task.name: adapter_path(task)})
    load_s = time.perf_counter() - started
    OUT.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"rows": len(rows), "seed": SEED, "load_s": load_s, "test_sha256": recorded_sha256(task),
                              "calibration_rows": len(calibration), "base_repo": JEFF_REPO,
                              "adapter_repo": task.repo, "revision": JEFF_REVISION}
    with (OUT / f"{task.name}-predictions.jsonl").open("w") as out:
        for label, adapter, split, split_rows in (("base", None, "test", rows), ("adapter", task.name, "test", rows),
                                                  ("adapter", task.name, "calibration", calibration)):
            model.use(adapter)
            predictions, seconds = [], []
            for row in split_rows:
                begun = time.perf_counter()
                [(raw, tokens)] = model.decide([row])
                total = sum(raw)  # MLX's float32 softmax; renormalize in float64 for jeff.evaluate's 1e-8 check
                if abs(total - 1) > 1e-4:
                    raise ValueError(f"{row['id']}: Jeff's probabilities sum to {total}")
                probabilities = [p / total for p in raw]
                seconds.append(time.perf_counter() - begun)
                prediction = make_prediction(row, probabilities, temperature=model.temperature)
                predictions.append(prediction)
                out.write(json.dumps({"model": label, "split": split, "id": row["id"], "probabilities": probabilities,
                                      "correct": prediction["correct"], "seconds": seconds[-1],
                                      "input_tokens": tokens}) + "\n")
            key = label if split == "test" else f"{label}_{split}"
            result[key] = {"metrics": summarize(predictions), "latency": latency_summary(seconds),
                           "temperature": model.temperature}
            print(f"{task.name:16s} {key:20s} acc {result[key]['metrics']['accuracy']:.3f} "
                  f"ECE {result[key]['metrics']['ece']:.3f}  median {result[key]['latency']['median_ms']:.0f} ms "
                  f"p95 {result[key]['latency']['p95_ms']:.0f} ms", flush=True)
    (OUT / f"{task.name}.json").write_text(json.dumps(result, indent=1) + "\n")
