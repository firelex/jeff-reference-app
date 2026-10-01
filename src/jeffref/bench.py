"""Decision accuracy, calibration and time of Qwen3.8-27B (MLX, this Mac) on the adapters' held-out test sets.

    jeffref bench-27b --task guard          answer the task's fixed test and calibration samples (resumes where it stopped)

Records go to results/big-8bit/<task>-<split>.jsonl, one JSON line per row, written as each row finishes. After the run,
results/big-8bit/summary.json is rewritten with the metrics of every task whose rows are all answered."""

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jeffref.scoring import big_prediction, fitted_temperature, latency_summary, read_records, summarize
from jeffref.tasks import LAYOUT, ROOT, SEED, SPLITS, TASKS, get_task, read_samples, recorded_sample, task_sample

RESULTS = ROOT / "results"
QUANT = "8bit"
CHUNK = 20  # rows between memory checks
MEMORY_LIMIT_GB = 50  # stop if MLX holds more than this (the 27B at 8-bit is about 30 GB plus a 2 GB buffer cache)


def records_path(quant: str, task: str, split: str) -> Path:
    return RESULTS / f"big-{quant}" / f"{task}-{split}.jsonl"


def ids_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()


def run(name: str, limit: int | None) -> None:
    """Answer the task's test sample, then its calibration sample, with the 27B. Only rows not answered yet are asked."""
    from jeffref.big import BigModel

    task = get_task(name)
    samples = {split: task_sample(task, split) for split in SPLITS}  # checks the downloads before the model loads
    model = BigModel()
    print(f"loaded {model.repo} ({model.revision}) in {model.load_s:.0f} s", flush=True)
    mx = model.mx
    for split, sample in samples.items():
        path = records_path(QUANT, name, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        done = read_records(path)
        stray = set(done) - {row["id"] for row in sample}
        if stray:
            raise ValueError(f"{path} holds {len(stray)} records outside this sample")
        todo = [row for row in sample if row["id"] not in done]
        if limit is not None:
            todo = todo[:limit]
        print(f"{name}/{split}: {len(sample)} rows in sample, {len(done)} done, {len(todo)} to answer", flush=True)
        started = time.perf_counter()
        with path.open("a") as out:
            for count, row in enumerate(todo, 1):
                decision = model.decide(row, LAYOUT)
                record = {
                    "id": row["id"], "codes": decision.codes,
                    "option_log_probs": decision.option_log_probs,
                    "option_mass": decision.option_mass, "reply": decision.reply, "answer_code": decision.answer_code,
                    "input_tokens": decision.input_tokens, "output_tokens": decision.output_tokens,
                    "prefill_s": decision.prefill_s, "total_s": decision.total_s,
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                if count % CHUNK == 0 or count == len(todo):
                    rate = count / (time.perf_counter() - started)
                    held = (mx.get_active_memory() + mx.get_cache_memory()) / 2**30
                    print(f"  {name}/{split} {count}/{len(todo)}  {rate:.2f} rows/s  last {decision.total_s:.2f} s "
                          f"({decision.input_tokens} tokens)  MLX holds {held:.1f} GB "
                          f"(peak {mx.get_peak_memory() / 2**30:.1f} GB)", flush=True)
                    if held > MEMORY_LIMIT_GB:
                        raise RuntimeError(f"MLX holds {held:.1f} GB, over the {MEMORY_LIMIT_GB} GB limit; stopping "
                                           "before the Mac runs out of memory")


def summary() -> dict[str, Any]:
    """Metrics of every task whose sampled test rows the 27B has all answered (results/big-8bit/summary.json). Scored on
    the recorded samples (results/samples), so no download is needed."""
    out: dict[str, Any] = {}
    for name, task in TASKS.items():
        path = records_path(QUANT, name, "test")
        if not path.exists():
            continue
        samples = read_samples(task)
        sample = recorded_sample(task, "test")
        records = read_records(path)
        missing = [row["id"] for row in sample if row["id"] not in records]
        if missing:
            print(f"{name}: {len(missing)} of {len(sample)} sampled rows not answered yet; not in the summary",
                  file=sys.stderr)
            continue
        predictions = [big_prediction(row, records[row["id"]]) for row in sample]
        seconds = [records[row["id"]]["total_s"] for row in sample]
        tokens = [records[row["id"]]["input_tokens"] for row in sample]
        result: dict[str, Any] = {
            "test_sha256": samples["test"]["sha256"],
            "rows": len(sample), "test_rows_total": samples["test"]["rows_in_file"], "seed": SEED,
            "row_ids_sha256": ids_sha256(sample), "raw": summarize(predictions), "latency": latency_summary(seconds),
            "input_tokens_median": sorted(tokens)[len(tokens) // 2],
            "option_mass_median": sorted(records[r["id"]]["option_mass"] for r in sample)[len(sample) // 2],
        }
        calibration_path = records_path(QUANT, name, "calibration")
        if calibration_path.exists():
            calibration = recorded_sample(task, "calibration")
            calibration_records = read_records(calibration_path)
            if all(row["id"] in calibration_records for row in calibration):
                temperature = fitted_temperature(calibration, [calibration_records[row["id"]] for row in calibration])
                result["temperature"] = temperature
                result["calibration_rows"] = len(calibration)
                result["calibrated"] = summarize([big_prediction(row, records[row["id"]], temperature) for row in sample])
        out[name] = result
    from jeffref.big import BIG_REPO, BIG_REVISION

    meta = {"model": {"repo": BIG_REPO, "revision": BIG_REVISION, "quantization": QUANT, "runtime": "mlx-lm",
                      "thinking": "off"},
            "machine": machine(), "written": datetime.now(timezone.utc).isoformat(),
            "rows_per_task": {n: t.rows for n, t in TASKS.items()},
            "calibration_rows_per_task": {n: t.calibration_rows for n, t in TASKS.items()}, "seed": SEED, "layout": LAYOUT}
    path = RESULTS / f"big-{QUANT}" / "summary.json"
    path.write_text(json.dumps({"meta": meta, "tasks": out}, indent=1) + "\n")
    print(f"wrote {path}")
    for name, result in out.items():
        cal = result.get("calibrated")
        print(f"{name:16s} n={result['rows']:4d} acc {result['raw']['accuracy']:.3f} ECE raw {result['raw']['ece']:.3f}"
              + (f" / fitted T={result['temperature']:.2f} {cal['ece']:.3f}" if cal else "")
              + f"  invalid {result['raw']['invalid']}  median {result['latency']['median_ms']:.0f} ms"
              f"  p95 {result['latency']['p95_ms']:.0f} ms")
    return out


def machine() -> dict[str, str]:
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True)
    memory = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
    return {"chip": chip.stdout.strip(), "memory_gb": f"{int(memory.stdout) / 2**30:.0f}", "os": platform.platform()}
