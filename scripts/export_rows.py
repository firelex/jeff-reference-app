"""Write results/per-row-v1.2.json: every test row of the Qwen3.8-27B comparison with all three models' answers.

For each adapter's test sample (300 rows; 500 for emotion and legal-clauses), one record per row: the correct label,
the options in order, and for Qwen3.8-27B (8-bit, thinking off), plain Jeff v1.2 and Jeff v1.2 + the task's adapter the
prediction, the probability of each option, whether it was right, and the seconds it took. Everything is read from the
recorded answers in results/big-8bit and results/jeff-mlx and scored with jeffref.scoring, the same code behind every
published number, so the accuracies recomputed from this file match the site exactly.

Usage: uv run python scripts/export_rows.py
"""

import json
from pathlib import Path

from jeffref.bench import records_path
from jeffref.scoring import big_prediction, fitted_temperature, read_records
from jeffref.tasks import TASKS, task_sample

RESULTS = Path(__file__).resolve().parent.parent / "results"
QUANT = "8bit"


def jeff_answers(task: str, split: str) -> dict[tuple[str, str], dict]:
    answers = {}
    for line in (RESULTS / "jeff-mlx" / f"{task}-predictions.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["split"] == split:
            answers[(record["model"], record["id"])] = record
    return answers


def main() -> None:
    tasks = {}
    rows = []
    for name, task in TASKS.items():
        sample = task_sample(task, "test")
        big = read_records(records_path(QUANT, name, "test"))
        calibration = task_sample(task, "calibration")
        temperature = fitted_temperature(calibration, [read_records(records_path(QUANT, name, "calibration"))[r["id"]]
                                                       for r in calibration])
        jeff = jeff_answers(name, "test")
        for row in sample:
            theirs = big_prediction(row, big[row["id"]])
            options = theirs["options"]
            record = {"task": name, "id": row["id"], "question_type": row["question"]["type"],
                      "options": options, "label": row["label"],
                      "qwen27b": {"prediction": theirs["prediction"], "probabilities": theirs["probabilities"],
                                  "correct": theirs["correct"], "seconds": big[row["id"]]["total_s"]}}
            for model, key in (("base", "jeff_base"), ("adapter", "jeff_adapter")):
                mine = jeff[(model, row["id"])]
                best = max(range(len(options)), key=lambda i: mine["probabilities"][i])
                record[key] = {"prediction": options[best], "probabilities": mine["probabilities"],
                               "correct": mine["correct"], "seconds": mine["seconds"]}
            rows.append(record)
        tasks[name] = {"test_rows": len(sample), "qwen27b_temperature": temperature}
    about = {
        "what": "Every test row of the Jeff v1.2 vs Qwen3.8-27B comparison, with all three models' answers.",
        "models": {"qwen27b": "Qwen3.8-27B, 8-bit MLX, thinking off, prompted with the options",
                   "jeff_base": "Jeff-Qwen3.5-0.8B v1.2 without an adapter",
                   "jeff_adapter": "Jeff-Qwen3.5-0.8B v1.2 with the task's LoRA adapter"},
        "fields": {"options": "the answer options in order; probabilities follow this order",
                   "label": "the correct option",
                   "probabilities": "Jeff: its calibrated probabilities. Qwen3.8-27B: its option log-probabilities "
                                    "normalised at temperature 1 (tasks.<name>.qwen27b_temperature is the temperature "
                                    "fitted on that task's calibration rows, used for its calibration figures)",
                   "correct": "prediction == label", "seconds": "wall-clock seconds for this decision on an M4 Max"},
        "source": "https://github.com/firelex/jeff-reference-app (scripts/export_rows.py)",
        "test_sets": "https://huggingface.co/mstrasser (each adapter's test set, matched by id)",
    }
    out = RESULTS / "per-row-v1.2.json"
    out.write_text(json.dumps({"about": about, "tasks": tasks, "rows": rows}, ensure_ascii=False) + "\n")
    print(f"wrote {out}: {len(rows)} rows, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
