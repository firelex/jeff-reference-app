"""The decision tasks: which Jeff adapter answers each one, where its held-out test set lives, and the fixed row sample.

Every adapter repository on Hugging Face (mstrasser/Jeff-Qwen3.5-0.8B-<task>, revision v1.2) holds the adapter files
plus the task's held-out test.jsonl and calibration.jsonl. `jeffref fetch` downloads each one into models/adapters/<task>.

results/samples/<task>.json records, for both splits, the SHA-256 of the file the published results were measured on
and the sampled rows' question, options and gold label (not their inputs). The cascade report is computed from these and
the recorded answers alone, so it needs no download. The benchmark commands check that a downloaded test set is
byte-identical to the one recorded there and that its sample has the same row ids."""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jeff.types import Example

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
JEFF_REPO = "mstrasser/Jeff-Qwen3.5-0.8B"
JEFF_REVISION = "v1.2"
JEFF_CHECKPOINT = MODELS / "jeff-v1.2-0.8b"
ADAPTERS = MODELS / "adapters"
SAMPLES = ROOT / "results" / "samples"
SEED = 20260930
SPLITS = ("test", "calibration")  # test: scored; calibration: fits the 27B's temperature and picks the cascade threshold


@dataclass(frozen=True)
class Task:
    name: str
    agent_step: str | None  # the inbox agent step this task stands for, or None for the extra tasks
    rows: int  # test rows scored (a fixed sample; the whole set if smaller)
    calibration_rows: int  # calibration rows (a fixed sample; the whole set if smaller)

    @property
    def repo(self) -> str:
        return f"{JEFF_REPO}-{self.name}"

    def local(self, split: str = "test") -> Path:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; use one of {SPLITS}")
        return ADAPTERS / self.name / f"{split}.jsonl"


# The five inbox-agent decisions first, then the extra tasks. Most tasks use 300 test rows and 150 calibration rows;
# emotion and legal-clauses use 500 and 200.
TASKS = {task.name: task for task in [
    Task("guard", agent_step="1. guard", rows=300, calibration_rows=150),
    Task("triage", agent_step="2. triage", rows=300, calibration_rows=150),
    Task("support-intents", agent_step="3. support-intents", rows=300, calibration_rows=150),
    Task("tools", agent_step="4. tools", rows=300, calibration_rows=150),
    Task("ground", agent_step="5. ground", rows=300, calibration_rows=150),
    Task("nav", agent_step=None, rows=300, calibration_rows=150),
    Task("emotion", agent_step=None, rows=500, calibration_rows=200),
    Task("spam", agent_step=None, rows=300, calibration_rows=150),
    Task("legal-clauses", agent_step=None, rows=500, calibration_rows=200),
]}
AGENT_TASKS = [name for name, task in TASKS.items() if task.agent_step]


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise ValueError(f"Unknown task {name!r}; known tasks: {', '.join(TASKS)}")
    return TASKS[name]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[Example]:
    rows = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
    if not rows:
        raise ValueError(f"{path} has no rows")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} has duplicate row ids")
    return rows


def sample_of(rows: list[Example], size: int, name: str, split: str, seed: int = SEED) -> list[Example]:
    """A fixed random sample of a split's rows: `size` rows (all rows if the set is smaller), drawn with `seed`, in file
    order. The same call always returns the same rows, so the 27B and Jeff are scored on identical rows."""
    if size >= len(rows):
        return rows
    chosen = sorted(random.Random(f"{seed}:{name}:{split}").sample(range(len(rows)), size))
    return [rows[index] for index in chosen]


def sample_rows(task: Task, size: int, split: str = "test", seed: int = SEED) -> list[Example]:
    """The fixed sample of a downloaded split (see sample_of)."""
    path = task.local(split)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; download it first (jeffref fetch --tasks {task.name})")
    return sample_of(read_rows(path), size, task.name, split, seed)


def sample_size(task: Task, split: str) -> int:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; use one of {SPLITS}")
    return task.rows if split == "test" else task.calibration_rows


def task_sample(task: Task, split: str = "test") -> list[Example]:
    """The task's sample of a downloaded split, with inputs, checked against results/samples/<task>.json: the file must
    be the one the published results were measured on, and the sampled row ids must be the recorded ones."""
    check_download(task, split)
    rows = sample_rows(task, sample_size(task, split), split)
    recorded = [row["id"] for row in recorded_sample(task, split)]
    if [row["id"] for row in rows] != recorded:
        raise RuntimeError(f"{task.name}/{split}: the sample of {task.local(split)} differs from {samples_path(task)}")
    return rows


# --- The recorded samples (results/samples/<task>.json) ---------------------------------------------------------------

def samples_path(task: Task) -> Path:
    return SAMPLES / f"{task.name}.json"


def key_row(row: Example) -> dict[str, Any]:
    """What scoring needs of a row (jeff.evaluate.make_prediction): its id, suite, family, question type and option keys
    (in order; their descriptions dropped), gold label and target, and the dataset name and human answer distribution
    from its source. Not its input (`state`) or the question's wording."""
    out: dict[str, Any] = {key: row[key] for key in ("id", "suite", "family", "target")}
    question = row["question"]
    out["question"] = {"type": question["type"]}
    if question["type"] == "choice":  # options are the criteria's keys
        out["question"]["criteria"] = {key: None for key in question["criteria"]}
    elif question["type"] == "score":  # options are the levels 0..n-1
        out["question"]["criteria"] = [None] * len(question["criteria"])
    elif question["type"] != "noul":  # a noul (yes/no) question's options are always [False, True]
        raise ValueError(f"{row['id']}: unknown question type {question['type']!r}")
    if "label" in row:
        out["label"] = row["label"]
    out["source"] = {key: row["source"][key] for key in ("dataset", "human_distribution") if key in row["source"]}
    return out


def build_samples(task: Task) -> dict[str, Any]:
    """The samples record of the downloaded test and calibration files: each file's SHA-256 and row count, and the
    sampled rows reduced to what scoring needs (key_row)."""
    record: dict[str, Any] = {"task": task.name, "seed": SEED}
    for split in SPLITS:
        path = task.local(split)
        rows = read_rows(path)
        sample = sample_of(rows, sample_size(task, split), task.name, split)
        record[split] = {"sha256": sha256(path), "rows_in_file": len(rows), "rows": [key_row(row) for row in sample]}
    return record


def read_samples(task: Task) -> dict[str, Any]:
    path = samples_path(task)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; it records the rows the published results were measured on")
    record = json.loads(path.read_text())
    if record["task"] != task.name or record["seed"] != SEED:
        raise ValueError(f"{path} is for task {record['task']!r} with seed {record['seed']}, not {task.name!r} with {SEED}")
    return record


def recorded_sample(task: Task, split: str = "test") -> list[Example]:
    """The sampled rows of a split as recorded in results/samples/<task>.json (question, options and label only)."""
    record = read_samples(task)[split]
    rows = record["rows"]
    if len(rows) != min(sample_size(task, split), record["rows_in_file"]):
        raise ValueError(f"{samples_path(task)} holds {len(rows)} {split} rows, not the task's sample size")
    return rows


def recorded_sha256(task: Task, split: str = "test") -> str:
    return str(read_samples(task)[split]["sha256"])


def accepted_sha256s(task: Task, split: str = "test") -> set[str]:
    """Results may come from the published file (sha256) or, for our recorded results, from the file it was measured
    on before an internal machine name in the provenance fields was replaced (measured_sha256); the sampled rows of
    both are identical."""
    record = read_samples(task)[split]
    return {str(record["sha256"]), str(record.get("measured_sha256", record["sha256"]))}


def check_download(task: Task, split: str) -> None:
    """Fail unless the downloaded split is byte-identical to the published file the results' samples were checked
    against (results/samples/<task>.json `sha256`). For guard and ground that file differs from the one measured on
    (`measured_sha256`) only in its provenance fields, where an internal machine name reads 'local'; every sampled
    row is identical, which was checked row by row before release."""
    path = task.local(split)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; download it first (jeffref fetch --tasks {task.name})")
    actual, expected = sha256(path), recorded_sha256(task, split)
    if actual != expected:
        raise RuntimeError(f"{path} has SHA-256 {actual}, but the published results were measured on {expected} "
                           f"({samples_path(task)}); this is not the same {task.name} {split} set")


# --- Adapters --------------------------------------------------------------------------------------------------------

ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json", "decision_config.json", "readout.safetensors")


def adapter_path(task: Task) -> Path:
    path = ADAPTERS / task.name
    missing = [name for name in ADAPTER_FILES if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"The {task.name} adapter in {path} lacks {missing}; download it with "
                                f"jeffref fetch --tasks {task.name}")
    return path


# Every v1.2 adapter (and the v1.2 base) is trained with the state-first prompt layout (checked in each adapter's
# decision_config.json by check_layout). The 27B is prompted with the same layout, so it sees exactly Jeff's prompt.
LAYOUT = "state-first"


def check_layout(task: Task) -> None:
    config = json.loads((adapter_path(task) / "decision_config.json").read_text())
    layout = config.get("prompt_layout", "state-first")  # Jeff's own rule: a checkpoint without the setting is state-first
    if layout != LAYOUT:
        raise ValueError(f"The {task.name} adapter uses prompt layout {layout!r}; this benchmark assumes {LAYOUT!r}")
