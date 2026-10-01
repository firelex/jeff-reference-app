"""The fixed samples: the same seed and sizes always give the same rows, and the recorded samples (results/samples) are
the rows the published results were measured on."""

import json

import pytest

from jeffref import tasks
from jeffref.bench import ids_sha256, records_path
from jeffref.tasks import SEED, SPLITS, TASKS


def row(identifier: str) -> dict:
    return {"id": identifier, "suite": "demo", "family": "f1", "source": {"dataset": "demo", "generator": "x"},
            "label": "track", "target": "track", "state": {"message": "Where is my parcel?"},
            "question": {"type": "choice", "instructions": "What does the customer want?",
                         "criteria": {"track": "Track an order", "refund": "Get a refund", "other": None}}}


def write_split(tmp_path, monkeypatch, task: tasks.Task, split: str, count: int) -> None:
    monkeypatch.setattr(tasks, "ADAPTERS", tmp_path)
    path = task.local(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row(f"r{i:04d}")) + "\n" for i in range(count)))


def test_task_sizes_and_seed() -> None:
    assert SEED == 20260930
    for name, task in TASKS.items():
        big = name in ("emotion", "legal-clauses")
        assert (task.rows, task.calibration_rows) == ((500, 200) if big else (300, 150))
    assert tasks.AGENT_TASKS == ["guard", "triage", "support-intents", "tools", "ground"]


def test_sample_is_fixed_and_in_file_order(tmp_path, monkeypatch) -> None:
    task = tasks.Task("demo", agent_step=None, rows=10, calibration_rows=5)
    write_split(tmp_path, monkeypatch, task, "test", 100)
    first = [r["id"] for r in tasks.sample_rows(task, 10)]
    assert first == [r["id"] for r in tasks.sample_rows(task, 10)]
    assert first == sorted(first) and len(set(first)) == 10
    assert len(tasks.sample_rows(task, 1000)) == 100  # whole set when smaller than the sample size


def test_sample_depends_on_task_name_and_split() -> None:
    rows = [row(f"r{i:04d}") for i in range(2000)]
    ids = lambda name, split: [r["id"] for r in tasks.sample_of(rows, 300, name, split)]  # noqa: E731
    assert ids("guard", "test") != ids("triage", "test")
    assert ids("guard", "test") != ids("guard", "calibration")
    for size in (150, 300, 500):
        sample = [r["id"] for r in tasks.sample_of(rows, size, "guard", "test")]
        assert len(sample) == len(set(sample)) == size


def test_recorded_samples_drop_inputs_and_keep_what_scoring_needs(tmp_path, monkeypatch) -> None:
    task = tasks.Task("demo", agent_step=None, rows=10, calibration_rows=5)
    for split in SPLITS:
        write_split(tmp_path, monkeypatch, task, split, 50)
    record = tasks.build_samples(task)
    key = record["test"]["rows"][0]
    assert "state" not in key and key["source"] == {"dataset": "demo"}
    assert key["question"] == {"type": "choice", "criteria": {"track": None, "refund": None, "other": None}}
    assert [r["id"] for r in record["test"]["rows"]] == [r["id"] for r in tasks.sample_rows(task, 10)]


def test_download_check_rejects_a_different_file(tmp_path, monkeypatch) -> None:
    task = tasks.Task("demo", agent_step=None, rows=10, calibration_rows=5)
    for split in SPLITS:
        write_split(tmp_path, monkeypatch, task, split, 50)
    monkeypatch.setattr(tasks, "SAMPLES", tmp_path / "samples")
    (tmp_path / "samples").mkdir()
    tasks.samples_path(task).write_text(json.dumps(tasks.build_samples(task)))
    assert [r["id"] for r in tasks.task_sample(task)] == [r["id"] for r in tasks.recorded_sample(task)]
    task.local("test").write_text(task.local("test").read_text() + json.dumps(row("extra")) + "\n")
    with pytest.raises(RuntimeError, match="not the same demo test set"):
        tasks.task_sample(task)


@pytest.mark.parametrize("name", list(TASKS))
def test_recorded_samples_have_the_task_sizes(name) -> None:
    task = TASKS[name]
    for split in SPLITS:
        sample = tasks.recorded_sample(task, split)
        assert len(sample) == tasks.sample_size(task, split)
        assert len({r["id"] for r in sample}) == len(sample)


@pytest.mark.parametrize("name", [n for n in TASKS if records_path("8bit", n, "test").exists()])
def test_recorded_samples_are_the_rows_the_published_results_used(name) -> None:
    """The 27B's recorded answers cover exactly the recorded sample, in order, and the row-id checksum in its summary
    matches; Jeff's predictions cover the same rows."""
    task = TASKS[name]
    summary = json.loads(records_path("8bit", name, "test").parent.joinpath("summary.json").read_text())["tasks"]
    assert ids_sha256(tasks.recorded_sample(task, "test")) == summary[name]["row_ids_sha256"]
    assert summary[name]["test_sha256"] in tasks.accepted_sha256s(task)
    predictions = [json.loads(line) for line in
                   (tasks.ROOT / "results" / "jeff-mlx" / f"{name}-predictions.jsonl").read_text().splitlines()]
    for split in SPLITS:
        ids = [r["id"] for r in tasks.recorded_sample(task, split)]
        big = [json.loads(line)["id"] for line in records_path("8bit", name, split).read_text().splitlines()]
        assert big == ids
        assert [p["id"] for p in predictions if p["model"] == "adapter" and p["split"] == split] == ids


@pytest.mark.parametrize("name", list(TASKS))
def test_downloaded_test_sets_give_the_recorded_samples(name) -> None:
    """Runs only where `jeffref fetch --tasks <name>` has downloaded the task's test sets."""
    task = TASKS[name]
    if not all(task.local(split).exists() for split in SPLITS):
        pytest.skip(f"{name}: test sets not downloaded (jeffref fetch --tasks {name})")
    for split in SPLITS:
        assert [r["id"] for r in tasks.task_sample(task, split)] == [r["id"] for r in tasks.recorded_sample(task, split)]
