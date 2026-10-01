"""Downloads from Hugging Face, each at an explicit revision:

- Jeff's base model, mstrasser/Jeff-Qwen3.5-0.8B at v1.2, into models/jeff-v1.2-0.8b;
- each task's adapter, mstrasser/Jeff-Qwen3.5-0.8B-<task> at v1.2 (the adapter files plus test.jsonl and
  calibration.jsonl), into models/adapters/<task>; the two data files are checked against the SHA-256 recorded in
  results/samples/<task>.json, so a benchmark runs only on the exact rows the published results used;
- Qwen3.8-27B at 8 bits, mlx-community/Qwen3.8-27B-8bit at a pinned commit, into the Hugging Face cache (about 29 GB).

A missing repository, revision or file stops with an error naming it."""

from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

from jeffref.big import BIG_REPO, BIG_REVISION
from jeffref.tasks import (ADAPTER_FILES, ADAPTERS, JEFF_CHECKPOINT, JEFF_REPO, JEFF_REVISION, SPLITS, Task,
                           check_download)

BASE_FILES = ("config.json", "model.safetensors", "readout.safetensors", "decision_config.json", "tokenizer.json",
              "tokenizer_config.json", "chat_template.jinja")


def download(repo: str, revision: str, local_dir: Path | None) -> Path:
    try:
        path = snapshot_download(repo, revision=revision, local_dir=local_dir)
    except (RepositoryNotFoundError, RevisionNotFoundError) as error:
        raise RuntimeError(f"Hugging Face has no {repo} at revision {revision!r} (or it is not public): {error}") from error
    return Path(path)


def require(path: Path, names: tuple[str, ...], repo: str, revision: str) -> None:
    missing = [name for name in names if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"{repo} at revision {revision} lacks {missing} (downloaded to {path})")


def fetch_base() -> Path:
    path = download(JEFF_REPO, JEFF_REVISION, JEFF_CHECKPOINT)
    require(path, BASE_FILES, JEFF_REPO, JEFF_REVISION)
    print(f"{JEFF_REPO} ({JEFF_REVISION}) -> {path}", flush=True)
    return path


def fetch_adapter(task: Task) -> Path:
    path = download(task.repo, JEFF_REVISION, ADAPTERS / task.name)
    require(path, ADAPTER_FILES + tuple(f"{split}.jsonl" for split in SPLITS), task.repo, JEFF_REVISION)
    for split in SPLITS:
        check_download(task, split)
    print(f"{task.repo} ({JEFF_REVISION}) -> {path}; test and calibration sets match results/samples/{task.name}.json",
          flush=True)
    return path


def fetch_big() -> Path:
    path = download(BIG_REPO, BIG_REVISION, None)
    require(path, ("config.json", "model.safetensors.index.json", "tokenizer.json"), BIG_REPO, BIG_REVISION)
    print(f"{BIG_REPO} ({BIG_REVISION[:12]}) -> {path}", flush=True)
    return path
