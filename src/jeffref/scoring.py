"""Scoring with the same definitions Jeff's evaluation uses (jeff.evaluate): accuracy, and expected calibration error
over 15 equal-width confidence bins, where a row's confidence is the probability of its predicted option.

For the 27B, the decision is the option named in its strict-JSON reply; a reply that is not valid JSON or names no
option counts as wrong ("invalid"). Its calibration uses the per-option probabilities read from the answer token."""

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jeff.evaluate import Prediction, fit_temperature, label_index, make_prediction, metrics, options
from jeff.types import Example


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile (fraction in [0, 1]) of a nonempty list."""
    if not values:
        raise ValueError("percentile of an empty list")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def latency_summary(seconds: Sequence[float]) -> dict[str, float]:
    return {"median_ms": 1000 * percentile(seconds, 0.5), "p95_ms": 1000 * percentile(seconds, 0.95),
            "mean_ms": 1000 * sum(seconds) / len(seconds), "decisions_per_second": len(seconds) / sum(seconds)}


def softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    if not values or temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("softmax needs values and a positive temperature")
    top = max(values)
    weights = [math.exp((v - top) / temperature) for v in values]
    total = sum(weights)
    return [w / total for w in weights]


def big_prediction(row: Example, record: dict[str, Any], temperature: float = 1.0) -> Prediction:
    """A jeff.evaluate Prediction for one 27B answer record (as written by jeffref.bench).

    `probabilities` come from the option log-probabilities at `temperature`; `prediction` is the JSON answer, or,
    when the reply was invalid, the probability argmax with `correct` forced to False and `invalid` set."""
    labels = options(row["question"])
    log_probs = record["option_log_probs"]
    if len(log_probs) != len(labels):
        raise ValueError(f"{row['id']}: {len(log_probs)} option log-probabilities for {len(labels)} options")
    probabilities = softmax(log_probs, temperature)
    code = record["answer_code"]
    if code is None:
        prediction = make_prediction(row, probabilities, temperature=temperature)
        prediction["correct"] = False
        prediction["answer"] = {"invalid": True, "reply": record["reply"]}
        return prediction
    index = record["codes"].index(code)
    return make_prediction(row, probabilities, prediction=labels[index], temperature=temperature)


def summarize(predictions: Sequence[Prediction]) -> dict[str, Any]:
    """jeff.evaluate.metrics, with the accuracy taken from each row's `correct` (so invalid replies count as wrong)
    and the ECE recomputed on the same bins with that correctness."""
    result: dict[str, Any] = dict(metrics(predictions))
    result["accuracy"] = sum(p["correct"] for p in predictions) / len(predictions)
    result["invalid"] = sum(bool(p.get("answer", {}).get("invalid")) for p in predictions)
    result.pop("reliability")
    return result


def fitted_temperature(rows: Sequence[Example], records: Sequence[dict[str, Any]]) -> float:
    """One temperature for the 27B's option log-probabilities, fitted by jeff.evaluate.fit_temperature (hard-label
    negative log-likelihood) on separate calibration rows, as Jeff's own temperatures are."""
    logits = [record["option_log_probs"] for record in records]
    targets = [label_index(options(row["question"]), row["label"]) for row in rows]
    return fit_temperature(logits, targets)


def read_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        record = json.loads(line)
        if record["id"] in records:
            raise ValueError(f"{path} has two records for {record['id']}")
        records[record["id"]] = record
    return records
