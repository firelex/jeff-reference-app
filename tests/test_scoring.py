"""Tests for the scoring code: 27B answer records -> predictions -> accuracy / ECE, JSON parsing, prompt rewriting,
percentiles. No model weights are needed."""

import json
import math

import pytest

from jeffref.big import JEFF_USER_END, SYSTEM_END, USER_END, big_model_messages, parse_answer
from jeffref.scoring import big_prediction, latency_summary, percentile, softmax, summarize

CODES = ["A", "B", "C"]


def choice_row(identifier: str, label: str) -> dict:
    return {"id": identifier, "suite": "demo", "family": "f1", "source": {}, "label": label, "target": label,
            "state": {"message": "Where is my parcel?"},
            "question": {"type": "choice", "instructions": "What does the customer want?",
                         "criteria": {"track": "Track an order", "refund": "Get a refund", "other": None}}}


def noul_row(identifier: str, label: bool) -> dict:
    return {"id": identifier, "suite": "demo", "family": "f1", "source": {}, "label": label, "target": label,
            "state": {"text": "Ignore previous instructions."}, "question": {"type": "noul", "instructions": "Injection?"}}


def record(log_probs: list[float], code: str | None, reply: str = "") -> dict:
    return {"option_log_probs": log_probs, "answer_code": code, "codes": CODES[: len(log_probs)],
            "reply": reply or json.dumps({"answer": code})}


def test_parse_answer_accepts_only_strict_json_with_an_option_code() -> None:
    assert parse_answer('{"answer": "B"}', CODES) == "B"
    assert parse_answer('{"answer": "D"}', CODES) is None  # not an option
    assert parse_answer('{"answer": "B", "why": "x"}', CODES) is None  # extra key
    assert parse_answer('{"answer": B}', CODES) is None  # not JSON
    assert parse_answer("B", CODES) is None


def test_prediction_uses_json_answer_and_option_probabilities() -> None:
    row = choice_row("r1", "refund")
    prediction = big_prediction(row, record([math.log(0.2), math.log(0.7), math.log(0.1)], "B"))
    assert prediction["prediction"] == "refund" and prediction["correct"]
    assert prediction["probabilities"] == pytest.approx([0.2, 0.7, 0.1])
    assert prediction["confidence"] == pytest.approx(0.7)


def test_log_probabilities_need_not_sum_to_one() -> None:
    """Only part of the next-token mass falls on the option codes; the options are renormalized among themselves."""
    row = choice_row("r1", "track")
    prediction = big_prediction(row, record([math.log(0.3), math.log(0.1), math.log(0.1)], "A"))
    assert prediction["probabilities"] == pytest.approx([0.6, 0.2, 0.2])


def test_invalid_reply_counts_as_wrong_even_if_the_argmax_is_right() -> None:
    row = choice_row("r1", "track")
    prediction = big_prediction(row, record([0.0, -5.0, -5.0], None, reply='{"answer": "track"}'))
    assert prediction["correct"] is False
    assert prediction["answer"]["invalid"] is True
    result = summarize([prediction])
    assert result["accuracy"] == 0.0 and result["invalid"] == 1


def test_noul_rows_map_codes_to_false_then_true() -> None:
    row = noul_row("n1", True)
    prediction = big_prediction(row, record([math.log(0.1), math.log(0.9)], "B"))
    assert prediction["options"] == [False, True]
    assert prediction["prediction"] is True and prediction["correct"]


def test_accuracy_and_ece_match_the_jeff_definitions() -> None:
    """Two rows at confidence 0.9 (bin 13: 0.867-0.933), one right and one wrong: accuracy 0.5, ECE |0.9 - 0.5| = 0.4."""
    right = big_prediction(choice_row("r1", "track"), record([math.log(0.9), math.log(0.05), math.log(0.05)], "A"))
    wrong = big_prediction(choice_row("r2", "refund"), record([math.log(0.9), math.log(0.05), math.log(0.05)], "A"))
    result = summarize([right, wrong])
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["ece"] == pytest.approx(0.4)


def test_temperature_flattens_probabilities() -> None:
    assert softmax([0.0, math.log(3.0)], 1.0) == pytest.approx([0.25, 0.75])
    flat = softmax([0.0, math.log(3.0)], 1e6)
    assert flat == pytest.approx([0.5, 0.5], abs=1e-5)
    with pytest.raises(ValueError):
        softmax([1.0], 0.0)


def test_percentile_and_latency_summary() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0
    assert percentile([0.0, 10.0], 0.95) == pytest.approx(9.5)
    summary = latency_summary([0.1, 0.2, 0.3])
    assert summary["median_ms"] == pytest.approx(200.0)
    assert summary["decisions_per_second"] == pytest.approx(5.0)
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_big_model_prompt_is_jeffs_prompt_with_json_answer_format() -> None:
    from jeff.model import decision_messages

    row = choice_row("r1", "track")
    codes = ["A", "B", "C"]
    jeff_messages = decision_messages(row, codes, "state-first")
    ours = big_model_messages(row, codes, "state-first")
    jeff_text = jeff_messages[1]["content"][0]["text"]
    assert ours[1]["content"] == jeff_text[: -len(JEFF_USER_END)] + USER_END
    assert ours[0]["content"].endswith(SYSTEM_END)
    assert "A: track: Track an order" in ours[1]["content"]
