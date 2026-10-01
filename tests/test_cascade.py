"""The cascade: routing by confidence, each task's threshold rule, and the published report rebuilt from the recorded
results alone."""

import json

import pytest

from jeffref import cascade
from jeffref.cascade import MARGIN, RESULTS, big_alone, report, route


def rows(*items: tuple[float, bool, bool]) -> list[dict]:
    return [{"confidence": c, "jeff_correct": j, "jeff_s": 0.1, "big_correct": b, "big_s": 4.0} for c, j, b in items]


def test_route_sends_only_rows_below_the_threshold_to_the_27b() -> None:
    sample = rows((0.9, True, False), (0.4, False, True), (0.6, False, True))
    assert route(sample, 0.0) == {"accuracy": pytest.approx(1 / 3), "mean_s": pytest.approx(0.1), "to_27b": 0.0}
    at_half = route(sample, 0.5)  # only the 0.4 row goes to the 27B, whose answer is used
    assert at_half["accuracy"] == pytest.approx(2 / 3) and at_half["to_27b"] == pytest.approx(1 / 3)
    assert at_half["mean_s"] == pytest.approx(0.1 + 4.0 / 3)  # Jeff's time on every row plus the 27B's where sent
    assert route(sample, 1.01)["accuracy"] == big_alone(sample)["accuracy"]


def test_margin_is_one_point() -> None:
    assert MARGIN == 0.01


def test_published_report_is_rebuilt_exactly_from_the_recorded_results() -> None:
    result, text = report(RESULTS)
    assert text == (RESULTS / "cascade.md").read_text()
    assert json.loads(json.dumps(result)) == json.loads((RESULTS / "cascade.json").read_text())


def test_headline_is_the_mean_of_every_adapter_except_emotion() -> None:
    result, _ = report(RESULTS)
    names = [name for name in result["tasks"] if name != "emotion"]
    rows = [result["tasks"][name] for name in names]
    overall = result["overall"]
    assert overall["tasks"] == names and "emotion" not in names
    assert overall["cascade_accuracy"] == pytest.approx(sum(t["cascade"]["accuracy"] for t in rows) / len(rows))
    assert overall["big_alone_accuracy"] == pytest.approx(sum(t["big_alone"]["accuracy"] for t in rows) / len(rows))
    agent = ["guard", "triage", "support-intents", "tools", "ground"]
    assert result["inbox_agent"]["tasks"] == agent
    assert result["all_tasks"]["tasks"] == list(result["tasks"])
    # leaving emotion out makes the gain smaller, as the report says
    every = result["all_tasks"]
    assert (overall["cascade_accuracy"] - overall["big_alone_accuracy"]
            < every["cascade_accuracy"] - every["big_alone_accuracy"])


def test_each_threshold_is_the_lowest_that_beats_the_27b_by_the_margin_on_calibration_rows() -> None:
    result, _ = report(RESULTS)
    for name, task_result in result["tasks"].items():
        calibration = cascade.rows_for(cascade.TASKS[name], "calibration")
        assert calibration is not None
        needed = big_alone(calibration)["accuracy"] + MARGIN
        chosen = task_result["threshold"]
        assert route(calibration, chosen)["accuracy"] >= needed
        assert all(route(calibration, x)["accuracy"] < needed for x in cascade.THRESHOLDS if x < chosen)


def test_a_task_still_being_measured_is_left_out(tmp_path) -> None:
    """A task whose results are not all recorded is left out of the report instead of failing it."""
    import shutil
    shutil.copytree(RESULTS, tmp_path / "results")
    (tmp_path / "results" / "jeff-mlx" / "legal-clauses-predictions.jsonl").unlink()
    result, _ = report(tmp_path / "results")
    assert "legal-clauses" not in result["tasks"] and "legal-clauses" not in result["overall"]["tasks"]
