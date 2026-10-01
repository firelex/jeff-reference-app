"""Argument parsing of the jeffref command."""

import pytest

from jeffref.cli import build_parser, main


def test_fetch_accepts_all_or_a_task_list() -> None:
    parser = build_parser()
    args = parser.parse_args(["fetch", "--base", "--tasks", "all", "--big"])
    assert args.base and args.big and len(args.tasks) == 9
    assert parser.parse_args(["fetch", "--tasks", "guard,tools"]).tasks == ["guard", "tools"]


def test_fetch_rejects_unknown_tasks_and_an_empty_request(capsys) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["fetch", "--tasks", "guard,nope"])
    with pytest.raises(SystemExit):
        main(["fetch"])
    assert "at least one of" in capsys.readouterr().err


def test_bench_commands_need_one_known_task() -> None:
    parser = build_parser()
    assert parser.parse_args(["bench-27b", "--task", "ground"]).task == "ground"
    assert parser.parse_args(["bench-27b", "--task", "ground", "--limit", "3"]).limit == 3
    assert parser.parse_args(["bench-jeff", "--task", "emotion"]).task == "emotion"
    for argv in (["bench-27b"], ["bench-jeff", "--task", "nope"], []):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_cascade_takes_no_arguments() -> None:
    assert build_parser().parse_args(["cascade"]).command == "cascade"


def test_only_one_model_command_at_a_time(tmp_path, monkeypatch) -> None:
    from jeffref import lock

    monkeypatch.setattr(lock, "LOCK", tmp_path / ".one-model.lock")
    with lock.one_model("bench-27b --task guard"):
        with pytest.raises(RuntimeError, match="bench-27b --task guard"):
            with lock.one_model("bench-jeff --task guard"):
                pass
    with lock.one_model("bench-jeff --task guard"):  # free again once the first command is done
        pass
