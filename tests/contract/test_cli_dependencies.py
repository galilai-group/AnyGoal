from __future__ import annotations

import sys

import pytest

from anygoal import cli


@pytest.mark.parametrize(
    "code,error_type",
    [
        ("import anygoal_missing_dependency_for_test", "ModuleNotFoundError"),
        (
            "from importlib.metadata import version; version('anygoal_missing_dependency_for_test')",
            "PackageNotFoundError",
        ),
    ],
)
def test_missing_module_in_runner_preserves_error_and_adds_hint(
    code, error_type, capfd
):
    result = cli._execute(
        [sys.executable, "-c", code],
        dry_run=False,
    )
    assert result == 1
    stderr = capfd.readouterr().err
    assert error_type in stderr
    assert "anygoal_missing_dependency_for_test" in stderr
    assert "anygoal[data]" in stderr
    assert "anygoal[eval]" in stderr
    assert "anygoal[all]" in stderr


def test_other_runner_failure_does_not_add_dependency_hint(capfd):
    result = cli._execute(
        [sys.executable, "-c", "raise ValueError('invalid evaluation input')"],
        dry_run=False,
    )
    assert result == 1
    stderr = capfd.readouterr().err
    assert "ValueError: invalid evaluation input" in stderr
    assert "anygoal[eval]" not in stderr


def test_successful_runner_streams_output_without_hint(capfd):
    result = cli._execute(
        [
            sys.executable,
            "-c",
            "import sys; print('result'); print('progress', file=sys.stderr)",
        ],
        dry_run=False,
    )
    assert result == 0
    output = capfd.readouterr()
    assert output.out == "result\n"
    assert output.err == "progress\n"


def test_missing_module_in_cli_handler_adds_hint(monkeypatch, capsys):
    def missing_dependency(*args):
        raise ModuleNotFoundError("No module named 'example_optional_dependency'")

    monkeypatch.setattr(cli, "collect_from_cli", missing_dependency)
    assert cli.main(["collect", "avoidt"]) == 1
    stderr = capsys.readouterr().err
    assert "example_optional_dependency" in stderr
    assert "anygoal[eval]" in stderr


def test_dry_run_does_not_launch_runner(capsys):
    assert cli._execute(["anygoal_nonexistent_executable"], dry_run=True) == 0
    output = capsys.readouterr()
    assert output.out == "anygoal_nonexistent_executable\n"
    assert not output.err
