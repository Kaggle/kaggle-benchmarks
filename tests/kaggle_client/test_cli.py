# Copyright 2026 Kaggle Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the kaggle-bench CLI entry point."""

import pytest
from unittest.mock import MagicMock, patch


def test_cli_help(capsys):
    """kaggle-bench --help exits cleanly and shows subcommands."""
    from kaggle_benchmarks.kaggle_client.cli import run
    with pytest.raises(SystemExit) as exc:
        with patch("sys.argv", ["kaggle-bench", "--help"]):
            run()
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "run" in captured.out
    assert "fork" in captured.out


def test_cli_run_help(capsys):
    """kaggle-bench run --help shows run-specific options."""
    from kaggle_benchmarks.kaggle_client.cli import run
    with pytest.raises(SystemExit) as exc:
        with patch("sys.argv", ["kaggle-bench", "run", "--help"]):
            run()
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "notebook_slug" in captured.out
    assert "--force" in captured.out
    assert "--wait" in captured.out


def test_cli_fork_help(capsys):
    """kaggle-bench fork --help shows fork-specific options."""
    from kaggle_benchmarks.kaggle_client.cli import run
    with pytest.raises(SystemExit) as exc:
        with patch("sys.argv", ["kaggle-bench", "fork", "--help"]):
            run()
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "source_notebook_id" in captured.out
    assert "--overwrite" in captured.out


def test_cli_run_calls_publish_and_run(tmp_path):
    """kaggle-bench run invokes publish_and_run with correct args."""
    from kaggle_benchmarks.kaggle_client.cli import run
    mock_client = MagicMock()
    mock_client.publish_and_run.return_value = "https://kaggle.com/gastondana627/my-bench"
    with patch("kaggle_benchmarks.kaggle_client.cli._get_client", return_value=mock_client):
        with patch("sys.argv", ["kaggle-bench", "run", "my-bench"]):
            run()
    mock_client.publish_and_run.assert_called_once_with(
        notebook_slug="my-bench",
        source_file=None,
        dataset_sources=None,
        force=False,
    )


def test_cli_fork_calls_fork(tmp_path):
    """kaggle-bench fork invokes client.fork with correct args."""
    from kaggle_benchmarks.kaggle_client.cli import run
    mock_client = MagicMock()
    mock_client.fork.return_value = tmp_path
    with patch("kaggle_benchmarks.kaggle_client.cli._get_client", return_value=mock_client):
        with patch("sys.argv", ["kaggle-bench", "fork", "alice/riddle-benchmark"]):
            run()
    mock_client.fork.assert_called_once_with(
        source_notebook_id="alice/riddle-benchmark",
        dest_notebook_slug=None,
        overwrite=False,
    )
