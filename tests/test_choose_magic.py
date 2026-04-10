# Copyright 2025 Kaggle Inc.
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

from unittest.mock import MagicMock, patch

import pytest

from kaggle_benchmarks.ui import ipython_magics
from kaggle_benchmarks.ui.ipython_magics import (
    _NO_RUNS_HTML,
    _NO_TASKS_HTML,
    choose,
)


@pytest.fixture
def working_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ipython_magics, "_KAGGLE_WORKING_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def mock_ipython():
    mock_shell = MagicMock()
    mock_shell.user_ns = {}
    with patch(
        "kaggle_benchmarks.ui.ipython_magics.core.getipython.get_ipython",
        return_value=mock_shell,
    ):
        yield mock_shell


def test_no_tasks_detected_displays_warning(working_dir, mock_ipython):
    """When no .task.json files exist, display the 'No tasks detected' warning."""
    with patch("IPython.display.display") as mock_display, patch(
        "IPython.display.HTML"
    ) as mock_html:
        choose("my_task")

        mock_html.assert_called_once_with(_NO_TASKS_HTML)
        mock_display.assert_called_once()


def test_no_runs_displays_warning(working_dir, mock_ipython):
    """When task files exist but no run files, display the 'run or evaluate' warning."""
    (working_dir / "my_task.task.json").write_text("{}")

    with patch("IPython.display.display") as mock_display, patch(
        "IPython.display.HTML"
    ) as mock_html:
        choose("my_task")

        mock_html.assert_called_once_with(_NO_RUNS_HTML)
        mock_display.assert_called_once()


def test_no_warning_when_task_and_run_exist(working_dir, mock_ipython):
    """When both task and run files exist, no warning should be displayed."""
    (working_dir / "my_task.task.json").write_text("{}")
    (working_dir / "my_task-run1.run.json").write_text("{}")

    with patch("IPython.display.display") as mock_display:
        choose("my_task")

        mock_display.assert_not_called()


def test_no_tasks_html_contains_decorator_link():
    """The 'no tasks' message should link to the @kbench.task decorator docs."""
    assert "@kbench.task" in _NO_TASKS_HTML
    assert "user_guide.md#the-kbenchtask-decorator" in _NO_TASKS_HTML


def test_no_runs_html_contains_run_evaluate_links():
    """The 'no runs' message should link to quick_start docs for .run()/.evaluate()."""
    assert ".run()" in _NO_RUNS_HTML
    assert ".evaluate()" in _NO_RUNS_HTML
    assert "quick_start.md#basic-task" in _NO_RUNS_HTML


def test_warning_html_uses_same_font_size():
    """Both warnings should use the same font size for text and code blocks."""
    assert "font-size: 13px" in _NO_TASKS_HTML
    assert "font-size: 13px" in _NO_RUNS_HTML
