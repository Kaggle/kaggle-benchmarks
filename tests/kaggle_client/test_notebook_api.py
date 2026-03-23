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

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from requests.exceptions import HTTPError

from kaggle_benchmarks.kaggle_client.notebook_api import (
    BenchmarkNotebookClient,
    ConcurrentRunError,
    KaggleAuthError,
    RunResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_api():
    """Create a mock Kaggle API."""
    api = MagicMock()
    api.get_config_value.return_value = "testuser"
    return api


@pytest.fixture
def client(mock_api, tmp_path):
    """Create a BenchmarkNotebookClient with a mocked API."""
    with patch.object(BenchmarkNotebookClient, "_authenticate", return_value=mock_api):
        return BenchmarkNotebookClient(base_dir=tmp_path)


def _make_http_error(status_code):
    """Create a mock HTTPError with the given response status code."""
    response = MagicMock()
    response.status_code = status_code
    return HTTPError(response=response)


def _make_404_error():
    """Create a mock HTTPError with a 404 response."""
    return _make_http_error(404)


def _make_benchmark_py(workspace: Path) -> None:
    """Write a minimal benchmark.py with cell delimiters."""
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "benchmark.py").write_text("# %%\nprint('hello')\n")


def _fake_kernels_output(kernel, path, force):
    """Simulate kernels_output by creating the output directory."""
    Path(path).mkdir(parents=True, exist_ok=True)


def _fake_kernels_output_with_run_json(kernel, path, force):
    """Simulate kernels_output that produces a run.json."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    (p / "run.json").write_text('{"score": 0.95}')


def _fake_kernels_output_with_multiple_run_jsons(kernel, path, force):
    """Simulate kernels_output that produces multiple *.run.json files."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    (p / "task_1.run.json").write_text('{"score": 1}')
    (p / "task_2.run.json").write_text('{"score": 2}')


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def test_notebook_id(client):
    """_notebook_id should be username/slug."""
    assert client._notebook_id("my-notebook") == "testuser/my-notebook"


def test_tracking_url(client):
    """_tracking_url should build a valid Kaggle URL."""
    assert (
        client._tracking_url("my-notebook")
        == "https://www.kaggle.com/testuser/my-notebook"
    )


def test_workspace_path(client, tmp_path):
    """_workspace should return base_dir / slug."""
    ws = client._workspace("my-notebook")
    assert ws == tmp_path / "my-notebook"


# ---------------------------------------------------------------------------
# _normalize_status
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_status, expected",
    [
        ("complete", "complete"),
        ("Complete", "complete"),
        ("KernelWorkerStatus.complete", "complete"),
        ("kernelworkerstatus.running", "running"),
        ("error", "error"),
    ],
)
def test_normalize_status_strings(raw_status, expected):
    """_normalize_status should strip enum prefixes and lower-case."""
    assert BenchmarkNotebookClient._normalize_status(raw_status) == expected


def test_normalize_status_object_with_attribute():
    """Simulates an API response object with a .status attribute."""

    class FakeStatus:
        status = "running"

    assert BenchmarkNotebookClient._normalize_status(FakeStatus()) == "running"


# ---------------------------------------------------------------------------
# publish_and_run
# ---------------------------------------------------------------------------


def test_publish_and_run_basic(client, mock_api, tmp_path):
    """Happy path: workspace exists, benchmark.py exists, no prior kernel."""
    slug = "test-bench"
    _make_benchmark_py(tmp_path / slug)

    # No existing kernel (404)
    mock_api.kernels_status.side_effect = _make_404_error()

    url = client.publish_and_run(slug)

    assert "testuser/test-bench" in url
    mock_api.kernels_push.assert_called_once_with(str(tmp_path / slug))

    # Verify metadata was written
    meta = json.loads((tmp_path / slug / "kernel-metadata.json").read_text())
    assert meta["id"] == "testuser/test-bench"
    assert "personal-benchmark" in meta["keywords"]

    # Verify .ipynb was created
    assert (tmp_path / slug / "benchmark.ipynb").exists()


def test_publish_and_run_with_source_file(client, mock_api, tmp_path):
    """Source file is copied into the workspace before processing."""
    slug = "test-bench"
    source = tmp_path / "my_script.py"
    source.write_text("# %%\nimport kaggle_benchmarks as kbench\n")

    mock_api.kernels_status.side_effect = _make_404_error()

    _ = client.publish_and_run(slug, source_file=str(source))

    workspace = tmp_path / slug
    assert (workspace / "benchmark.py").exists()
    assert (workspace / "benchmark.ipynb").exists()
    mock_api.kernels_push.assert_called_once()


def test_publish_and_run_dataset_sources(client, mock_api, tmp_path):
    """Dataset sources are passed through to metadata."""
    slug = "test-bench"
    _make_benchmark_py(tmp_path / slug)
    mock_api.kernels_status.side_effect = _make_404_error()

    client.publish_and_run(slug, dataset_sources=["alice/data", "bob/more-data"])

    meta = json.loads((tmp_path / slug / "kernel-metadata.json").read_text())
    assert meta["dataset_sources"] == ["alice/data", "bob/more-data"]


def test_publish_and_run_concurrent_guard(client, mock_api, tmp_path):
    """Raises ConcurrentRunError when notebook is already running."""
    slug = "test-bench"
    _make_benchmark_py(tmp_path / slug)

    mock_api.kernels_status.return_value = "running"

    with pytest.raises(ConcurrentRunError, match="already running"):
        client.publish_and_run(slug)

    mock_api.kernels_push.assert_not_called()


def test_publish_and_run_concurrent_guard_queued(client, mock_api, tmp_path):
    """Raises ConcurrentRunError when notebook is queued."""
    slug = "test-bench"
    _make_benchmark_py(tmp_path / slug)

    mock_api.kernels_status.return_value = "queued"

    with pytest.raises(ConcurrentRunError, match="already running"):
        client.publish_and_run(slug)


def test_publish_and_run_concurrent_guard_non_404_error(client, mock_api, tmp_path):
    """Non-404 HTTPError during status check should propagate as HTTPError."""
    slug = "test-bench"
    _make_benchmark_py(tmp_path / slug)

    mock_api.kernels_status.side_effect = _make_http_error(500)

    with pytest.raises(HTTPError):
        client.publish_and_run(slug)

    mock_api.kernels_push.assert_not_called()


def test_publish_and_run_force(client, mock_api, tmp_path):
    """force=True bypasses the concurrent run guard entirely."""
    slug = "test-bench"
    _make_benchmark_py(tmp_path / slug)

    _ = client.publish_and_run(slug, force=True)

    # kernels_status should NOT be called (guard skipped)
    mock_api.kernels_status.assert_not_called()
    mock_api.kernels_push.assert_called_once()


def test_publish_and_run_missing_file(client, tmp_path):
    """Raises FileNotFoundError when benchmark.py doesn't exist."""
    with pytest.raises(FileNotFoundError, match="Benchmark file not found"):
        client.publish_and_run("nonexistent")


def test_publish_and_run_missing_source_file(client, tmp_path):
    """Raises FileNotFoundError when source_file doesn't exist."""
    with pytest.raises(FileNotFoundError, match="Source file not found"):
        client.publish_and_run("test-bench", source_file="/nonexistent/path.py")


# ---------------------------------------------------------------------------
# get_results
# ---------------------------------------------------------------------------


def test_get_results_complete_immediately(client, mock_api, tmp_path):
    """Kernel is already complete — downloads output immediately."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()

    mock_api.kernels_status.return_value = "complete"
    mock_api.kernels_output.side_effect = _fake_kernels_output_with_run_json

    result = client.get_results(slug)

    assert result.status == "complete"
    assert result.output_dir is not None
    assert result.error is None
    # Tests that it works with the single legacy `run.json` as well
    runs = dict(result.iter_runs())
    assert len(runs) == 1
    assert runs["run.json"] == {"score": 0.95}


def test_get_results_multiple_run_files(client, mock_api, tmp_path):
    """Kernel completes and produces multiple *.run.json files."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()

    mock_api.kernels_status.return_value = "complete"
    mock_api.kernels_output.side_effect = _fake_kernels_output_with_multiple_run_jsons

    result = client.get_results(slug)

    assert result.status == "complete"
    assert result.output_dir is not None

    runs = dict(result.iter_runs())
    assert len(runs) == 2
    assert runs["task_1.run.json"] == {"score": 1}
    assert runs["task_2.run.json"] == {"score": 2}


def test_get_results_polls_until_complete(client, mock_api, tmp_path, monkeypatch):
    """Polls through queued -> running -> complete."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()
    monkeypatch.setattr(time, "sleep", lambda s: None)

    statuses = iter(["queued", "running", "complete"])
    mock_api.kernels_status.side_effect = lambda *a, **k: next(statuses)
    mock_api.kernels_output.side_effect = _fake_kernels_output

    collected = []
    result = client.get_results(slug, poll_interval=0.01, on_status=collected.append)

    assert result.status == "complete"
    # Callbacks: "queued" (in-loop), "running" (in-loop), "complete" (post-loop)
    assert collected == ["queued", "running", "complete"]


def test_get_results_timeout(client, mock_api, tmp_path, monkeypatch):
    """Returns timeout when exceeding the timeout limit."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()
    monkeypatch.setattr(time, "sleep", lambda s: None)

    mock_api.kernels_status.return_value = "running"

    result = client.get_results(slug, poll_interval=1, timeout=0)

    assert result.status == "timeout"
    assert result.output_dir is None


def test_get_results_cancel(client, mock_api, tmp_path):
    """Returns cancelled when cancel_event is set."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()

    mock_api.kernels_status.return_value = "running"

    cancel_event = threading.Event()
    cancel_event.set()  # Set immediately

    result = client.get_results(slug, cancel_event=cancel_event)

    assert result.status == "cancelled"


def test_get_results_cancel_mid_wait(client, mock_api, tmp_path):
    """Returns cancelled when cancel_event fires during poll wait."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()

    mock_api.kernels_status.return_value = "running"

    cancel_event = threading.Event()

    # Set the cancel event after a short delay to trigger during wait()
    def cancel_after_delay():
        time.sleep(0.05)
        cancel_event.set()

    timer = threading.Thread(target=cancel_after_delay, daemon=True)
    timer.start()

    result = client.get_results(
        slug,
        poll_interval=10,  # Long poll — cancel fires during the wait
        cancel_event=cancel_event,
    )
    timer.join(timeout=5)

    assert result.status == "cancelled"


def test_get_results_error_status(client, mock_api, tmp_path):
    """Returns error when Kaggle reports an error status."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()

    mock_api.kernels_status.return_value = "error"

    result = client.get_results(slug)

    assert result.status == "error"
    assert result.error is not None
    assert "error" in result.error


def test_get_results_initial_404_retries(client, mock_api, tmp_path, monkeypatch):
    """Handles initial 404s after push with retries."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()
    monkeypatch.setattr(time, "sleep", lambda s: None)

    # First 2 calls return 404, then "complete"
    call_count = 0

    def status_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise _make_404_error()
        return "complete"

    mock_api.kernels_status.side_effect = status_side_effect
    mock_api.kernels_output.side_effect = _fake_kernels_output

    result = client.get_results(slug, poll_interval=0.01)

    assert result.status == "complete"
    assert call_count == 3  # 2 retries + 1 success


def test_get_results_all_retries_exhausted(client, mock_api, tmp_path, monkeypatch):
    """Returns error when all 404 retries are exhausted."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()
    monkeypatch.setattr(time, "sleep", lambda s: None)

    mock_api.kernels_status.side_effect = _make_404_error()

    result = client.get_results(slug)

    assert result.status == "error"
    assert "retries" in result.error


def test_get_results_non_404_error_during_retries(
    client, mock_api, tmp_path, monkeypatch
):
    """Non-404 HTTPError during initial retry loop should propagate immediately."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()
    monkeypatch.setattr(time, "sleep", lambda s: None)

    mock_api.kernels_status.side_effect = _make_http_error(500)

    with pytest.raises(HTTPError):
        client.get_results(slug)


def test_get_results_on_status_callback(client, mock_api, tmp_path, monkeypatch):
    """on_status callback receives intermediate and final statuses but deduplicates repeats."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()
    monkeypatch.setattr(time, "sleep", lambda s: None)

    statuses = iter(["running", "running", "complete"])
    mock_api.kernels_status.side_effect = lambda *a, **k: next(statuses)
    mock_api.kernels_output.side_effect = _fake_kernels_output

    collected = []
    result = client.get_results(slug, poll_interval=0.01, on_status=collected.append)

    assert result.status == "complete"
    assert collected == ["running", "complete"]


def test_get_results_no_run_json(client, mock_api, tmp_path):
    """iter_runs() yields nothing when no run.json is produced."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()

    mock_api.kernels_status.return_value = "complete"
    mock_api.kernels_output.side_effect = _fake_kernels_output

    result = client.get_results(slug)

    assert result.status == "complete"
    assert not dict(result.iter_runs())


def test_get_results_custom_output_dir(client, mock_api, tmp_path):
    """output_dir parameter overrides the default output path."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()

    custom_output = tmp_path / "my_custom_output"

    mock_api.kernels_status.return_value = "complete"
    mock_api.kernels_output.side_effect = _fake_kernels_output_with_run_json

    result = client.get_results(slug, output_dir=str(custom_output))

    assert result.status == "complete"
    assert result.output_dir == str(custom_output)
    assert custom_output.exists()

    runs = dict(result.iter_runs())
    assert len(runs) == 1


def test_get_results_clears_existing_output(client, mock_api, tmp_path):
    """Existing output files are cleared before downloading by default."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()

    # Create pre-existing output directory with an old file
    output_dir = tmp_path / slug / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    old_file = output_dir / "old_result.txt"
    old_file.write_text("stale data")

    mock_api.kernels_status.return_value = "complete"
    mock_api.kernels_output.side_effect = _fake_kernels_output_with_run_json

    result = client.get_results(slug)

    assert result.status == "complete"
    # Old file should have been removed
    assert not old_file.exists()


def test_get_results_no_clear_output(client, mock_api, tmp_path):
    """clear_output=False preserves existing files in the output directory."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()

    # Create pre-existing output directory with an old file
    output_dir = tmp_path / slug / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    old_file = output_dir / "old_result.txt"
    old_file.write_text("keep me")

    mock_api.kernels_status.return_value = "complete"
    mock_api.kernels_output.side_effect = _fake_kernels_output_with_run_json

    result = client.get_results(slug, clear_output=False)

    assert result.status == "complete"
    # Old file should still exist
    assert old_file.exists()
    assert old_file.read_text() == "keep me"
    # New file should also exist
    runs = dict(result.iter_runs())
    assert len(runs) == 1


# ---------------------------------------------------------------------------
# RunResult.iter_runs edge cases
# ---------------------------------------------------------------------------


def test_iter_runs_nonexistent_output_dir(tmp_path):
    """iter_runs() yields nothing when output_dir does not exist."""
    result = RunResult(
        status="complete",
        output_dir=str(tmp_path / "nonexistent"),
        tracking_url=None,
    )
    assert list(result.iter_runs()) == []
    assert dict(result.iter_runs()) == {}


def test_iter_runs_none_output_dir():
    """iter_runs() yields nothing when output_dir is None."""
    result = RunResult(
        status="error",
        output_dir=None,
        tracking_url=None,
    )
    assert list(result.iter_runs()) == []
    assert dict(result.iter_runs()) == {}


# ---------------------------------------------------------------------------
# fork
# ---------------------------------------------------------------------------


def _make_fake_pull(ipynb_name="notebook.ipynb"):
    """Create a fake kernels_pull that writes a notebook and metadata."""

    def fake_pull(notebook, path, metadata):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        notebook_data = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["print('hello')\n"],
                    "metadata": {},
                    "outputs": [],
                    "execution_count": None,
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        (p / ipynb_name).write_text(json.dumps(notebook_data))
        if metadata:
            (p / "kernel-metadata.json").write_text(
                json.dumps({"id": f"source/{ipynb_name.replace('.ipynb', '')}"})
            )

    return fake_pull


def _make_fake_pull_no_ipynb():
    """Create a fake kernels_pull that writes only metadata (no .ipynb)."""

    def fake_pull(notebook, path, metadata):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        # Write a plain .py script file instead of .ipynb
        (p / "script.py").write_text("print('hello')\n")
        if metadata:
            (p / "kernel-metadata.json").write_text(
                json.dumps({"id": "source/script-notebook", "code_file": "script.py"})
            )

    return fake_pull


def test_fork_basic(client, mock_api, tmp_path):
    """Happy path: pulls notebook and converts to .py."""
    mock_api.kernels_pull.side_effect = _make_fake_pull("riddle-benchmark.ipynb")

    result = client.fork("alice/riddle-benchmark")

    assert result == tmp_path / "riddle-benchmark"
    assert (result / "benchmark.py").exists()
    assert (result / "kernel-metadata.json").exists()
    assert (result / "benchmark.ipynb").exists()

    # Verify the .py file has cell delimiters
    py_content = (result / "benchmark.py").read_text()
    assert "# %%" in py_content
    assert "print('hello')" in py_content


def test_fork_custom_slug(client, mock_api, tmp_path):
    """Custom notebook_slug overrides the default."""
    mock_api.kernels_pull.side_effect = _make_fake_pull("notebook.ipynb")

    result = client.fork("alice/riddle-benchmark", dest_notebook_slug="my-riddle")

    assert result == tmp_path / "my-riddle"
    assert (result / "benchmark.py").exists()


def test_fork_exists_error(client, mock_api, tmp_path):
    """Raises FileExistsError when workspace already exists."""
    (tmp_path / "riddle-benchmark").mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        client.fork("alice/riddle-benchmark")


def test_fork_overwrite(client, mock_api, tmp_path):
    """overwrite=True removes the existing workspace."""
    workspace = tmp_path / "riddle-benchmark"
    workspace.mkdir()
    (workspace / "old-file.txt").write_text("old content")

    mock_api.kernels_pull.side_effect = _make_fake_pull("riddle-benchmark.ipynb")

    _ = client.fork("alice/riddle-benchmark", overwrite=True)

    assert not (workspace / "old-file.txt").exists()
    assert (workspace / "benchmark.py").exists()


def test_fork_missing_notebook_raises_value_error(client, mock_api):
    """fork() should raise a friendly ValueError for any HTTPError."""
    mock_api.kernels_pull.side_effect = _make_404_error()

    with pytest.raises(
        ValueError, match="Failed to pull notebook 'kaggle/does-not-exist'"
    ):
        client.fork("kaggle/does-not-exist")


def test_fork_http_error_500_raises_value_error(client, mock_api):
    """fork() wraps any HTTPError (including 500) in ValueError."""
    mock_api.kernels_pull.side_effect = _make_http_error(500)

    with pytest.raises(ValueError, match="Failed to pull notebook"):
        client.fork("kaggle/server-error-notebook")


def test_fork_no_ipynb_file(client, mock_api, tmp_path, caplog):
    """fork() should log a warning when no .ipynb file is found (script notebook)."""
    mock_api.kernels_pull.side_effect = _make_fake_pull_no_ipynb()

    import logging

    with caplog.at_level(logging.WARNING):
        result = client.fork("alice/script-notebook")

    assert result == tmp_path / "script-notebook"
    assert (result / "kernel-metadata.json").exists()
    # No .ipynb → no benchmark.py conversion
    assert not (result / "benchmark.py").exists()
    # Warning should be logged
    assert "No .ipynb file found" in caplog.text


# ---------------------------------------------------------------------------
# validate_and_get_username
# ---------------------------------------------------------------------------


def test_validate_and_get_username_success(client, mock_api):
    """Returns the username when credentials are valid."""
    assert client.validate_and_get_username() == "testuser"
    mock_api.get_config_value.assert_called_with("username")


def test_validate_and_get_username_failure(client, mock_api):
    """Raises KaggleAuthError when credentials are invalid."""
    mock_api.get_config_value.side_effect = Exception("Invalid credentials")

    with pytest.raises(KaggleAuthError, match="invalid or missing"):
        client.validate_and_get_username()


def test_validate_and_get_username_empty(client, mock_api):
    """Raises KaggleAuthError when username is empty."""
    mock_api.get_config_value.return_value = ""

    with pytest.raises(KaggleAuthError, match="invalid or missing"):
        client.validate_and_get_username()


# ---------------------------------------------------------------------------
# _authenticate
# ---------------------------------------------------------------------------


def test_authenticate_import_error():
    """Raises KaggleAuthError when kaggle package is not installed."""
    with patch.dict(
        "sys.modules",
        {"kaggle": None, "kaggle.api": None, "kaggle.api.kaggle_api_extended": None},
    ):
        # Force ImportError by removing the module from sys.modules cache
        import sys

        saved = {}
        for mod_name in list(sys.modules):
            if mod_name.startswith("kaggle"):
                saved[mod_name] = sys.modules.pop(mod_name)

        try:
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'kaggle'"),
            ):
                with pytest.raises(KaggleAuthError, match="kaggle.*required"):
                    BenchmarkNotebookClient._authenticate()
        finally:
            sys.modules.update(saved)


def test_authenticate_os_error():
    """Raises KaggleAuthError when authentication credentials fail."""
    mock_kaggle_api_cls = MagicMock()
    mock_api_instance = MagicMock()
    mock_api_instance.authenticate.side_effect = OSError("No such file: kaggle.json")
    mock_kaggle_api_cls.return_value = mock_api_instance

    # Test the actual _authenticate method with mocked import
    with patch.dict(
        "sys.modules",
        {
            "kaggle": MagicMock(),
            "kaggle.api": MagicMock(),
            "kaggle.api.kaggle_api_extended": MagicMock(KaggleApi=mock_kaggle_api_cls),
        },
    ):
        with pytest.raises(KaggleAuthError, match="authentication failed"):
            BenchmarkNotebookClient._authenticate()
