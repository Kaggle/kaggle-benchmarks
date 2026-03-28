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

import io
import json
import tarfile
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
    _authenticate,
)

_API_MOD = "kaggle_benchmarks.kaggle_client.notebook_api"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_api():
    """Create a mock KaggleClient (kagglesdk)."""
    api = MagicMock()
    # Set username for Basic auth scenarios
    api.username = "testuser"
    api.api_token = None
    return api


@pytest.fixture
def client(mock_api, tmp_path):
    """Create a BenchmarkNotebookClient with a mocked API."""
    with patch(f"{_API_MOD}._authenticate", return_value=(mock_api, "testuser")):
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


def _make_save_kernel_response(error=""):
    """Create a mock ApiSaveKernelResponse."""
    resp = MagicMock()
    resp.error = error
    resp.invalid_tags = []
    return resp


def _make_status_response(status_str):
    """Create a mock ApiGetKernelSessionStatusResponse."""

    class FakeStatus:
        status = status_str

    return FakeStatus()


def _make_archive_response(*files):
    """Create a mock streamed response with a tar archive containing the given files.

    Args:
        *files: Tuples of (filename, content_string) to include in the archive.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, content in files:
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    buf.seek(0)

    response = MagicMock()
    response.iter_content = lambda chunk_size=8192: iter([buf.read()])
    return response


def _make_get_kernel_response(
    source_json=None, metadata_dict=None, ipynb_name="notebook.ipynb"
):
    """Create a mock ApiGetKernelResponse.

    Args:
        source_json: The notebook source (raw .ipynb JSON dict). If None, uses a default.
        metadata_dict: The metadata dict. If None, uses a default.
        ipynb_name: Used to construct the default metadata 'ref'.
    """
    if source_json is None:
        source_json = {
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

    if metadata_dict is None:
        metadata_dict = {
            "ref": f"source/{ipynb_name.replace('.ipynb', '')}",
            "title": ipynb_name.replace(".ipynb", ""),
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_internet": True,
            "enable_tpu": False,
            "dataset_data_sources": [],
            "competition_data_sources": [],
            "kernel_data_sources": [],
            "model_data_sources": [],
            "category_ids": [],
        }

    import types

    resp = MagicMock()
    resp.blob.source = json.dumps(source_json)
    # Use SimpleNamespace so getattr(meta, "missing_field", None) returns None correctly
    resp.metadata = types.SimpleNamespace(**metadata_dict)
    return resp


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
# publish_and_run
# ---------------------------------------------------------------------------


def test_publish_and_run_basic(client, mock_api, tmp_path):
    """Happy path: workspace exists, benchmark.py exists, no prior kernel."""
    slug = "test-bench"
    _make_benchmark_py(tmp_path / slug)

    # No existing kernel (404)
    mock_api.kernels.kernels_api_client.get_kernel_session_status.side_effect = (
        _make_404_error()
    )
    mock_api.kernels.kernels_api_client.save_kernel.return_value = (
        _make_save_kernel_response()
    )

    url = client.publish_and_run(slug)

    assert "testuser/test-bench" in url
    mock_api.kernels.kernels_api_client.save_kernel.assert_called_once()

    # Verify the save_kernel request has correct slug
    call_args = mock_api.kernels.kernels_api_client.save_kernel.call_args
    req = call_args[0][0]
    assert req.slug == "testuser/test-bench"

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

    mock_api.kernels.kernels_api_client.get_kernel_session_status.side_effect = (
        _make_404_error()
    )
    mock_api.kernels.kernels_api_client.save_kernel.return_value = (
        _make_save_kernel_response()
    )

    _ = client.publish_and_run(slug, source_file=str(source))

    workspace = tmp_path / slug
    assert (workspace / "benchmark.py").exists()
    assert (workspace / "benchmark.ipynb").exists()
    mock_api.kernels.kernels_api_client.save_kernel.assert_called_once()


def test_publish_and_run_dataset_sources(client, mock_api, tmp_path):
    """Dataset sources are passed through to metadata."""
    slug = "test-bench"
    _make_benchmark_py(tmp_path / slug)
    mock_api.kernels.kernels_api_client.get_kernel_session_status.side_effect = (
        _make_404_error()
    )
    mock_api.kernels.kernels_api_client.save_kernel.return_value = (
        _make_save_kernel_response()
    )

    client.publish_and_run(slug, dataset_sources=["alice/data", "bob/more-data"])

    meta = json.loads((tmp_path / slug / "kernel-metadata.json").read_text())
    assert meta["dataset_sources"] == ["alice/data", "bob/more-data"]


def test_publish_and_run_concurrent_guard(client, mock_api, tmp_path):
    """Raises ConcurrentRunError when notebook is already running."""
    slug = "test-bench"
    _make_benchmark_py(tmp_path / slug)

    mock_api.kernels.kernels_api_client.get_kernel_session_status.return_value = (
        _make_status_response("running")
    )

    with pytest.raises(ConcurrentRunError, match="already running"):
        client.publish_and_run(slug)

    mock_api.kernels.kernels_api_client.save_kernel.assert_not_called()


def test_publish_and_run_concurrent_guard_queued(client, mock_api, tmp_path):
    """Raises ConcurrentRunError when notebook is queued."""
    slug = "test-bench"
    _make_benchmark_py(tmp_path / slug)

    mock_api.kernels.kernels_api_client.get_kernel_session_status.return_value = (
        _make_status_response("queued")
    )

    with pytest.raises(ConcurrentRunError, match="already running"):
        client.publish_and_run(slug)


def test_publish_and_run_concurrent_guard_non_404_error(client, mock_api, tmp_path):
    """Non-404 HTTPError during status check should propagate as HTTPError."""
    slug = "test-bench"
    _make_benchmark_py(tmp_path / slug)

    mock_api.kernels.kernels_api_client.get_kernel_session_status.side_effect = (
        _make_http_error(500)
    )

    with pytest.raises(HTTPError):
        client.publish_and_run(slug)

    mock_api.kernels.kernels_api_client.save_kernel.assert_not_called()


def test_publish_and_run_force(client, mock_api, tmp_path):
    """force=True bypasses the concurrent run guard entirely."""
    slug = "test-bench"
    _make_benchmark_py(tmp_path / slug)

    mock_api.kernels.kernels_api_client.save_kernel.return_value = (
        _make_save_kernel_response()
    )

    _ = client.publish_and_run(slug, force=True)

    # get_kernel_session_status should NOT be called (guard skipped)
    mock_api.kernels.kernels_api_client.get_kernel_session_status.assert_not_called()
    mock_api.kernels.kernels_api_client.save_kernel.assert_called_once()


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

    mock_api.kernels.kernels_api_client.get_kernel_session_status.return_value = (
        _make_status_response("complete")
    )
    mock_api.kernels.kernels_api_client.download_kernel_output.return_value = (
        _make_archive_response(("run.json", '{"score": 0.95}'))
    )

    result = client.get_results(slug)

    assert result.status == "complete"
    assert result.output_dir is not None
    assert result.error is None
    # Tests that it works with the single legacy `run.json` as well
    runs = dict(result.iter_run_results())
    assert len(runs) == 1
    assert runs["run.json"] == {"score": 0.95}


def test_get_results_multiple_run_files(client, mock_api, tmp_path):
    """Kernel completes and produces multiple *.run.json files."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()

    mock_api.kernels.kernels_api_client.get_kernel_session_status.return_value = (
        _make_status_response("complete")
    )
    mock_api.kernels.kernels_api_client.download_kernel_output.return_value = (
        _make_archive_response(
            ("task_1.run.json", '{"score": 1}'),
            ("task_2.run.json", '{"score": 2}'),
        )
    )

    result = client.get_results(slug)

    assert result.status == "complete"
    assert result.output_dir is not None

    runs = dict(result.iter_run_results())
    assert len(runs) == 2
    assert runs["task_1.run.json"] == {"score": 1}
    assert runs["task_2.run.json"] == {"score": 2}


def test_get_results_polls_until_complete(client, mock_api, tmp_path, monkeypatch):
    """Polls through queued -> running -> complete."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()
    monkeypatch.setattr(time, "sleep", lambda s: None)

    statuses = iter(
        [
            _make_status_response("queued"),
            _make_status_response("running"),
            _make_status_response("complete"),
        ]
    )
    mock_api.kernels.kernels_api_client.get_kernel_session_status.side_effect = (
        lambda *a, **k: next(statuses)
    )
    mock_api.kernels.kernels_api_client.download_kernel_output.return_value = (
        _make_archive_response()
    )

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

    mock_api.kernels.kernels_api_client.get_kernel_session_status.return_value = (
        _make_status_response("running")
    )

    result = client.get_results(slug, poll_interval=1, timeout=0)

    assert result.status == "timeout"
    assert result.output_dir is None


def test_get_results_cancel(client, mock_api, tmp_path):
    """Returns cancelled when cancel_event is set."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()

    mock_api.kernels.kernels_api_client.get_kernel_session_status.return_value = (
        _make_status_response("running")
    )

    cancel_event = threading.Event()
    cancel_event.set()  # Set immediately

    result = client.get_results(slug, cancel_event=cancel_event)

    assert result.status == "cancelled"


def test_get_results_cancel_mid_wait(client, mock_api, tmp_path):
    """Returns cancelled when cancel_event fires during poll wait."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()

    mock_api.kernels.kernels_api_client.get_kernel_session_status.return_value = (
        _make_status_response("running")
    )

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

    mock_api.kernels.kernels_api_client.get_kernel_session_status.return_value = (
        _make_status_response("error")
    )

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
        return _make_status_response("complete")

    mock_api.kernels.kernels_api_client.get_kernel_session_status.side_effect = (
        status_side_effect
    )
    mock_api.kernels.kernels_api_client.download_kernel_output.return_value = (
        _make_archive_response()
    )

    result = client.get_results(slug, poll_interval=0.01)

    assert result.status == "complete"
    assert call_count == 3  # 2 retries + 1 success


def test_get_results_all_retries_exhausted(client, mock_api, tmp_path, monkeypatch):
    """Returns error when all 404 retries are exhausted."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()
    monkeypatch.setattr(time, "sleep", lambda s: None)

    mock_api.kernels.kernels_api_client.get_kernel_session_status.side_effect = (
        _make_404_error()
    )

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

    mock_api.kernels.kernels_api_client.get_kernel_session_status.side_effect = (
        _make_http_error(500)
    )

    with pytest.raises(HTTPError):
        client.get_results(slug)


def test_get_results_on_status_callback(client, mock_api, tmp_path, monkeypatch):
    """on_status callback receives intermediate and final statuses but deduplicates repeats."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()
    monkeypatch.setattr(time, "sleep", lambda s: None)

    statuses = iter(
        [
            _make_status_response("running"),
            _make_status_response("running"),
            _make_status_response("complete"),
        ]
    )
    mock_api.kernels.kernels_api_client.get_kernel_session_status.side_effect = (
        lambda *a, **k: next(statuses)
    )
    mock_api.kernels.kernels_api_client.download_kernel_output.return_value = (
        _make_archive_response()
    )

    collected = []
    result = client.get_results(slug, poll_interval=0.01, on_status=collected.append)

    assert result.status == "complete"
    assert collected == ["running", "complete"]


def test_get_results_no_run_json(client, mock_api, tmp_path):
    """iter_run_results() yields nothing when no run.json is produced."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()

    mock_api.kernels.kernels_api_client.get_kernel_session_status.return_value = (
        _make_status_response("complete")
    )
    mock_api.kernels.kernels_api_client.download_kernel_output.return_value = (
        _make_archive_response()
    )

    result = client.get_results(slug)

    assert result.status == "complete"
    assert not dict(result.iter_run_results())


def test_get_results_custom_output_dir(client, mock_api, tmp_path):
    """output_dir parameter overrides the default output path."""
    slug = "test-bench"
    (tmp_path / slug).mkdir()

    custom_output = tmp_path / "my_custom_output"

    mock_api.kernels.kernels_api_client.get_kernel_session_status.return_value = (
        _make_status_response("complete")
    )
    mock_api.kernels.kernels_api_client.download_kernel_output.return_value = (
        _make_archive_response(("run.json", '{"score": 0.95}'))
    )

    result = client.get_results(slug, output_dir=str(custom_output))

    assert result.status == "complete"
    assert result.output_dir == str(custom_output)
    assert custom_output.exists()

    runs = dict(result.iter_run_results())
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

    mock_api.kernels.kernels_api_client.get_kernel_session_status.return_value = (
        _make_status_response("complete")
    )
    mock_api.kernels.kernels_api_client.download_kernel_output.return_value = (
        _make_archive_response(("run.json", '{"score": 0.95}'))
    )

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

    mock_api.kernels.kernels_api_client.get_kernel_session_status.return_value = (
        _make_status_response("complete")
    )
    mock_api.kernels.kernels_api_client.download_kernel_output.return_value = (
        _make_archive_response(("run.json", '{"score": 0.95}'))
    )

    result = client.get_results(slug, clear_output=False)

    assert result.status == "complete"
    # Old file should still exist
    assert old_file.exists()
    assert old_file.read_text() == "keep me"
    # New file should also exist
    runs = dict(result.iter_run_results())
    assert len(runs) == 1


# ---------------------------------------------------------------------------
# RunResult.iter_run_results edge cases
# ---------------------------------------------------------------------------


def test_iter_run_results_nonexistent_output_dir(tmp_path):
    """iter_run_results() yields nothing when output_dir does not exist."""
    result = RunResult(
        status="complete",
        output_dir=str(tmp_path / "nonexistent"),
        tracking_url=None,
    )
    assert list(result.iter_run_results()) == []
    assert dict(result.iter_run_results()) == {}


def test_iter_run_results_none_output_dir():
    """iter_run_results() yields nothing when output_dir is None."""
    result = RunResult(
        status="error",
        output_dir=None,
        tracking_url=None,
    )
    assert list(result.iter_run_results()) == []
    assert dict(result.iter_run_results()) == {}


# ---------------------------------------------------------------------------
# fork
# ---------------------------------------------------------------------------


def test_fork_basic(client, mock_api, tmp_path):
    """Happy path: pulls notebook and converts to .py."""
    mock_api.kernels.kernels_api_client.get_kernel.return_value = (
        _make_get_kernel_response(ipynb_name="riddle-benchmark.ipynb")
    )

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
    mock_api.kernels.kernels_api_client.get_kernel.return_value = (
        _make_get_kernel_response(ipynb_name="notebook.ipynb")
    )

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

    mock_api.kernels.kernels_api_client.get_kernel.return_value = (
        _make_get_kernel_response(ipynb_name="riddle-benchmark.ipynb")
    )

    _ = client.fork("alice/riddle-benchmark", overwrite=True)

    assert not (workspace / "old-file.txt").exists()
    assert (workspace / "benchmark.py").exists()


def test_fork_missing_notebook_raises_value_error(client, mock_api):
    """fork() should raise a friendly ValueError for any HTTPError."""
    mock_api.kernels.kernels_api_client.get_kernel.side_effect = _make_404_error()

    with pytest.raises(
        ValueError, match="Failed to pull notebook 'kaggle/does-not-exist'"
    ):
        client.fork("kaggle/does-not-exist")


def test_fork_http_error_500_raises_value_error(client, mock_api):
    """fork() wraps any HTTPError (including 500) in ValueError."""
    mock_api.kernels.kernels_api_client.get_kernel.side_effect = _make_http_error(500)

    with pytest.raises(ValueError, match="Failed to pull notebook"):
        client.fork("kaggle/server-error-notebook")


def test_fork_no_source(client, mock_api, tmp_path, caplog):
    """fork() should log a warning when no source is found (empty blob)."""
    resp = MagicMock()
    resp.blob.source = None
    resp.metadata = None  # No metadata either
    mock_api.kernels.kernels_api_client.get_kernel.return_value = resp

    import logging

    with caplog.at_level(logging.WARNING):
        result = client.fork("alice/script-notebook")

    assert result == tmp_path / "script-notebook"
    # No source → no benchmark.py conversion
    assert not (result / "benchmark.py").exists()
    # Warning should be logged
    assert "No source found" in caplog.text


# ---------------------------------------------------------------------------
# _authenticate (integration of the above)
# ---------------------------------------------------------------------------


def test_authenticate_import_error():
    """Raises KaggleAuthError when kagglesdk package is not installed."""
    with patch.dict(
        "sys.modules",
        {
            "kagglesdk": None,
            "kagglesdk.kaggle_client": None,
            "kagglesdk.kaggle_env": None,
        },
    ):
        # Force ImportError by removing the module from sys.modules cache
        import sys

        saved = {}
        for mod_name in list(sys.modules):
            if mod_name.startswith("kagglesdk"):
                saved[mod_name] = sys.modules.pop(mod_name)

        try:
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'kagglesdk'"),
            ):
                with pytest.raises(KaggleAuthError, match="kagglesdk.*required"):
                    _authenticate()
        finally:
            sys.modules.update(saved)
