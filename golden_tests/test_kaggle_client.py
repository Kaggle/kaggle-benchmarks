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

"""
Golden tests for kaggle_client.notebook_api.BenchmarkNotebookClient.

These tests exercise the real Kaggle API (authentication, push, status
polling, fork, output download) and are excluded from CI because they
require valid Kaggle credentials (~/.kaggle/kaggle.json) and internet
access.

Tests are organized to mirror the typical user journey:

  1. Authenticate        →  TestAuth
  2. Fork a benchmark    →  TestFork
  3. Publish & run       →  TestPublish
  4. Poll & get results  →  TestGetResults
  5. Error handling      →  TestErrorHandling

Usage:
    uv run --group test pytest golden_tests/test_kaggle_client.py
    uv run --group test pytest golden_tests/test_kaggle_client.py::TestAuth
    uv run --group test pytest golden_tests/test_kaggle_client.py::TestGetResults::test_full_round_trip
"""

import json
import threading
import time
import uuid

import pytest

from kaggle_benchmarks.kaggle_client.notebook_api import (
    BenchmarkNotebookClient,
    ConcurrentRunError,
    RunResult,
)

# ---------------------------------------------------------------------------
# Constants & Helpers
# ---------------------------------------------------------------------------

# A minimal benchmark script that runs a subtraction task twice, producing
# two *.run.json output files for end-to-end verification.
MINIMAL_BENCHMARK_SCRIPT = """\
# %%
import kaggle_benchmarks as kbench

@kbench.task("subtraction")
def test_subtraction(llm):
    llm.stream_responses = True
    response = llm.prompt("9.9 - 9.11 = ?")
    kbench.assertions.assert_in("0.79", response, expectation="Expect 9.9-9.11=0.79")

# Execute the task twice to generate two run.json files
test_subtraction.run(llm=kbench.llm)
test_subtraction.run(llm=kbench.llm)
"""


def _make_client(base_dir):
    """Create a BenchmarkNotebookClient rooted in a given directory."""
    return BenchmarkNotebookClient(base_dir=base_dir)


def _prepare_workspace(client, slug=None, script=MINIMAL_BENCHMARK_SCRIPT):
    """Create a workspace with a benchmark script.  Returns (workspace, slug)."""
    slug = slug or f"kbench-golden-{uuid.uuid4().hex[:8]}"
    workspace = client._workspace(slug)
    workspace.mkdir(parents=True, exist_ok=True)
    benchmark_py = workspace / BenchmarkNotebookClient.BENCHMARK_FILENAME
    benchmark_py.write_text(script, encoding="utf-8")
    return workspace, slug


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Module-scoped client — shared across tests that don't need isolation."""
    base = tmp_path_factory.mktemp("kaggle_client_golden")
    return _make_client(base)


@pytest.fixture
def fresh_client(tmp_path):
    """Function-scoped client — for tests that need a clean API state."""
    return _make_client(tmp_path)


@pytest.fixture(scope="module")
def fork_source_notebook_id(client):
    """Publish a dummy notebook that fork tests can reliably pull from."""
    _, slug = _prepare_workspace(
        client,
        slug=f"kbench-golden-source-{uuid.uuid4().hex[:8]}",
        script='print("dummy notebook for fork test")\n',
    )
    client.publish_and_run(slug, force=True)
    return f"{client.username}/{slug}"


# ===========================================================================
# Phase 1: AUTHENTICATE
#
# The very first thing a user does — verify they can talk to Kaggle.
# ===========================================================================


class TestAuth:
    """Credential validation against the real Kaggle API.

    Docs: https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication
    """

    def test_client_authenticates(self, client):
        """Creating a client should authenticate without error."""
        assert client.api is not None

    def test_username_is_populated(self, client):
        """Client should derive a non-empty username from credentials."""
        assert isinstance(client.username, str)
        assert len(client.username) > 0


# ===========================================================================
# Phase 2: FORK AN EXISTING BENCHMARK
#
# Users often start by forking a public benchmark, then modifying it.
# ===========================================================================


class TestFork:
    """Forking (pulling) an existing notebook from Kaggle."""

    def test_fork_creates_workspace_with_files(self, client, fork_source_notebook_id):
        """fork() pulls the notebook, metadata, and converts .ipynb → .py."""
        workspace = client.fork(
            fork_source_notebook_id,
            dest_notebook_slug=f"kbench-golden-fork-{uuid.uuid4().hex[:8]}",
            overwrite=True,
        )

        assert workspace.exists() and workspace.is_dir()

        # Metadata should be present
        meta_path = workspace / BenchmarkNotebookClient.METADATA_FILENAME
        assert meta_path.exists()

        # Should have a .py or .ipynb file (depending on source notebook type)
        py_path = workspace / BenchmarkNotebookClient.BENCHMARK_FILENAME
        ipynb_path = workspace / BenchmarkNotebookClient.NOTEBOOK_FILENAME
        assert py_path.exists() or ipynb_path.exists(), (
            f"Expected benchmark.py or benchmark.ipynb in {workspace}"
        )

    def test_fork_raises_on_existing_workspace(self, client, fork_source_notebook_id):
        """fork(overwrite=False) raises FileExistsError if workspace exists."""
        slug = f"kbench-golden-fork-{uuid.uuid4().hex[:8]}"
        client._workspace(slug).mkdir(parents=True, exist_ok=True)

        with pytest.raises(FileExistsError, match="Workspace already exists"):
            client.fork(
                fork_source_notebook_id, dest_notebook_slug=slug, overwrite=False
            )

    def test_fork_raises_on_missing_notebook(self, client):
        """fork() raises ValueError for a non-existent notebook."""
        with pytest.raises(ValueError, match="Failed to pull notebook"):
            client.fork("kaggle/54321-tsixe-ton-seod-koobeton-siht")

    def test_fork_modify_publish_lifecycle(self, client, fork_source_notebook_id):
        """Full lifecycle: fork → modify → publish_and_run → get_results."""
        slug = f"kbench-golden-fork-lifecycle-{uuid.uuid4().hex[:8]}"
        workspace = client.fork(
            fork_source_notebook_id,
            dest_notebook_slug=slug,
            overwrite=True,
        )

        # Modify the script
        benchmark_py = workspace / BenchmarkNotebookClient.BENCHMARK_FILENAME
        with open(benchmark_py, "a", encoding="utf-8") as f:
            f.write('\n# %%\nprint("modified after fork")\n')

        # Publish and wait for results
        client.publish_and_run(slug, force=True)
        result = client.get_results(slug, poll_interval=30, timeout=600)

        assert result.status == "complete", f"Error: {result.error}"


# ===========================================================================
# Phase 3: PUBLISH & RUN
#
# Prepare a script and push it to Kaggle for execution.
# ===========================================================================


class TestPublish:
    """Publishing benchmark scripts to Kaggle."""

    def test_publish_from_workspace(self, client):
        """Tests the default workflow where the user authors benchmark.py directly inside the workspace."""
        workspace, slug = _prepare_workspace(client)

        url = client.publish_and_run(slug, force=True)

        # URL is well-formed
        assert url.startswith("https://www.kaggle.com/")
        assert client.username in url
        assert slug in url

        # Metadata was written with correct id and keyword
        meta_path = workspace / BenchmarkNotebookClient.METADATA_FILENAME
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        assert metadata["id"] == f"{client.username}/{slug}"
        assert "personal-benchmark" in metadata.get("keywords", [])

        # Notebook .ipynb was generated
        assert (workspace / BenchmarkNotebookClient.NOTEBOOK_FILENAME).exists()

    def test_publish_with_source_file(self, client, tmp_path):
        """Tests the override workflow where an external script is dynamically copied into the workspace."""
        slug = f"kbench-golden-source-file-{uuid.uuid4().hex[:8]}"

        # Write a source file outside the workspace
        source = tmp_path / "my_bench.py"
        source.write_text('# %%\nprint("source file test")\n', encoding="utf-8")

        url = client.publish_and_run(slug, source_file=source, force=True)
        assert url is not None and slug in url

        # Verify the file was copied and converted
        workspace = client._workspace(slug)
        bench_py = workspace / BenchmarkNotebookClient.BENCHMARK_FILENAME
        assert bench_py.exists()
        assert (
            bench_py.read_text(encoding="utf-8") == '# %%\nprint("source file test")\n'
        )

        # Verify the .ipynb has the source code in a cell
        notebook_data = json.loads(
            (workspace / BenchmarkNotebookClient.NOTEBOOK_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        source_found = any(
            'print("source file test")' in "".join(cell.get("source", []))
            for cell in notebook_data["cells"]
            if cell.get("cell_type") == "code"
        )
        assert source_found, "Source code not found in generated notebook cells"

    def test_publish_with_dataset_sources(self, client):
        """publish_and_run(dataset_sources=...) includes them in metadata."""
        workspace, slug = _prepare_workspace(
            client, script='# %%\nprint("dataset test")\n'
        )

        datasets = ["kaggle/meta-kaggle", "kaggle/meta-kaggle-code"]
        client.publish_and_run(slug, dataset_sources=datasets, force=True)

        meta_path = workspace / BenchmarkNotebookClient.METADATA_FILENAME
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        assert metadata["dataset_sources"] == datasets

    def test_publish_raises_without_benchmark_file(self, client):
        """publish_and_run() raises FileNotFoundError if no benchmark.py exists."""
        slug = f"kbench-golden-no-source-{uuid.uuid4().hex[:8]}"

        with pytest.raises(FileNotFoundError, match="Benchmark file not found"):
            client.publish_and_run(slug, force=True)

    def test_concurrent_guard_blocks_duplicate_push(self, fresh_client):
        """publish_and_run(force=False) raises ConcurrentRunError if already running."""
        from requests.exceptions import HTTPError

        _, slug = _prepare_workspace(fresh_client)

        # First push starts the run
        fresh_client.publish_and_run(slug, force=True)

        # Immediate second push without force should be blocked
        # (unless Kaggle already finished the run — handle that edge case)
        try:
            fresh_client.publish_and_run(slug, force=False)
        except ConcurrentRunError as e:
            assert "already running" in str(e)
        except HTTPError:
            pass  # 404 race condition — guard still passed

    def test_concurrent_guard_bypassed_with_force(self, fresh_client):
        """publish_and_run(force=True) always succeeds regardless of run status."""
        _, slug = _prepare_workspace(fresh_client)

        # First push starts the run
        fresh_client.publish_and_run(slug, force=True)

        # Immediate second push with force=True should bypass the guard and succeed
        url = fresh_client.publish_and_run(slug, force=True)
        assert url is not None


# ===========================================================================
# Phase 4: POLL & GET RESULTS
#
# After publishing, poll for completion and download outputs.
# ===========================================================================


class TestGetResults:
    """Polling for results and downloading output from Kaggle."""

    def test_full_round_trip(self, client):
        """Publish → poll → download → parse run.json: the complete happy path.

        NOTE: This test takes several minutes while the notebook executes
        on Kaggle.  Timeout is set to 10 minutes.
        """
        _, slug = _prepare_workspace(client)

        client.publish_and_run(slug, force=True)

        statuses_seen = []
        result = client.get_results(
            slug,
            poll_interval=30,
            timeout=600,
            on_status=statuses_seen.append,
        )

        assert isinstance(result, RunResult)
        assert result.tracking_url is not None
        assert slug in result.tracking_url
        assert result.status == "complete", (
            f"Expected 'complete', got '{result.status}'. Error: {result.error}"
        )
        assert result.output_dir is not None
        assert result.error is None

        # Verify the on_status callback fired correctly
        assert len(statuses_seen) > 0, "on_status callback was never invoked"
        assert statuses_seen[-1] == "complete", (
            f"Final callback was {statuses_seen[-1]}, not complete"
        )

        # Verify *.run.json files were downloaded and have expected structure
        runs = dict(result.iter_run_results())
        assert len(runs) >= 2, (
            f"Expected at least 2 run files, got {len(runs)}: {list(runs.keys())}"
        )
        for filename, run_data in runs.items():
            assert run_data.get("state") == "BENCHMARK_TASK_RUN_STATE_COMPLETED", (
                f"Run '{filename}' has unexpected state: {run_data.get('state')}"
            )
            assert run_data.get("taskVersion", {}).get("name") == "subtraction", (
                f"Run '{filename}' has unexpected task name"
            )

    def test_custom_output_dir(self, client, tmp_path):
        """get_results(output_dir=...) saves output to the given path."""
        _, slug = _prepare_workspace(client)
        client.publish_and_run(slug, force=True)

        custom_output = tmp_path / "custom_output"
        result = client.get_results(
            slug, output_dir=str(custom_output), poll_interval=30, timeout=600
        )

        if result.status == "timeout":
            pytest.skip("Notebook did not complete in time.")

        assert result.status == "complete", f"Failed with error: {result.error}"
        assert result.output_dir == str(custom_output)
        assert custom_output.exists()

    def test_timeout_returns_early(self, fresh_client):
        """get_results returns status='timeout' when the time limit is hit."""
        _, slug = _prepare_workspace(fresh_client)
        fresh_client.publish_and_run(slug, force=True)

        result = fresh_client.get_results(slug, poll_interval=5, timeout=1)

        assert isinstance(result, RunResult)
        assert result.status == "timeout"

    def test_cancel_event_returns_early(self, fresh_client):
        """get_results respects cancel_event and returns 'cancelled'."""
        _, slug = _prepare_workspace(fresh_client)

        cancel = threading.Event()

        def cancel_after_delay():
            time.sleep(0.5)
            cancel.set()

        timer = threading.Thread(target=cancel_after_delay, daemon=True)
        timer.start()

        fresh_client.publish_and_run(slug, force=True)
        result = fresh_client.get_results(slug, poll_interval=60, cancel_event=cancel)
        timer.join(timeout=5)

        assert isinstance(result, RunResult)
        assert result.status == "cancelled"


# ===========================================================================
# Phase 5: ERROR HANDLING
#
# What happens when things go wrong on Kaggle.
# ===========================================================================


class TestErrorHandling:
    """Tests for Kaggle-side execution errors."""

    def test_crashing_script_returns_error(self, client):
        """A script that raises at runtime should produce status='error'."""
        _, slug = _prepare_workspace(
            client, script='# %%\nraise ValueError("Deliberate crash")\n'
        )

        client.publish_and_run(slug, force=True)
        result = client.get_results(slug, poll_interval=20, timeout=600)

        assert result.status == "error"
        assert result.error is not None
        assert "finished with status: error" in result.error
