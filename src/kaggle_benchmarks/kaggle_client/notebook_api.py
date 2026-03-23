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

"""Core client for the Kaggle benchmark notebook workflow."""

import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from kaggle_benchmarks.kaggle_client.utils import (
    convert_ipynb_to_py,
    convert_py_to_ipynb,
    resolve_metadata,
)

logger = logging.getLogger(__name__)


class KaggleAuthError(RuntimeError):
    """Raised when Kaggle authentication fails or credentials are invalid."""


class ConcurrentRunError(RuntimeError):
    """Raised when a notebook is already running and force=False."""


@dataclass
class RunResult:
    """Result of a benchmark notebook execution.

    Attributes:
        status: One of "queued", "running", "complete", "error",
            "cancelled", or "timeout".
        output_dir: String path to the output directory (not Path,
            for JSON serialization). None if no output available.
        tracking_url: URL to the notebook on Kaggle.
        error: Error message (if any).
    """

    status: str
    output_dir: str | None
    tracking_url: str | None
    error: str | None = None

    def iter_runs(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yields (filename, parsed_data) for all *run.json files in the output directory."""
        if not self.output_dir:
            return

        output_path = Path(self.output_dir)
        if not output_path.exists():
            return

        for run_file in output_path.iterdir():
            if run_file.name.endswith("run.json"):
                yield run_file.name, json.loads(run_file.read_text(encoding="utf-8"))


class BenchmarkNotebookClient:
    """Client for the Kaggle benchmark notebook workflow.

    Wraps the Kaggle API (KaggleApi from kaggle-python) to handle
    publishing, running, and retrieving results from benchmark
    notebooks (tagged with 'personal-benchmark' keyword).
    """

    BENCHMARK_FILENAME = "benchmark.py"
    NOTEBOOK_FILENAME = "benchmark.ipynb"
    METADATA_FILENAME = "kernel-metadata.json"
    OUTPUT_DIRNAME = "output"

    AUTH_ERROR_INSTRUCTIONS = (
        "To authenticate, use one of the following methods:\n\n"
        "  Option 1: Environment variable\n"
        "    export KAGGLE_API_TOKEN=<your-token>\n\n"
        "  Option 2: API token file\n"
        "    Save your token to ~/.kaggle/access_token\n\n"
        "  Option 3: Legacy credentials file\n"
        "    Save kaggle.json to ~/.kaggle/kaggle.json\n\n"
        "Get your API token at: https://www.kaggle.com/settings\n"
        "Docs: https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication"
    )

    # Retry config for initial 404s from kernels_status after kernels_push.
    # See: kaggle_api_example.py pull_notebook_output()
    _STATUS_RETRIES = 5
    _STATUS_RETRY_WAIT = 10  # seconds

    def __init__(
        self,
        base_dir: str | Path = ".",
    ):
        """Initialize the BenchmarkNotebookClient.

        Args:
            base_dir: Parent directory for benchmark workspaces.
        """
        self.api = self._authenticate()
        self.username = self.validate_and_get_username()
        self.base_dir = Path(base_dir)

    @staticmethod
    def _authenticate():
        """Authenticate with the Kaggle API.

        Returns:
            Authenticated KaggleApi instance.

        Raises:
            KaggleAuthError: If the kaggle package is not installed or
                authentication fails.
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as e:
            raise KaggleAuthError(
                "The 'kaggle' package is required. Install with:\n"
                "  pip install kaggle-benchmarks[kaggle-client]"
            ) from e

        api = KaggleApi()
        try:
            api.authenticate()
        except OSError as e:
            raise KaggleAuthError(
                f"Kaggle authentication failed: {e}\n\n"
                f"{BenchmarkNotebookClient.AUTH_ERROR_INSTRUCTIONS}"
            ) from e
        return api

    def _workspace(self, notebook_slug: str) -> Path:
        """Return the workspace directory for a notebook slug."""
        return self.base_dir / notebook_slug

    def _notebook_id(self, notebook_slug: str) -> str:
        """Return the full notebook ID (username/slug)."""
        return f"{self.username}/{notebook_slug}"

    def _tracking_url(self, notebook_slug: str) -> str:
        """Return the Kaggle tracking URL for a notebook."""
        return f"https://www.kaggle.com/{self._notebook_id(notebook_slug)}"

    @staticmethod
    def _normalize_status(status: object) -> str:
        """Normalize a Kaggle notebook status to a lowercase string.

        The Kaggle API may return a KernelWorkerStatus enum
        (e.g. "KernelWorkerStatus.complete") or a plain string.
        This method normalizes both to a simple lowercase string
        like "complete".
        """
        # Handle response wrappers with a .status attribute
        status_raw = getattr(status, "status", status)
        status_str = str(status_raw).lower()
        # Strip enum class prefix (e.g. "kernelworkerstatus.complete" -> "complete")
        if "." in status_str:
            status_str = status_str.split(".")[-1]
        return status_str

    # --- Primary Operations ---

    def publish_and_run(
        self,
        notebook_slug: str,
        source_file: str | Path | None = None,
        dataset_sources: list[str] | None = None,
        force: bool = False,
    ) -> str:
        """Convert .py -> .ipynb, push to Kaggle, and trigger execution.

        Workspace: <base_dir>/<notebook_slug>/

        A notebook acts as the execution vehicle for your benchmark tasks. While you
        can define and run multiple tasks within a single notebook, the Kaggle
        leaderboard currently requires a single "main" task per notebook to be saved
        and evaluated. You can use the `%choose <task_name>` magic command at the end
        of your notebook to select which task's results to publish.

        For more details on task selection, see:
        https://github.com/Kaggle/kaggle-benchmarks/blob/ci/quick_start.md#82-using-choose-to-select-a-notebooks-task

        Args:
            notebook_slug: Short notebook name (e.g., 'my-benchmark').
            source_file: Optional path to a .py file to copy into workspace. It will replace benchmark.py.
            dataset_sources: Kaggle dataset slugs to mount at /kaggle/input/.
            force: If True, push even if a previous run is in progress.

        Returns:
            Tracking URL for the running notebook.

        Raises:
            FileNotFoundError: If benchmark.py not found in workspace.
            ConcurrentRunError: If a concurrent run is detected and force=False.
        """
        from requests.exceptions import HTTPError

        workspace = self._workspace(notebook_slug)
        workspace.mkdir(parents=True, exist_ok=True)
        benchmark_py = workspace / self.BENCHMARK_FILENAME

        # Copy source_file into workspace (if provided)
        if source_file is not None:
            source = Path(source_file)
            if not source.exists():
                raise FileNotFoundError(f"Source file not found: {source}")
            shutil.copy2(source, benchmark_py)

        if not benchmark_py.exists():
            raise FileNotFoundError(
                f"Benchmark file not found: {benchmark_py}. "
                f"Create it or pass source_file to copy from."
            )

        # Resolve metadata (load existing kernel-metadata.json or generate new)
        # 'personal-benchmark' keyword is ensured by resolve_metadata
        metadata = resolve_metadata(
            workspace_dir=workspace,
            notebook_slug=notebook_slug,
            username=self.username,
            dataset_sources=dataset_sources,
        )

        # Convert benchmark.py (.py with # %% delimiters) to .ipynb
        notebook_path = workspace / self.NOTEBOOK_FILENAME
        convert_py_to_ipynb(benchmark_py, notebook_path)

        # Concurrent run guard
        notebook_id = self._notebook_id(notebook_slug)
        if not force:
            try:
                raw_status = self.api.kernels_status(notebook_id)
                status = self._normalize_status(raw_status)
                if status in ("queued", "running"):
                    raise ConcurrentRunError(
                        f"Notebook '{notebook_id}' is already running "
                        f"(status: {status}). "
                        "Use force=True to push a new version anyway."
                    )
            except HTTPError as e:
                if e.response.status_code == 404:
                    pass  # No existing notebook — safe to push
                else:
                    raise

        # Write metadata and push via api.kernels_push()
        meta_path = workspace / self.METADATA_FILENAME
        meta_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")

        self.api.kernels_push(str(workspace))

        return self._tracking_url(notebook_slug)

    def _wait_for_notebook_creation(self, notebook_id: str) -> str | None:
        """Wait for Kaggle to index a newly pushed notebook.

        Returns:
            The initial status string, or None if retries exceeded.
        """
        from requests.exceptions import HTTPError

        for attempt in range(self._STATUS_RETRIES):
            try:
                raw_status = self.api.kernels_status(notebook_id)
                return self._normalize_status(raw_status)
            except HTTPError as e:
                if e.response.status_code == 404:
                    logger.info(
                        "Notebook not found yet (attempt %d/%d), waiting...",
                        attempt + 1,
                        self._STATUS_RETRIES,
                    )
                    time.sleep(self._STATUS_RETRY_WAIT)
                else:
                    raise
        return None

    def _poll_notebook_status(
        self,
        notebook_id: str,
        initial_status: str,
        poll_interval: int,
        timeout: int | None,
        cancel_event: threading.Event | None,
        on_status: Callable[[str], None] | None,
    ) -> str:
        """Poll notebook status until complete, error, cancelled, or timeout."""
        status_str = initial_status
        last_reported_status = None
        start_time = time.monotonic()

        while status_str not in ("complete", "error", "cancelled"):
            if cancel_event is not None and cancel_event.is_set():
                return "cancelled"

            if timeout is not None and (time.monotonic() - start_time) >= timeout:
                return "timeout"

            if on_status is not None and status_str != last_reported_status:
                on_status(status_str)
                last_reported_status = status_str

            # Wait with cancel_event support for early exit
            if cancel_event is not None:
                cancel_event.wait(poll_interval)
                if cancel_event.is_set():
                    return "cancelled"
            else:
                time.sleep(poll_interval)

            raw_status = self.api.kernels_status(notebook_id)
            status_str = self._normalize_status(raw_status)

        # Report final status
        if on_status is not None and status_str != last_reported_status:
            on_status(status_str)

        return status_str

    def _download_notebook_output(
        self, notebook_id: str, output_path: Path, clear_output: bool
    ) -> None:
        """Clear existing output (if requested) and download new output from Kaggle."""
        if clear_output and output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.api.kernels_output(kernel=notebook_id, path=str(output_path), force=True)

    def get_results(
        self,
        notebook_slug: str,
        output_dir: str | None = None,
        poll_interval: int = 60,
        timeout: int | None = None,
        cancel_event: threading.Event | None = None,
        on_status: Callable[[str], None] | None = None,
        clear_output: bool = True,
    ) -> RunResult:
        """Poll execution status and download output files.

        Workspace: <base_dir>/<notebook_slug>/

        Polls api.kernels_status() until the notebook completes (or fails),
        then downloads output via api.kernels_output().

        Neither timeout nor cancel_event stops the Kaggle run itself —
        the notebook continues executing on Kaggle. They only stop
        the local polling.

        Args:
            notebook_slug: Short notebook name (e.g., 'my-benchmark').
            output_dir: Where to save output files (defaults to
                <base_dir>/<notebook_slug>/output/).
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait before returning
                RunResult(status="timeout"). None means wait indefinitely.
            cancel_event: A threading.Event; when set, the poll loop exits
                with RunResult(status="cancelled").
            on_status: Callback invoked when the notebook status changes.
            clear_output: If True (default), clear the output directory
                before downloading new output. If False, download into
                the existing directory without clearing.

        Returns:
            RunResult with status, output path, and parsed run.json data.
        """
        notebook_id = self._notebook_id(notebook_slug)
        workspace = self._workspace(notebook_slug)
        tracking_url = self._tracking_url(notebook_slug)

        if output_dir is None:
            output_path = workspace / self.OUTPUT_DIRNAME
        else:
            output_path = Path(output_dir)

        status_str = self._wait_for_notebook_creation(notebook_id)
        if status_str is None:
            return RunResult(
                status="error",
                output_dir=None,
                tracking_url=tracking_url,
                error=(
                    f"Failed to find notebook '{notebook_id}' "
                    f"after {self._STATUS_RETRIES} retries."
                ),
            )

        final_status = self._poll_notebook_status(
            notebook_id=notebook_id,
            initial_status=status_str,
            poll_interval=poll_interval,
            timeout=timeout,
            cancel_event=cancel_event,
            on_status=on_status,
        )

        if final_status != "complete":
            error_msg = None
            if final_status not in ("timeout", "cancelled"):
                error_msg = f"Notebook execution finished with status: {final_status}"

            return RunResult(
                status=final_status,
                output_dir=None,
                tracking_url=tracking_url,
                error=error_msg,
            )

        self._download_notebook_output(notebook_id, output_path, clear_output)

        return RunResult(
            status="complete",
            output_dir=str(output_path),
            tracking_url=tracking_url,
        )

    def fork(
        self,
        source_notebook_id: str,
        dest_notebook_slug: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Pull an existing benchmark from Kaggle as a starting point.

        Workspace: <base_dir>/<dest_notebook_slug>/

        Downloads via api.kernels_pull() and converts:
        1. Pulls the .ipynb and kernel-metadata.json
        2. Renames the pulled .ipynb to benchmark.ipynb to prevent Kaggle from pushing multiple notebooks during publish.
        3. Converts benchmark.ipynb -> benchmark.py with # %% cell delimiters

        The kernel-metadata.json is preserved so that publish_and_run()
        can reuse it.

        Args:
            source_notebook_id: Full Kaggle notebook path including owner
                (e.g., 'alice/riddle-benchmark').
            dest_notebook_slug: Local name for the benchmark directory.
                Defaults to the basename of source_notebook_id.
            overwrite: If True, replace the existing workspace directory.
                If False (default), raises FileExistsError.

        Returns:
            Path to the workspace directory containing the .py file.
        """
        if dest_notebook_slug is None:
            dest_notebook_slug = source_notebook_id.split("/")[-1]

        workspace = self._workspace(dest_notebook_slug)

        if workspace.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Workspace already exists: {workspace}. "
                    "Use overwrite=True to replace it."
                )
            shutil.rmtree(workspace)

        workspace.mkdir(parents=True, exist_ok=True)

        # Pull notebook and metadata from Kaggle
        from requests.exceptions import HTTPError

        try:
            self.api.kernels_pull(
                source_notebook_id, path=str(workspace), metadata=True
            )
        except HTTPError as e:
            raise ValueError(
                f"Failed to pull notebook '{source_notebook_id}'. "
                "Ensure the notebook exists, is public (or you have access), "
                "and you have accepted any necessary competition rules."
            ) from e

        # Convert .ipynb to .py with # %% cell delimiters
        ipynb_files = list(workspace.glob("*.ipynb"))
        if ipynb_files:
            ipynb_path = ipynb_files[0]
            py_path = workspace / self.BENCHMARK_FILENAME
            convert_ipynb_to_py(ipynb_path, py_path)

            std_ipynb_path = workspace / self.NOTEBOOK_FILENAME
            if ipynb_path != std_ipynb_path:
                ipynb_path.rename(std_ipynb_path)
        else:
            logger.warning(
                "No .ipynb file found after pulling '%s'. "
                "The notebook may be a script notebook.",
                source_notebook_id,
            )

        return workspace

    def validate_and_get_username(self) -> str:
        """Validate Kaggle credentials and return the authenticated username.

        Calls the Kaggle API to verify that the stored credentials
        are valid and retrieves the authenticated username.

        Returns:
            The authenticated Kaggle username.

        Raises:
            KaggleAuthError: If credentials are missing or invalid,
                with a message guiding the user to set up
                authentication.
        """
        try:
            username = self.api.get_config_value("username")
            if not username:
                raise ValueError("Kaggle username is empty")
            return username
        except Exception as e:
            raise KaggleAuthError(
                f"Kaggle credentials are invalid or missing: {e}\n\n"
                f"{BenchmarkNotebookClient.AUTH_ERROR_INSTRUCTIONS}"
            ) from e
