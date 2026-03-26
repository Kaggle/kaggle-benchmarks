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
import os
import shutil
import tarfile
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from kaggle_benchmarks.kaggle_client import utils as kaggle_utils

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

    def iter_run_results(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yields (filename, parsed_data) for all *run.json files in the output directory."""
        if not self.output_dir or not (output_path := Path(self.output_dir)).exists():
            return

        for run_file in output_path.glob("*run.json"):
            yield run_file.name, json.loads(run_file.read_text(encoding="utf-8"))


# =============================================================================
# Module-level helper functions (authentication)
# =============================================================================

_AUTH_ERROR_INSTRUCTIONS = (
    "To authenticate, use one of the following methods (in priority order):\n\n"
    "  Option 1: Bearer token (recommended)\n"
    "    export KAGGLE_API_TOKEN=<your-token>\n"
    "    Or save your token to ~/.kaggle/access_token\n\n"
    "  Option 2: Basic auth via environment variables\n"
    "    export KAGGLE_USERNAME=<your-username>\n"
    "    export KAGGLE_KEY=<your-api-key>\n\n"
    "  Option 3: Basic auth via credentials file\n"
    "    Save kaggle.json to ~/.kaggle/kaggle.json\n\n"
    "Get your API token at: https://www.kaggle.com/settings\n"
    "Docs: https://github.com/Kaggle/kagglehub/tree/main#authenticate"
)


def _authenticate() -> tuple[Any, str]:
    """Authenticate with the Kaggle API and resolve the username.

    Credential resolution relies on the kagglesdk for token introspection
    and kagglehub for standard Basic auth/json fallbacks.

    Returns:
        A tuple of (KaggleClient, username).

    Raises:
        KaggleAuthError: If the SDK is missing, no credentials are found,
            or the username cannot be determined.
    """
    try:
        from kagglesdk.kaggle_client import KaggleClient
        from kagglesdk.kaggle_env import get_access_token_from_env, get_env
        from kagglesdk.security.types.oauth_service import IntrospectTokenRequest
    except ImportError as e:
        raise KaggleAuthError(
            "The 'kagglesdk' package is required. Install with:\n"
            "  pip install kaggle-benchmarks[kaggle-client]"
        ) from e

    try:
        client = KaggleClient(env=get_env())
    except Exception as e:
        raise KaggleAuthError(
            f"Kaggle authentication failed: {e}\n\n{_AUTH_ERROR_INSTRUCTIONS}"
        ) from e

    def get_token_user() -> str | None:
        if api_token := get_access_token_from_env()[0]:
            try:
                req = IntrospectTokenRequest()
                req.token = api_token
                resp = client.security.oauth_client.introspect_token(req)
                return resp.username if resp.active else None
            except Exception:
                pass
        return None

    def get_hub_user() -> str | None:
        try:
            import kagglehub

            return (
                creds.username
                if (creds := kagglehub.config.get_kaggle_credentials())
                else None
            )
        except Exception:
            return None

    username = client.username or get_token_user() or get_hub_user()

    if not username:
        raise KaggleAuthError(
            f"No Kaggle credentials found or invalid.\n\n{_AUTH_ERROR_INSTRUCTIONS}"
        )

    return client, username


# =============================================================================
# Client class
# =============================================================================


class BenchmarkNotebookClient:
    """Client for the Kaggle benchmark notebook workflow.

    Wraps the Kaggle SDK (kagglesdk) to handle publishing, running,
    and retrieving results from benchmark notebooks (tagged with
    'personal-benchmark' keyword).
    """

    BENCHMARK_FILENAME = "benchmark.py"
    NOTEBOOK_FILENAME = "benchmark.ipynb"
    METADATA_FILENAME = "kernel-metadata.json"
    OUTPUT_DIRNAME = "output"

    # Retry config for initial 404s from get_kernel_session_status after save_kernel.
    _STATUS_RETRIES = 5
    _STATUS_RETRY_WAIT = 10  # seconds

    # =========================================================================
    # Authentication & Identity
    # =========================================================================

    def __init__(
        self,
        base_dir: str | Path = ".",
    ):
        """Initialize the BenchmarkNotebookClient.

        Args:
            base_dir: Parent directory for benchmark workspaces.
        """
        self.api, self.username = _authenticate()
        self.base_dir = Path(base_dir)

    # =========================================================================
    # Forking
    # =========================================================================

    def fork(
        self,
        source_notebook_id: str,
        dest_notebook_slug: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Pull an existing benchmark from Kaggle as a starting point.

        Workspace: <base_dir>/<dest_notebook_slug>/

        Downloads via get_kernel() and converts:
        1. Pulls the .ipynb source and kernel metadata
        2. Writes the source as benchmark.ipynb
        3. Converts benchmark.ipynb -> benchmark.py with # %% cell delimiters
        4. Reconstructs kernel-metadata.json from API response

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

        # Pull notebook source and metadata from Kaggle via get_kernel()
        from kagglesdk.kernels.types.kernels_api_service import (
            ApiGetKernelRequest,
        )
        from requests.exceptions import HTTPError

        try:
            req = ApiGetKernelRequest()
            req.user_name, req.kernel_slug = source_notebook_id.split("/")
            response = self.api.kernels.kernels_api_client.get_kernel(req)
        except HTTPError as e:
            raise ValueError(
                f"Failed to pull notebook '{source_notebook_id}'. "
                "Ensure the notebook exists, is public (or you have access), "
                "and you have accepted any necessary competition rules."
            ) from e

        # Write the notebook source to benchmark.ipynb
        # response.blob.source is exactly the raw .ipynb JSON for notebook-type kernels
        if response.blob and response.blob.source:
            ipynb_path = workspace / self.NOTEBOOK_FILENAME
            ipynb_path.write_text(response.blob.source, encoding="utf-8")

            # Convert .ipynb to .py with # %% cell delimiters
            py_path = workspace / self.BENCHMARK_FILENAME
            kaggle_utils.convert_ipynb_to_py(ipynb_path, py_path)
        else:
            logger.warning(
                "No source found after pulling '%s'. "
                "The notebook may be a script notebook.",
                source_notebook_id,
            )

        # Reconstruct kernel-metadata.json from response.metadata
        if response.metadata:
            metadata = kaggle_utils.parse_remote_metadata(
                meta=response.metadata,
                default_id=source_notebook_id,
                default_slug=dest_notebook_slug,
            )
            meta_path = workspace / self.METADATA_FILENAME
            meta_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")

        return workspace

    # =========================================================================
    # Publish & Run
    # =========================================================================

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
            dataset_sources: Optional list of Kaggle dataset slugs (e.g., ``["owner/dataset-name"]``)
                to mount at ``/kaggle/input/``.
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
        metadata = kaggle_utils.build_local_metadata(
            workspace_dir=workspace,
            notebook_slug=notebook_slug,
            username=self.username,
            dataset_sources=dataset_sources,
        )

        # Convert benchmark.py (.py with # %% delimiters) to .ipynb
        notebook_path = workspace / self.NOTEBOOK_FILENAME
        kaggle_utils.convert_py_to_ipynb(benchmark_py, notebook_path)

        # Concurrent run guard
        notebook_id = self._notebook_id(notebook_slug)
        if not force:
            try:
                status = self._get_kernel_session_status(notebook_id)
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

        # Write metadata to disk (for local inspection and fork reuse)
        meta_path = workspace / self.METADATA_FILENAME
        meta_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")

        # Read the generated .ipynb and push inline via save_kernel
        notebook_content = notebook_path.read_text(encoding="utf-8")
        req = self._build_save_request(
            notebook_id=self._notebook_id(notebook_slug),
            notebook_content=notebook_content,
            metadata=metadata,
        )

        response = self.api.kernels.kernels_api_client.save_kernel(req)
        if response.error:
            raise RuntimeError(f"Kaggle push failed: {response.error}")

        return self._tracking_url(notebook_slug)

    def _build_save_request(
        self,
        notebook_id: str,
        notebook_content: str,
        metadata: dict[str, Any],
    ):
        """Build the API request to save/push a notebook."""
        from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest

        req = ApiSaveKernelRequest()

        # Special required fields
        req.slug = notebook_id
        req.new_title = metadata.get("title", notebook_id.split("/")[-1])
        req.text = notebook_content

        # Map the remaining configured attributes dynamically
        for json_key, (api_key, _) in kaggle_utils.KAGGLE_METADATA_MAP.items():
            if json_key in metadata:
                setattr(req, api_key, metadata[json_key])

        return req

    # =========================================================================
    # Polling & Results
    # =========================================================================

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

        Polls get_kernel_session_status() until the notebook completes (or fails),
        then downloads output via download_kernel_output().

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

    def _wait_for_notebook_creation(self, notebook_id: str) -> str | None:
        """Wait for Kaggle to index a newly pushed notebook.

        Returns:
            The initial status string, or None if retries exceeded.
        """
        from requests.exceptions import HTTPError

        for attempt in range(self._STATUS_RETRIES):
            try:
                return self._get_kernel_session_status(notebook_id)
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

            status_str = self._get_kernel_session_status(notebook_id)

        # Report final status
        if on_status is not None and status_str != last_reported_status:
            on_status(status_str)

        return status_str

    def _download_notebook_output(
        self, notebook_id: str, output_path: Path, clear_output: bool
    ) -> None:
        """Clear existing output (if requested) and download new output from Kaggle."""
        from kagglesdk.kernels.types.kernels_api_service import (
            ApiDownloadKernelOutputRequest,
        )

        if clear_output and output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Build request (no file_path -> downloads all output as archive)
        req = ApiDownloadKernelOutputRequest()
        req.owner_slug, req.kernel_slug = notebook_id.split("/")

        # download_kernel_output returns a streamed requests.Response
        response = self.api.kernels.kernels_api_client.download_kernel_output(req)

        # Save the archive to a temp file, then extract
        with tempfile.NamedTemporaryFile(delete=False, suffix=".archive") as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            archive_path = tmp.name

        try:
            if tarfile.is_tarfile(archive_path):
                with tarfile.open(archive_path) as tf:
                    tf.extractall(output_path)
            elif zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path, "r") as zf:
                    zf.extractall(output_path)
            else:
                raise RuntimeError("Unexpected archive format from Kaggle API")
        finally:
            os.remove(archive_path)

    def _get_kernel_session_status(self, notebook_id: str):
        """Get the session status for a notebook via kagglesdk.

        Args:
            notebook_id: Full notebook ID (format: "username/slug").

        Returns:
            The normalized status string.
        """
        from kagglesdk.kernels.types.kernels_api_service import (
            ApiGetKernelSessionStatusRequest,
        )

        user_name, kernel_slug = notebook_id.split("/")
        req = ApiGetKernelSessionStatusRequest()
        req.user_name = user_name
        req.kernel_slug = kernel_slug
        response = self.api.kernels.kernels_api_client.get_kernel_session_status(req)
        return kaggle_utils.normalize_status(response)

    # =========================================================================
    # Internal Utilities
    # =========================================================================

    def _workspace(self, notebook_slug: str) -> Path:
        """Return the workspace directory for a notebook slug."""
        return self.base_dir / notebook_slug

    def _notebook_id(self, notebook_slug: str) -> str:
        """Return the full notebook ID (username/slug)."""
        return f"{self.username}/{notebook_slug}"

    def _tracking_url(self, notebook_slug: str) -> str:
        """Return the Kaggle tracking URL for a notebook."""
        return f"https://www.kaggle.com/{self._notebook_id(notebook_slug)}"
