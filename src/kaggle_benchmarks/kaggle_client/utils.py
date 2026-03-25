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

"""Utility functions for the Kaggle benchmark client."""

import json
import re
import warnings
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# File format conversion
# ---------------------------------------------------------------------------


def convert_py_to_ipynb(py_path: str | Path, ipynb_path: str | Path) -> None:
    """Converts a Python script in percent format to a Jupyter Notebook.

    Warns if no '# %%' cell delimiters are found.
    """
    import jupytext

    py_path = Path(py_path)
    content = py_path.read_text(encoding="utf-8")

    # Check for '# %%' delimiters
    if not re.search(r"^#\s*%%", content, re.MULTILINE):
        warnings.warn(
            f"'{py_path.name}' has no '# %%' cell delimiters. "
            "The entire file will be uploaded as a single notebook cell.",
            UserWarning,
            stacklevel=2,
        )

    notebook = jupytext.reads(content, fmt="py:percent")

    # Kaggle's notebook runner (papermill) requires a kernelspec to evaluate cells.
    notebook.metadata.setdefault(
        "kernelspec",
        {"display_name": "Python 3", "language": "python", "name": "python3"},
    )

    jupytext.write(notebook, ipynb_path)


def convert_ipynb_to_py(ipynb_path: str | Path, py_path: str | Path) -> None:
    """Converts a Jupyter Notebook to a Python script in percent format."""
    import jupytext

    notebook = jupytext.read(ipynb_path)
    jupytext.write(notebook, py_path, fmt="py:percent")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


# Maps kernel-metadata.json keys to (ApiSaveKernelRequest attribute, default value)
KAGGLE_METADATA_MAP = {
    "language": ("language", "python"),
    "kernel_type": ("kernel_type", "notebook"),
    "is_private": ("is_private", True),
    "enable_gpu": ("enable_gpu", False),
    "enable_tpu": ("enable_tpu", False),
    "enable_internet": ("enable_internet", True),
    "dataset_sources": ("dataset_data_sources", []),
    "competition_sources": ("competition_data_sources", []),
    "kernel_sources": ("kernel_data_sources", []),
    "model_sources": ("model_data_sources", []),
    "keywords": ("category_ids", []),
    "docker_image": ("docker_image", None),
    "machine_shape": ("machine_shape", "None"),
}


def build_local_metadata(
    workspace_dir: Path | str,
    notebook_slug: str,
    username: str,
    dataset_sources: list[str] | None = None,
    is_private: bool = True,
    title: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Builds the local `kernel-metadata.json` dictionary for pushing to Kaggle.

    Loads existing workspace metadata, merges runtime CLI overrides, and enforces
    required benchmark schemas (such as the 'personal-benchmark' tag).
    """
    meta_path = Path(workspace_dir) / "kernel-metadata.json"

    metadata = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )
    for json_key, (_, default_val) in KAGGLE_METADATA_MAP.items():
        metadata.setdefault(json_key, default_val)

    # --- Mandatory overrides ---
    overrides = {
        # The Kaggle API uses "id" as the unique kernel identifier in the format
        # "username/slug". The slug is the URL-friendly name (e.g., "my-benchmark").
        "id": f"{username}/{notebook_slug}",
        # title defaults to slug but can be overridden for a prettier display name.
        "title": title or notebook_slug,
        # Clear id_no so the API uses "id" (id_no takes precedence if both are set).
        "id_no": None,
        # Fixed filename: users write benchmark.py, we convert to benchmark.ipynb.
        "code_file": "benchmark.ipynb",
        # Private by default for benchmarks
        "is_private": is_private,
        # Apply additional overrides (e.g., enable_gpu, enable_tpu).
        **{k: v for k, v in kwargs.items() if v is not None},
    }

    if dataset_sources is not None:
        overrides["dataset_sources"] = dataset_sources

    metadata.update(overrides)

    # Ensure "personal-benchmark" tag is present.
    if "personal-benchmark" not in (keywords := metadata.setdefault("keywords", [])):
        keywords.append("personal-benchmark")

    return metadata


def parse_remote_metadata(
    meta: Any, default_id: str, default_slug: str
) -> dict[str, Any]:
    """Converts a Kaggle API `Kernel` object into a local `kernel-metadata.json` dictionary.

    Translates SDK-specific attribute names (like `dataset_data_sources`) back into
    standard Kaggle JSON fields to allow for local editing on disk.
    """
    meta_dict = {
        "id": getattr(meta, "ref", default_id),
        "title": getattr(meta, "title", default_slug),
    }

    for json_key, (api_key, default_val) in KAGGLE_METADATA_MAP.items():
        val = getattr(meta, api_key, None)
        # Handle repeated enum/list types cleanly by defaulting to []
        if isinstance(default_val, list):
            meta_dict[json_key] = list(val or [])
        else:
            meta_dict[json_key] = val if val is not None else default_val

    return meta_dict


# ---------------------------------------------------------------------------
# Status normalization
# ---------------------------------------------------------------------------


def normalize_status(status: object) -> str:
    """Normalize a Kaggle notebook status to a lowercase string.

    The Kaggle API may return a KernelWorkerStatus enum
    (e.g. "KernelWorkerStatus.complete") or a plain string.
    This method normalizes both to a simple lowercase string
    like "complete".
    """
    status_raw = getattr(status, "status", status)
    return str(status_raw).lower().split(".")[-1]
