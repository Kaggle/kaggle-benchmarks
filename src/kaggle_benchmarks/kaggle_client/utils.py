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
import re
import warnings
from pathlib import Path
from typing import Any


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
    jupytext.write(notebook, ipynb_path)


def convert_ipynb_to_py(ipynb_path: str | Path, py_path: str | Path) -> None:
    """Converts a Jupyter Notebook to a Python script in percent format."""
    import jupytext

    notebook = jupytext.read(ipynb_path)
    jupytext.write(notebook, py_path, fmt="py:percent")


def resolve_metadata(
    workspace_dir: Path | str,
    notebook_slug: str,
    username: str,
    dataset_sources: list[str] | None = None,
    is_private: bool = True,
    title: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Assembles the kernel-metadata.json payload for Kaggle.

    Loads existing metadata if present, applies overrides, and ensures
    mandatory fields/tags are set.
    """
    meta_path = Path(workspace_dir) / "kernel-metadata.json"

    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        # Default metadata template for new notebooks.
        # See: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md
        metadata = {
            "language": "python",
            "kernel_type": "notebook",
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": True,  # Required for LLM API calls
            "dataset_sources": [],  # Kaggle datasets to mount at /kaggle/input/
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
            "keywords": [],  # Tags; we ensure "personal-benchmark" is added
            "docker_image": None,  # Custom docker image (future use)
            "machine_shape": "None",  # "None" = default, "gpu", "tpu" etc.
        }

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
    keywords = metadata.setdefault("keywords", [])
    if "personal-benchmark" not in keywords:
        keywords.append("personal-benchmark")

    return metadata
