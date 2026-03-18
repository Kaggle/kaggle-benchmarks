import json
import re
import warnings
from pathlib import Path


def convert_py_to_ipynb(py_path: str | Path, ipynb_path: str | Path) -> None:
    """Converts a Python script in percent format to a Jupyter Notebook.

    Warns if no '# %%' cell delimiters are found.
    """
    import jupytext

    py_path = Path(py_path)
    ipynb_path = Path(ipynb_path)

    with open(py_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check for '# %%' delimiters
    if not re.search(r"^#\s*%%", content, re.MULTILINE):
        warnings.warn(
            f"{py_path.name} has no '# %%' cell delimiters. "
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
) -> dict:
    """Assembles the kernel-metadata.json payload for Kaggle.

    Loads existing metadata if present, applies overrides, and ensures
    mandatory fields/tags are set.
    """
    workspace_dir = Path(workspace_dir)
    meta_path = workspace_dir / "kernel-metadata.json"

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        # Default metadata template for new notebooks.
        # See: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md
        metadata = {
            "id": "",  # Full kernel ID: "username/slug"
            "title": "",  # Display title (defaults to slug)
            "code_file": "benchmark.ipynb",  # Fixed filename; see convention below
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,  # Private by default for benchmarks
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": True,  # Required for LLM API calls
            "dataset_sources": [],  # Kaggle datasets to mount at /kaggle/input/
            "competition_sources": [],
            "kernel_sources": [],  # Other kernels whose output to mount
            "model_sources": [],  # Kaggle Models to attach
            "keywords": [],  # Tags; we ensure "personal-benchmark" is added
            "docker_image": None,  # Custom docker image (future use)
            "machine_shape": "None",  # "None" = default, "gpu", "tpu" etc.
        }

    # --- Mandatory overrides ---
    # The Kaggle API uses "id" as the unique kernel identifier in the format
    # "username/slug". The slug is the URL-friendly name (e.g., "my-benchmark").
    metadata["id"] = f"{username}/{notebook_slug}"

    # title defaults to slug but can be overridden for a prettier display name.
    metadata["title"] = title if title is not None else notebook_slug

    # Clear id_no so the API uses "id" (id_no takes precedence if both are set).
    metadata["id_no"] = None

    # Fixed filename: users write benchmark.py, we convert to benchmark.ipynb.
    metadata["code_file"] = "benchmark.ipynb"

    if dataset_sources is not None:
        metadata["dataset_sources"] = dataset_sources

    metadata["is_private"] = is_private

    # Apply additional overrides (e.g., enable_gpu, enable_tpu).
    for k, v in kwargs.items():
        if v is not None:
            metadata[k] = v

    # Ensure "personal-benchmark" tag is present.
    keywords = metadata.get("keywords")
    if keywords is None:
        keywords = []
        metadata["keywords"] = keywords

    if "personal-benchmark" not in keywords:
        keywords.append("personal-benchmark")

    return metadata
