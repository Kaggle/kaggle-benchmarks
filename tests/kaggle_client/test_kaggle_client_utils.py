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
from unittest.mock import MagicMock

import pytest

from kaggle_benchmarks.kaggle_client import utils as kaggle_utils

# ---------------------------------------------------------------------------
# File format conversion
# ---------------------------------------------------------------------------


def test_convert_py_to_ipynb(tmp_path):
    py_file = tmp_path / "benchmark.py"
    ipynb_file = tmp_path / "benchmark.ipynb"

    content = """# %% [markdown]
# # Title

# %%
print("Hello")
"""
    py_file.write_text(content)

    kaggle_utils.convert_py_to_ipynb(py_file, ipynb_file)

    assert ipynb_file.exists()
    with open(ipynb_file, "r") as f:
        notebook = json.load(f)

    assert len(notebook["cells"]) == 2
    assert notebook["cells"][0]["cell_type"] == "markdown"
    assert notebook["cells"][1]["cell_type"] == "code"

    # Verify the kernelspec is added so papermill can run the notebook
    assert "kernelspec" in notebook["metadata"]
    assert notebook["metadata"]["kernelspec"] == {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }


def test_convert_py_to_ipynb_warning(tmp_path):
    py_file = tmp_path / "benchmark.py"
    ipynb_file = tmp_path / "benchmark.ipynb"

    content = """print("Hello")
print("World")
"""
    py_file.write_text(content)

    with pytest.warns(UserWarning, match="has no '# %%' cell delimiters"):
        kaggle_utils.convert_py_to_ipynb(py_file, ipynb_file)

    assert ipynb_file.exists()


def test_convert_ipynb_to_py(tmp_path):
    ipynb_file = tmp_path / "benchmark.ipynb"
    py_file = tmp_path / "benchmark.py"

    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Title\n"], "metadata": {}},
            {
                "cell_type": "code",
                "source": ["print('Hello')\n"],
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(ipynb_file, "w") as f:
        json.dump(notebook, f)

    kaggle_utils.convert_ipynb_to_py(ipynb_file, py_file)

    assert py_file.exists()
    content = py_file.read_text()
    assert "# %% [markdown]" in content
    assert "# # Title" in content
    assert "# %%" in content
    assert "print('Hello')" in content


def test_roundtrip_conversion(tmp_path):
    """Test that py -> ipynb -> py preserves cell structure."""
    py_file = tmp_path / "benchmark.py"
    ipynb_file = tmp_path / "benchmark.ipynb"
    py_roundtrip = tmp_path / "benchmark_roundtrip.py"

    content = """# %% [markdown]
# # Title

# %%
print("Hello")

# %%
x = 42
"""
    py_file.write_text(content)

    kaggle_utils.convert_py_to_ipynb(py_file, ipynb_file)
    kaggle_utils.convert_ipynb_to_py(ipynb_file, py_roundtrip)

    roundtrip_content = py_roundtrip.read_text()
    assert "# %% [markdown]" in roundtrip_content
    assert "# # Title" in roundtrip_content
    assert 'print("Hello")' in roundtrip_content
    assert "x = 42" in roundtrip_content


# ---------------------------------------------------------------------------
# build_local_metadata
# ---------------------------------------------------------------------------


def test_build_local_metadata_new(tmp_path):
    workspace = tmp_path
    slug = "my-bench"
    username = "alice"

    metadata = kaggle_utils.build_local_metadata(workspace, slug, username)

    assert metadata["id"] == "alice/my-bench"
    assert metadata["title"] == "my-bench"  # Defaults to slug
    assert metadata["id_no"] is None
    assert "personal-benchmark" in metadata["keywords"]
    assert metadata["is_private"] is True
    assert "model_sources" in metadata
    assert metadata["docker_image"] is None


def test_build_local_metadata_custom_title(tmp_path):
    """Test that title can be customized separately from slug."""
    workspace = tmp_path
    slug = "my-bench"
    username = "alice"

    metadata = kaggle_utils.build_local_metadata(
        workspace, slug, username, title="My Awesome Benchmark"
    )

    assert metadata["id"] == "alice/my-bench"  # slug used for id
    assert metadata["title"] == "My Awesome Benchmark"  # custom title


def test_build_local_metadata_existing(tmp_path):
    workspace = tmp_path
    slug = "my-bench"
    username = "alice"

    existing = {
        "id": "bob/old-bench",
        "title": "old-bench",
        "id_no": 12345,
        "keywords": ["foo"],
        "docker_image": "some-image",
    }

    with open(workspace / "kernel-metadata.json", "w") as f:
        json.dump(existing, f)

    metadata = kaggle_utils.build_local_metadata(
        workspace, slug, username, dataset_sources=["alice/data"]
    )

    assert metadata["id"] == "alice/my-bench"
    assert metadata["title"] == "my-bench"
    assert metadata["id_no"] is None
    assert "personal-benchmark" in metadata["keywords"]
    assert "foo" in metadata["keywords"]
    assert metadata["docker_image"] == "some-image"
    assert metadata["dataset_sources"] == ["alice/data"]


def test_build_local_metadata_overrides(tmp_path):
    workspace = tmp_path
    slug = "my-bench"
    username = "alice"

    metadata = kaggle_utils.build_local_metadata(
        workspace,
        slug,
        username,
        enable_gpu=True,
        enable_internet=False,
        is_private=False,
        docker_image="custom-image",
    )

    assert metadata["id"] == "alice/my-bench"
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False
    assert metadata["is_private"] is False
    assert metadata["docker_image"] == "custom-image"
    assert metadata["enable_tpu"] is False  # Default


def test_build_local_metadata_idempotent_keywords(tmp_path):
    """Test that calling build_local_metadata doesn't duplicate 'personal-benchmark'."""
    workspace = tmp_path
    slug = "my-bench"
    username = "alice"

    existing = {
        "keywords": ["personal-benchmark", "foo"],
    }
    with open(workspace / "kernel-metadata.json", "w") as f:
        json.dump(existing, f)

    metadata = kaggle_utils.build_local_metadata(workspace, slug, username)

    assert metadata["keywords"].count("personal-benchmark") == 1
    assert "foo" in metadata["keywords"]


def test_build_local_metadata_malformed_json(tmp_path):
    """Test behavior when kernel-metadata.json contains invalid JSON."""
    workspace = tmp_path
    (workspace / "kernel-metadata.json").write_text("not valid json")

    with pytest.raises(json.JSONDecodeError):
        kaggle_utils.build_local_metadata(workspace, "my-bench", "alice")


# ---------------------------------------------------------------------------
# parse_remote_metadata
# ---------------------------------------------------------------------------


def test_parse_remote_metadata_from_api_response():
    """Extracts all fields from a fully populated API metadata object."""
    meta = MagicMock()
    meta.ref = "alice/my-benchmark"
    meta.title = "My Benchmark"
    meta.language = "python"
    meta.kernel_type = "notebook"
    meta.is_private = False
    meta.enable_gpu = True
    meta.enable_internet = True
    meta.enable_tpu = False
    meta.dataset_data_sources = ["alice/dataset"]
    meta.competition_data_sources = ["comp1"]
    meta.kernel_data_sources = ["alice/kernel"]
    meta.model_data_sources = ["alice/model"]
    meta.category_ids = ["personal-benchmark", "nlp"]

    result = kaggle_utils.parse_remote_metadata(
        meta, default_id="fallback/id", default_slug="fallback"
    )

    assert result["id"] == "alice/my-benchmark"
    assert result["title"] == "My Benchmark"
    assert result["language"] == "python"
    assert result["kernel_type"] == "notebook"
    assert result["is_private"] is False
    assert result["enable_gpu"] is True
    assert result["enable_internet"] is True
    assert result["enable_tpu"] is False
    assert result["dataset_sources"] == ["alice/dataset"]
    assert result["competition_sources"] == ["comp1"]
    assert result["kernel_sources"] == ["alice/kernel"]
    assert result["model_sources"] == ["alice/model"]
    assert result["keywords"] == ["personal-benchmark", "nlp"]


def test_parse_remote_metadata_uses_defaults_for_missing_attrs():
    """Falls back to defaults when API metadata has missing attributes."""
    meta = MagicMock(spec=[])  # spec=[] means no attributes exist

    result = kaggle_utils.parse_remote_metadata(
        meta, default_id="owner/slug", default_slug="my-slug"
    )

    assert result["id"] == "owner/slug"
    assert result["title"] == "my-slug"
    assert result["language"] == "python"
    assert result["kernel_type"] == "notebook"
    assert result["is_private"] is True
    assert result["enable_gpu"] is False
    assert result["enable_internet"] is True
    assert result["enable_tpu"] is False
    assert result["dataset_sources"] == []
    assert result["competition_sources"] == []
    assert result["kernel_sources"] == []
    assert result["model_sources"] == []
    assert result["keywords"] == []


def test_parse_remote_metadata_handles_none_lists():
    """Converts None list fields to empty lists."""
    meta = MagicMock()
    meta.ref = "alice/bench"
    meta.title = "Bench"
    meta.language = "python"
    meta.kernel_type = "notebook"
    meta.is_private = True
    meta.enable_gpu = False
    meta.enable_internet = True
    meta.enable_tpu = False
    meta.dataset_data_sources = None
    meta.competition_data_sources = None
    meta.kernel_data_sources = None
    meta.model_data_sources = None
    meta.category_ids = None

    result = kaggle_utils.parse_remote_metadata(
        meta, default_id="x/y", default_slug="y"
    )

    assert result["dataset_sources"] == []
    assert result["competition_sources"] == []
    assert result["kernel_sources"] == []
    assert result["model_sources"] == []
    assert result["keywords"] == []


# ---------------------------------------------------------------------------
# normalize_status
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
    """normalize_status should strip enum prefixes and lower-case."""
    assert kaggle_utils.normalize_status(raw_status) == expected


def test_normalize_status_object_with_attribute():
    """Simulates an API response object with a .status attribute."""

    class FakeStatus:
        status = "running"

    assert kaggle_utils.normalize_status(FakeStatus()) == "running"
