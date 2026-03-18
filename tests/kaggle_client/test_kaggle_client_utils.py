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

import pytest

from kaggle_benchmarks.kaggle_client.utils import (
    convert_ipynb_to_py,
    convert_py_to_ipynb,
    resolve_metadata,
)


def test_convert_py_to_ipynb(tmp_path):
    py_file = tmp_path / "benchmark.py"
    ipynb_file = tmp_path / "benchmark.ipynb"

    content = """# %% [markdown]
# # Title

# %%
print("Hello")
"""
    py_file.write_text(content)

    convert_py_to_ipynb(py_file, ipynb_file)

    assert ipynb_file.exists()
    with open(ipynb_file, "r") as f:
        notebook = json.load(f)

    assert len(notebook["cells"]) == 2
    assert notebook["cells"][0]["cell_type"] == "markdown"
    assert notebook["cells"][1]["cell_type"] == "code"


def test_convert_py_to_ipynb_warning(tmp_path):
    py_file = tmp_path / "benchmark.py"
    ipynb_file = tmp_path / "benchmark.ipynb"

    content = """print("Hello")
print("World")
"""
    py_file.write_text(content)

    with pytest.warns(UserWarning, match="has no '# %%' cell delimiters"):
        convert_py_to_ipynb(py_file, ipynb_file)

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

    convert_ipynb_to_py(ipynb_file, py_file)

    assert py_file.exists()
    content = py_file.read_text()
    assert "# %% [markdown]" in content
    assert "# # Title" in content
    assert "# %%" in content
    assert "print('Hello')" in content


def test_resolve_metadata_new(tmp_path):
    workspace = tmp_path
    slug = "my-bench"
    username = "alice"

    metadata = resolve_metadata(workspace, slug, username)

    assert metadata["id"] == "alice/my-bench"
    assert metadata["title"] == "my-bench"  # Defaults to slug
    assert metadata["id_no"] is None
    assert "personal-benchmark" in metadata["keywords"]
    assert metadata["is_private"] is True
    assert "model_sources" in metadata
    assert metadata["docker_image"] is None


def test_resolve_metadata_custom_title(tmp_path):
    """Test that title can be customized separately from slug."""
    workspace = tmp_path
    slug = "my-bench"
    username = "alice"

    metadata = resolve_metadata(workspace, slug, username, title="My Awesome Benchmark")

    assert metadata["id"] == "alice/my-bench"  # slug used for id
    assert metadata["title"] == "My Awesome Benchmark"  # custom title


def test_resolve_metadata_existing(tmp_path):
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

    metadata = resolve_metadata(
        workspace, slug, username, dataset_sources=["alice/data"]
    )

    assert metadata["id"] == "alice/my-bench"
    assert metadata["title"] == "my-bench"
    assert metadata["id_no"] is None
    assert "personal-benchmark" in metadata["keywords"]
    assert "foo" in metadata["keywords"]
    assert metadata["docker_image"] == "some-image"
    assert metadata["dataset_sources"] == ["alice/data"]


def test_resolve_metadata_overrides(tmp_path):
    workspace = tmp_path
    slug = "my-bench"
    username = "alice"

    metadata = resolve_metadata(
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


def test_resolve_metadata_idempotent_keywords(tmp_path):
    """Test that calling resolve_metadata doesn't duplicate 'personal-benchmark'."""
    workspace = tmp_path
    slug = "my-bench"
    username = "alice"

    existing = {
        "keywords": ["personal-benchmark", "foo"],
    }
    with open(workspace / "kernel-metadata.json", "w") as f:
        json.dump(existing, f)

    metadata = resolve_metadata(workspace, slug, username)

    assert metadata["keywords"].count("personal-benchmark") == 1
    assert "foo" in metadata["keywords"]


def test_resolve_metadata_malformed_json(tmp_path):
    """Test behavior when kernel-metadata.json contains invalid JSON."""
    workspace = tmp_path
    (workspace / "kernel-metadata.json").write_text("not valid json")

    with pytest.raises(json.JSONDecodeError):
        resolve_metadata(workspace, "my-bench", "alice")


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

    convert_py_to_ipynb(py_file, ipynb_file)
    convert_ipynb_to_py(ipynb_file, py_roundtrip)

    roundtrip_content = py_roundtrip.read_text()
    assert "# %% [markdown]" in roundtrip_content
    assert "# # Title" in roundtrip_content
    assert 'print("Hello")' in roundtrip_content
    assert "x = 42" in roundtrip_content
