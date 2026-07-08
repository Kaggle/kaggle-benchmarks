# Copyright 2025 Kaggle Inc.
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

import uuid

import pytest

from kaggle_benchmarks.envs import docker, local

envs = [
    pytest.param(local.InternalUnsafeLocalEnvironment, {}, id="local"),
    pytest.param(
        docker.DockerEnvironment,
        dict(image="python:3.11", working_dir="/tmp"),
        id="docker",
        marks=pytest.mark.skipif(
            not docker.available(), reason="Docker is not installed"
        ),
    ),
]


@pytest.mark.parametrize(("cls", "params"), envs)
def test_run(cls, params):
    with cls(**params) as env:
        result = env.run(["echo", "Hi"])

        assert result.exit_code == 0
        assert result.stdout.strip() == "Hi"
        assert result.stderr.strip() == ""

        result = env.run(["cat", "schrödinger"])

        assert result.exit_code != 0
        assert result.stdout.strip() == ""
        assert result.stderr.strip()


@pytest.mark.parametrize(("cls", "params"), envs)
@pytest.mark.parametrize("path", ["test", "./test2", "subfolder/file"])
def test_read_write(cls, params, path):
    with cls(**params) as env:
        content = str(uuid.uuid4())
        env[path] = content
        assert content == env[path]

        with pytest.raises(FileNotFoundError):
            env["cage"]


@pytest.mark.parametrize(("cls", "params"), envs)
def test_missing_file(cls, params):
    with cls(**params) as env:
        with pytest.raises(FileNotFoundError):
            env["cage"]


@pytest.mark.parametrize(("cls", "params"), envs)
def test_absolute_path(cls, params):
    with cls(**params) as env:
        with pytest.raises(ValueError):
            env["/cage"]

        with pytest.raises(ValueError):
            env["/cage"] = "test"


@pytest.mark.parametrize(("cls", "params"), envs)
@pytest.mark.parametrize(
    "path",
    [
        "../escape",
        "../../escape",
        "subfolder/../../escape",
        "./../escape",
    ],
    ids=["parent", "grandparent", "nested-traversal", "dot-parent"],
)
def test_parent_traversal_rejected(cls, params, path, tmp_path, monkeypatch):
    """Regression test for Kaggle/kaggle-benchmarks#159.

    Relative ``..`` traversal must not be able to escape the environment's
    temporary directory, even when the resolved path would land in a
    directory the current user has write access to.
    """
    # Point traversal at a scratch directory we own so a buggy implementation
    # would actually succeed in writing/reading there, making the test a true
    # positive rather than incidentally passing on a permission error.
    monkeypatch.chdir(tmp_path)

    with cls(**params) as env:
        with pytest.raises(ValueError):
            env[path] = "escaped"

        with pytest.raises(ValueError):
            env[path]

        with pytest.raises(ValueError):
            env.open(path, "w")


@pytest.mark.parametrize(("cls", "params"), envs)
def test_setitem_does_not_write_outside_directory(cls, params, tmp_path, monkeypatch):
    """Even if the ValueError check were skipped, no file should land outside
    the environment's temp directory."""
    monkeypatch.chdir(tmp_path)
    marker_name = "kbench_path_escape_marker"

    with cls(**params) as env:
        with pytest.raises(ValueError):
            env[f"../{marker_name}"] = "escaped"

    assert not (tmp_path / marker_name).exists()
