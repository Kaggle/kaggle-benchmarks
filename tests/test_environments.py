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
        result = env.run("echo Hi")

        assert result.exit_code == 0
        assert result.stdout.strip() == "Hi"
        assert result.stderr.strip() == ""

        result = env.run("echo Hi".split())

        assert result.exit_code == 0
        assert result.stdout.strip() == "Hi"
        assert result.stderr.strip() == ""

        result = env.run("cat schrödinger")

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
