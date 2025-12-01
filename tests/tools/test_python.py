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

import pytest

from kaggle_benchmarks.envs import docker, local
from kaggle_benchmarks.tools import python

envs = [
    pytest.param(local.LocalEnvironment, {}, id="local"),
    pytest.param(
        docker.DockerEnvironment,
        dict(image="python:3.11"),
        id="docker",
        marks=pytest.mark.skipif(
            not docker.available(), reason="Docker is not installed"
        ),
    ),
]


@pytest.mark.parametrize(("cls", "params"), envs)
def test_hello_world(cls, params):
    with cls(**params) as env:
        code = "print('hello world')"
        out = python.run(code, env)
        assert 0 == out.exit_code
        assert "hello world\n" == out.stdout


def test_input():
    out = python.script_runner.run_code("print(input())", input="hello\n")
    assert "hello" == out.stdout.strip()


def test_repl():
    repl = python.IPythonREPL()
    out = repl.run_code("x = 2")
    assert isinstance(out, python.CellOutput)
    assert out.status == "ok"
    assert out.traceback is None

    out = repl.run_code("print(x)")
    assert isinstance(out, python.CellOutput)
    assert out.stdout == "2\n"
    assert out.traceback is None

    out = repl.run_code("x + '2'")
    assert isinstance(out, python.CellOutput)
    assert out.status == "error"
    assert out.traceback is not None
    assert "TypeError" in out.traceback

    out = repl.run_code("x")
    assert isinstance(out, python.CellOutput)
    assert out.status == "ok", out
    assert out.output == "2"
    assert out.traceback is None
