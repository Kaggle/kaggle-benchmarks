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


def test_extract_code_single_block():
    """Test extracting a single Python code block."""
    text = """
Here is some code:

```python
x = 1
print(x)
```
"""
    result = python.extract_code(text)
    assert result == "x = 1\nprint(x)"


def test_extract_code_multiple_blocks_default():
    """Test that default behavior returns only first block."""
    text = """
```python
x = 1
```

```python
y = 2
```
"""
    result = python.extract_code(text)
    assert result == "x = 1"


def test_extract_code_multiple_blocks_all():
    """Test extracting all Python code blocks."""
    text = """
```python
x = 1
```

Some explanation here.

```python
y = x + 1
print(y)
```
"""
    result = python.extract_code(text, all_blocks=True)
    assert result == "x = 1\n\ny = x + 1\nprint(y)"


def test_extract_code_no_blocks():
    """Test fallback when no code blocks present."""
    text = "Just some plain text"
    result = python.extract_code(text)
    assert result == text


def test_extract_code_mixed_languages():
    """Test that only Python blocks are extracted."""
    text = """
```javascript
console.log('hello');
```

```python
print('hello')
```
"""
    result = python.extract_code(text)
    assert result == "print('hello')"


def test_repl_with_multiple_blocks():
    """Test REPL execution with multiple concatenated code blocks."""
    repl = python.IPythonREPL()

    multi_block_text = """
```python
x = 10
```

```python
y = x * 2
print(y)
```
"""
    code = python.extract_code(multi_block_text, all_blocks=True)
    out = repl.run_code(code)

    assert out.status == "ok"
    assert out.stdout.strip() == "20"


def test_replace_ansi_colors():
    # Test basic color replacement
    text = "\x1b[31mRed\x1b[0m"
    # Note: Current implementation has a bug where \x1b[0m is matched by the regex
    # and replaced with <span style=""> instead of closing the span.
    # We test for exact behavior preservation here.
    expected = '<span style="color: red;">Red<span style="">'
    assert python.replace_ansi_colors(text) == expected

    # Test multiple styles (one valid, one invalid/ignored)
    text = "\x1b[31;1mRed\x1b[0m"
    expected = '<span style="color: red;">Red<span style="">'
    assert python.replace_ansi_colors(text) == expected

    # Test background color
    text = "\x1b[42mGreenBG\x1b[0m"
    expected = '<span style="background-color: green;">GreenBG<span style="">'
    assert python.replace_ansi_colors(text) == expected
