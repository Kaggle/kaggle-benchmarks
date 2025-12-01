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

import json

import pytest

from kaggle_benchmarks import utils


def test_code_blocks():
    block = utils.extract_code_block("""
```json
{"key": "value"}
```
""")
    assert '{"key": "value"}' == block.strip()


def test_named_code_blocks():
    block = utils.extract_code_block(
        """
```json
{"key": "value"}
```
""",
        name="json",
    )
    assert '{"key": "value"}' == block.strip()


def test_multiple_code_blocks():
    block = utils.extract_code_block(
        """
```json
{"key": "value"}
```
```python
import this
```
""",
        name="python",
    )
    assert "import this" == block.strip()


def test_json_encoder():
    class A:
        name = "b"

    value = json.dumps({"a": A()}, cls=utils.JSONEncoder)
    assert value == '{"a": "A: b"}'

    class B:
        a = 1

        def __repr__(self):
            return f"B: {self.a}"

    value = json.dumps({"x": 1, "a": B()}, cls=utils.JSONEncoder)
    assert value == '{"x": 1, "a": "B: 1"}'


@pytest.mark.parametrize(
    "name, expected",
    [
        ("a b/c\\d#e", "a_b_c_de"),
        ("normal_name", "normal_name"),
        ("", ""),
        ("Task with/slashes", "Task_with_slashes"),
    ],
)
def test_normalize_name(name, expected):
    assert utils.normalize_name(name) == expected
