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
import uuid

import httpx
import pydantic
import pytest
import respx

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


@respx.mock
@pytest.mark.parametrize("method", ["get", "post"])
def test_client_caches_despite_server_headers(tmp_path, method, cfg):
    cfg.cache_directory = tmp_path
    cfg.enable_caching = True
    url = f"https://test.com/{uuid.uuid4()}"
    client = utils.build_httpx_client(filename="test")
    route = respx.request(method, url).mock(
        return_value=httpx.Response(200, content="")
    )

    resp1 = client.request(method, url, headers={"Cache-Control": "no-cache"})
    assert resp1.status_code == 200
    assert route.called
    assert not resp1.extensions.get("hishel_from_cache")

    resp2 = client.request(method, url)
    assert resp2.status_code == 200
    assert route.call_count == 1
    assert resp2.extensions.get("hishel_from_cache") is True


def test_client_respects_disable_config(cfg):
    cfg.enable_caching = False
    client = utils.build_httpx_client()

    assert isinstance(client, httpx.Client)
    assert not hasattr(client, "_controller")


@respx.mock
@pytest.mark.parametrize("method", ["get", "post"])
def test_client_does_not_cache_error_responses(tmp_path, method, cfg):
    cfg.cache_directory = tmp_path
    cfg.enable_caching = True

    url = f"https://test.com/{uuid.uuid4()}"
    client = utils.build_httpx_client(filename="test")
    route = respx.request(method, url).mock(return_value=httpx.Response(400))

    resp1 = client.request(method, url)
    assert resp1.status_code == 400
    assert route.called
    assert not resp1.extensions.get("hishel_from_cache")

    resp2 = client.request(method, url)
    assert resp2.status_code == 400
    assert route.call_count == 2
    assert not resp2.extensions.get("hishel_from_cache")


class NestedModel(pydantic.BaseModel):
    a: int


class MyModel(pydantic.BaseModel):
    nested: NestedModel
    b: str


class SimpleModel(pydantic.BaseModel):
    a: int
    b: str


@pytest.mark.parametrize(
    "model, expected",
    [
        (MyModel, True),
        (SimpleModel, False),
        (None, False),
        (int, False),
        (str, False),
    ],
)
def test_has_nested_models(model, expected):
    assert utils.has_nested_models(model) is expected
