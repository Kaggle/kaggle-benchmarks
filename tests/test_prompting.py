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

import datetime
import json
from dataclasses import dataclass

import pydantic
import pytest

from kaggle_benchmarks import actors, chats, messages, prompting
from kaggle_benchmarks.actors.llms import LLMResponse
from kaggle_benchmarks.prompting import ResponseParsingError


def test_str():
    for x in ["", "ab", "cdef"]:
        assert x == prompting.parse_response(prompting.process_schema(str), x)


@pytest.mark.parametrize(
    "schema_type, input_value, expected_value",
    [
        pytest.param(int, 123, 123, id="int"),
        pytest.param(float, 123.45, 123.45, id="float"),
        pytest.param(bool, True, True, id="bool_true"),
        pytest.param(bool, False, False, id="bool_false"),
        pytest.param(
            datetime.datetime,
            "2025-01-01T10:00:00Z",
            datetime.datetime(2025, 1, 1, 10, 0, tzinfo=datetime.timezone.utc),
            id="datetime",
        ),
    ],
)
def test_primitive_types(schema_type, input_value, expected_value):
    json_input = json.dumps({"value": input_value})
    output_value = prompting.parse_response(
        prompting.process_schema(schema_type), json_input
    )
    assert output_value == expected_value


def test_dataclass():
    @dataclass
    class A:
        a: str = "default"
        b: int = 1

    assert A("a", 2) == prompting.parse_response(
        prompting.process_schema(A), '{"a": "a", "b": 2}'
    )
    assert A("v") == prompting.parse_response(
        prompting.process_schema(A), '```json{"a": "v"}```'
    )


def test_typed_dict():
    response = prompting.parse_response(
        prompting.process_schema({"x": int, "y": int}), '{"x": 2, "y": -2}'
    )

    assert response.x == 2
    assert response.y == -2


def test_llm():
    @dataclass
    class A:
        test_field: str = "default"
        another_field: int = 1

    class LLM(actors.LLMChat):
        def invoke(
            self, messages: list[messages.Message], system: str | None, **kwargs
        ) -> LLMResponse:
            texts = "\n".join(m.text for m in messages) + str(system)
            assert "test_field" in texts
            assert "another_field" in texts
            return LLMResponse(content='{"test_field": "a", "another_field": 2}')

    with chats.new("test"):
        response = LLM().prompt(message="?", schema=A)
        assert isinstance(response, A)
        assert response == A("a", 2)


def test_pydantic_error():
    class Model(pydantic.BaseModel):
        a: int

    with pytest.raises(ResponseParsingError):
        prompting.parse_response(prompting.process_schema(Model), '{"a": "not an int"}')


def test_nested_pydantic():
    class Inner(pydantic.BaseModel):
        c: int

    class Outer(pydantic.BaseModel):
        a: str
        b: Inner

    json_input = '{"a": "hello", "b": {"c": 123}}'
    expected_output = Outer(a="hello", b=Inner(c=123))
    output = prompting.parse_response(prompting.process_schema(Outer), json_input)
    assert output == expected_output


def test_pydantic_validation_error_str():
    """Tests that the string representation of ResponseParsingError has key info."""

    class Model(pydantic.BaseModel):
        a: int

    input_val = '{"a": "not an int"}'
    with pytest.raises(ResponseParsingError) as excinfo:
        prompting.parse_response(prompting.process_schema(Model), input_val)

    error_str = str(excinfo.value)
    pydantic_error_detail = str(excinfo.value.error)

    assert "Response parsing failed" in error_str
    assert input_val in error_str
    assert "Target Schema" in error_str
    assert '"title": "Model"' in error_str
    assert "Parsing Error" in error_str
    # Check for key info in the pydantic error, not the exact message
    assert "a" in pydantic_error_detail
    assert "not an int" in pydantic_error_detail


def test_dataclass_json_decode_error_str():
    """Tests ResponseParsingError for a dataclass with invalid JSON has key info."""

    @dataclass
    class MyData:
        name: str
        value: int

    input_val = '{"name": "test", "value": 123,}'  # Malformed JSON with trailing comma
    with pytest.raises(ResponseParsingError) as excinfo:
        prompting.parse_response(prompting.process_schema(MyData), input_val)

    error_str = str(excinfo.value)
    json_error_detail = str(excinfo.value.error)

    assert "Response parsing failed" in error_str
    assert input_val in error_str
    assert "Target Schema" in error_str
    assert '"title": "MyData"' in error_str
    assert "Parsing Error" in error_str
    # Check that the error is a JSON decoding error with position info
    assert "line 1" in json_error_detail
    assert "column" in json_error_detail
