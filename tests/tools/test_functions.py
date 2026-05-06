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

import pytest
from pydantic import BaseModel

from kaggle_benchmarks.tools.functions import (
    ToolSchemaError,
    function_to_genai_tool,
    function_to_openai_tool,
    get_function_schema,
)


def sample_function(a: int, b: str = "hello") -> float:
    """This is a sample function."""
    return float(a)


def test_get_function_schema():
    schema = get_function_schema(sample_function)
    assert schema["type"] == "object"
    assert schema["properties"]["a"]["type"] == "integer"
    assert schema["properties"]["b"]["type"] == "string"
    assert schema["properties"]["b"]["default"] == "hello"
    assert "a" in schema["required"]


def test_get_function_schema_with_unserializable_arg():
    class Unserializable:
        pass

    def func_with_unserializable_arg(a: Unserializable):
        return a

    with pytest.raises(ToolSchemaError):
        get_function_schema(func_with_unserializable_arg)


def test_function_to_openai_tool():
    tool = function_to_openai_tool(sample_function)
    assert tool["type"] == "function"
    func = tool["function"]
    assert func["name"] == "sample_function"
    assert func["description"] == "This is a sample function."
    assert func["parameters"]["properties"]["a"]["type"] == "integer"
    assert "a" in func["parameters"]["required"]


def test_function_to_openai_pydantic():
    class Args(BaseModel):
        a: int
        b: str

    def sample_function_pydantic(args: Args):
        """This is a sample function."""
        return args.a

    tool = function_to_openai_tool(sample_function_pydantic)
    func = tool["function"]
    assert func["name"] == "sample_function_pydantic"
    assert func["parameters"]["properties"]["args"]["$ref"] == "#/$defs/Args"
    defs = func["parameters"]["$defs"]
    assert defs["Args"]["properties"]["a"]["type"] == "integer"
    assert defs["Args"]["properties"]["b"]["type"] == "string"


def test_function_to_genai_tool():
    tool = function_to_genai_tool(sample_function)
    assert tool.name == "sample_function"
    assert tool.description == "This is a sample function."
    assert tool.parameters.properties["a"].type.name == "INTEGER"


def test_function_to_genai_pydantic():
    class Args(BaseModel):
        a: int
        b: str

    def sample_function_pydantic(args: Args):
        """This is a sample function."""
        return args.a

    tool = function_to_genai_tool(sample_function_pydantic)
    assert tool.name == "sample_function_pydantic"
    assert tool.description == "This is a sample function."
    assert tool.parameters.properties["args"].type.name == "OBJECT"
