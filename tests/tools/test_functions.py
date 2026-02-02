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

from kaggle_benchmarks.tools.functions import (
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


def test_function_to_openai_tool():
    tool = function_to_openai_tool(sample_function)
    assert tool["type"] == "function"
    assert tool["name"] == "sample_function"
    assert tool["description"] == "This is a sample function."
    assert tool["parameters"]["properties"]["a"]["type"] == "integer"
    assert "a" in tool["parameters"]["required"]


def test_function_to_genai_tool_from_func():
    tool = function_to_genai_tool(sample_function)
    assert tool.name == "sample_function"
    assert tool.description == "This is a sample function."
    assert tool.parameters.properties["a"].type.name == "INTEGER"


def test_function_to_genai_tool_from_dict():
    openai_tool = {
        "name": "sample_function",
        "description": "This is a sample function.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
        },
    }
    tool = function_to_genai_tool(openai_tool)
    assert tool.name == "sample_function"
    assert tool.description == "This is a sample function."
    assert tool.parameters.properties["a"].type.name == "INTEGER"
