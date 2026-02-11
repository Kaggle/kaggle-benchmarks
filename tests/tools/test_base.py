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

from kaggle_benchmarks.tools.base import (
    ToolInvocation,
    describe_tools,
    invoke_tool,
)


def simple_tool(a: int, b: str = "default") -> str:
    """A simple tool for testing."""
    return f"{a}-{b}"


def tool_that_raises():
    """A tool that always raises an exception."""
    raise ValueError("This tool failed.")


def test_describe_tools_no_tools():
    assert describe_tools([]) == "No tools available."


def test_describe_tools_with_tools():
    description = describe_tools([simple_tool, tool_that_raises])
    assert "simple_tool(a: int, b: str = 'default') -> str" in description
    assert "A simple tool for testing." in description
    assert "tool_that_raises()" in description
    assert "A tool that always raises an exception." in description


def test_invoke_tool_success():
    call = ToolInvocation(name="simple_tool", arguments={"a": 1, "b": "test"})
    result = invoke_tool(call, [simple_tool])
    assert result.output == "1-test"
    assert result.name == "simple_tool"


def test_invoke_tool_not_found():
    call = ToolInvocation(name="non_existent_tool", arguments={})
    result = invoke_tool(call, [simple_tool])
    assert "Error: Tool 'non_existent_tool' not found." in result.output


def test_invoke_tool_exception():
    call = ToolInvocation(name="tool_that_raises", arguments={})
    result = invoke_tool(call, [tool_that_raises])
    assert "Error invoking tool 'tool_that_raises': This tool failed." in result.output
