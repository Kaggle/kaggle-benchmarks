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

"""Tests for the tool invocation loop in LLMChat.prompt()."""

import pytest

from kaggle_benchmarks import assertions, chats
from kaggle_benchmarks.llm_messages import LLMMessage
from kaggle_benchmarks.tools.base import (
    ToolInvocation,
    ToolInvocationLimitExhausted,
    ToolInvocationResult,
    invoke_tool,
)
from tests.mocks import MockedChat

# ---------------------------------------------------------------------------
# Test tool functions
# ---------------------------------------------------------------------------


def _add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b


def _multiply(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b


def _always_fails() -> str:
    """This tool always fails."""
    raise ValueError("Simulated error")


def _no_args() -> str:
    """A tool that takes no arguments."""
    return "ok"


# ---------------------------------------------------------------------------
# ToolInvocation.from_api_dict unit tests
# ---------------------------------------------------------------------------


class TestToolInvocationFromApiDict:
    """Tests for ToolInvocation.from_api_dict() edge cases."""

    def test_dict_arguments(self):
        """Normal case: arguments as a dict."""
        invocation = ToolInvocation.from_api_dict(
            {"id": "c1", "function": {"name": "add", "arguments": {"a": 1, "b": 2}}}
        )
        assert invocation.name == "add"
        assert invocation.arguments == {"a": 1, "b": 2}
        assert invocation.call_id == "c1"

    def test_string_arguments(self):
        """OpenAI returns arguments as a JSON string."""
        invocation = ToolInvocation.from_api_dict(
            {
                "id": "c2",
                "function": {
                    "name": "add",
                    "arguments": '{"a": 1, "b": 2}',
                },
            }
        )
        assert invocation.arguments == {"a": 1, "b": 2}

    def test_none_arguments(self):
        """Some models return arguments=null for parameterless tools."""
        invocation = ToolInvocation.from_api_dict(
            {"id": "c3", "function": {"name": "noop", "arguments": None}}
        )
        assert invocation.arguments == {}

    def test_missing_arguments(self):
        """Some models omit the arguments key entirely."""
        invocation = ToolInvocation.from_api_dict(
            {"id": "c4", "function": {"name": "noop"}}
        )
        assert invocation.arguments == {}

    def test_missing_id(self):
        """call_id should be None when id is not provided."""
        invocation = ToolInvocation.from_api_dict(
            {"function": {"name": "add", "arguments": {"a": 1}}}
        )
        assert invocation.call_id is None


# ---------------------------------------------------------------------------
# ToolInvocationResult tests
# ---------------------------------------------------------------------------


class TestToolInvocationResult:
    """Tests for ToolInvocationResult fields and methods."""

    def test_text_returns_output(self):
        result = ToolInvocationResult(name="f", arguments={}, output=42)
        assert result.text == "42"

    def test_text_returns_error(self):
        result = ToolInvocationResult(
            name="f", arguments={}, error="Something went wrong"
        )
        assert result.text == "Something went wrong"

    def test_error_takes_precedence_in_text(self):
        result = ToolInvocationResult(name="f", arguments={}, output=42, error="oops")
        assert result.text == "oops"

    def test_describe_success(self):
        result = ToolInvocationResult(name="add", arguments={"a": 1}, output=2)
        assert "add" in result.describe()
        assert "2" in result.describe()
        assert "Error" not in result.describe()

    def test_describe_error(self):
        result = ToolInvocationResult(name="add", arguments={}, error="Error: boom")
        desc = result.describe()
        assert "Error" in desc
        assert "boom" in desc


# ---------------------------------------------------------------------------
# invoke_tool tests
# ---------------------------------------------------------------------------


class TestInvokeTool:
    """Tests for invoke_tool with error field separation."""

    def test_tool_not_found_sets_error(self):
        """When tool is not found, error field should be set (not output)."""
        invocation = ToolInvocation(name="nonexistent", arguments={})
        result = invoke_tool(invocation, [_add])

        assert result.error is not None
        assert "not found" in result.error
        assert result.output is None

    def test_tool_exception_sets_error(self):
        """When tool raises, error field should be set (not output)."""
        invocation = ToolInvocation(name="_always_fails", arguments={})
        result = invoke_tool(invocation, [_always_fails])

        assert result.error is not None
        assert "Simulated error" in result.error
        assert result.output is None

    def test_tool_success_sets_output(self):
        """Successful tool call sets output, not error."""
        invocation = ToolInvocation(name="_add", arguments={"a": 3, "b": 4})
        result = invoke_tool(invocation, [_add])

        assert result.output == 7.0
        assert result.error is None


# ---------------------------------------------------------------------------
# Tool invocation loop tests
# ---------------------------------------------------------------------------


class TestToolInvocationLoop:
    """Tests for the prompt() tool invocation loop."""

    def test_tool_called_and_result_returned(self):
        """The LLM requests a tool call, the tool runs, and the final answer
        uses the result."""
        tool_response = MockedChat.make_tool_call_response("_add", {"a": 3, "b": 4})
        final_response = LLMMessage(sender=None, content="The answer is 7.")

        llm = MockedChat(responses=[tool_response, final_response])
        result = llm.prompt("What is 3 + 4?", tools=[_add])

        assert result == "The answer is 7."

    def test_tool_not_called_when_no_tool_calls(self):
        """When the LLM doesn't request any tools, prompt() returns directly."""
        response = LLMMessage(sender=None, content="The answer is 42.")
        llm = MockedChat(responses=[response])

        result = llm.prompt("What is 42?", tools=[_add])
        assert result == "The answer is 42."
        assert len(llm.invocations) == 1  # Only one call, no loop

    def test_tool_error_sent_back_to_llm(self):
        """When a tool raises an exception, the error message is sent back."""
        tool_response = MockedChat.make_tool_call_response("_always_fails", {})
        final_response = LLMMessage(sender=None, content="The tool failed.")

        llm = MockedChat(responses=[tool_response, final_response])
        result = llm.prompt("Call the tool.", tools=[_always_fails])

        assert result == "The tool failed."

    def test_extra_arguments_filtered(self):
        """Extra arguments from the API (like 'signature') are filtered out."""
        tool_response = MockedChat.make_tool_call_response(
            "_add", {"a": 5, "b": 10, "signature": "extra_field"}
        )
        final_response = LLMMessage(sender=None, content="15")

        llm = MockedChat(responses=[tool_response, final_response])
        result = llm.prompt("5 + 10?", tools=[_add])

        # Should succeed despite the extra 'signature' argument.
        assert result == "15"

    def test_assert_tool_was_invoked_in_forked_chat(self):
        """assert_tool_was_invoked finds tools in the forked subchat."""
        tool_response = MockedChat.make_tool_call_response("_add", {"a": 1, "b": 2})
        final_response = LLMMessage(sender=None, content="3")

        llm = MockedChat(responses=[tool_response, final_response])
        llm.prompt("1+2?", tools=[_add])

        # The assertion should find the tool result in the nested fork.
        result = assertions.assert_tool_was_invoked(_add)
        assert result.passed

    def test_assert_tool_was_invoked_by_name(self):
        """assert_tool_was_invoked accepts a string tool name."""
        tool_response = MockedChat.make_tool_call_response("_add", {"a": 1, "b": 2})
        final_response = LLMMessage(sender=None, content="3")

        llm = MockedChat(responses=[tool_response, final_response])
        llm.prompt("1+2?", tools=[_add])

        assert assertions.assert_tool_was_invoked("_add").passed
        assert not assertions.assert_tool_was_invoked("_multiply").passed

    def test_no_fork_without_tools(self):
        """When no tools are provided, prompt() doesn't fork the chat."""
        response = LLMMessage(sender=None, content="Hello.")
        llm = MockedChat(responses=[response])

        result = llm.prompt("Hi")
        assert result == "Hello."

        # No nested chats in history.
        chat = chats.get_current_chat()
        nested = [item for item in chat.history if isinstance(item, chats.Chat)]
        assert len(nested) == 0

    def test_tool_invocation_limit_exhausted(self):
        """ToolInvocationLimitExhausted is raised when max rounds are exceeded."""
        tool_response = MockedChat.make_tool_call_response("_add", {"a": 1, "b": 2})
        llm = MockedChat(responses=[tool_response], cycle=True)

        with pytest.raises(ToolInvocationLimitExhausted):
            llm.prompt("Keep calling tools.", tools=[_add])

    def test_max_tool_rounds_parameter(self):
        """max_tool_rounds can be set to 1 to limit iterations."""
        tool_response = MockedChat.make_tool_call_response("_add", {"a": 1, "b": 2})
        llm = MockedChat(responses=[tool_response], cycle=True)

        with pytest.raises(ToolInvocationLimitExhausted):
            llm.prompt(
                "Keep calling tools.",
                tools=[_add],
                extra_api_params={"max_tool_rounds": 1},
            )
        # With max_tool_rounds=1, only 1 invocation should have been made.
        assert len(llm.invocations) == 1

    def test_none_arguments_handled(self):
        """When the model returns None arguments, the tool still executes."""
        tool_response = MockedChat.make_tool_call_response("_no_args", None)
        final_response = LLMMessage(sender=None, content="done")

        llm = MockedChat(responses=[tool_response, final_response])
        result = llm.prompt("Call the tool.", tools=[_no_args])

        assert result == "done"

    def test_multiple_tool_calls_in_single_response(self):
        """When the LLM requests multiple tools at once, all are invoked."""
        msg = LLMMessage(sender=None, content="")
        msg._meta["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "_add", "arguments": {"a": 1, "b": 2}},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "_multiply",
                    "arguments": {"a": 3, "b": 4},
                },
            },
        ]
        final_response = LLMMessage(sender=None, content="1+2=3, 3*4=12")

        llm = MockedChat(responses=[msg, final_response])
        result = llm.prompt("Calculate both.", tools=[_add, _multiply])

        assert result == "1+2=3, 3*4=12"
        # 2 invocations: first with tool calls, second with final answer.
        assert len(llm.invocations) == 2

    def test_tool_not_found_through_loop(self):
        """When the LLM calls a nonexistent tool, the error is sent back."""
        tool_response = MockedChat.make_tool_call_response("nonexistent_tool", {"x": 1})
        final_response = LLMMessage(sender=None, content="That tool doesn't exist.")

        llm = MockedChat(responses=[tool_response, final_response])
        result = llm.prompt("Call the tool.", tools=[_add])

        assert result == "That tool doesn't exist."
