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

from kaggle_benchmarks import assertions, chats
from kaggle_benchmarks.llm_messages import LLMMessage
from tests.mocks import MockedChat


def _make_tool_call_response(name: str, arguments: dict, call_id: str = "call_1"):
    """Creates an LLMMessage that simulates a tool call from the LLM."""
    msg = LLMMessage(sender=None, content="")
    msg._meta["tool_calls"] = [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }
    ]
    return msg


def _add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b


def _always_fails() -> str:
    """This tool always fails."""
    raise ValueError("Simulated error")


class TestToolInvocationLoop:
    """Tests for the prompt() tool invocation loop."""

    def test_tool_called_and_result_returned(self):
        """The LLM requests a tool call, the tool runs, and the final answer
        uses the result."""
        # First response: LLM requests tool call.
        # Second response: LLM gives final answer using tool result.
        tool_response = _make_tool_call_response("_add", {"a": 3, "b": 4})
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
        tool_response = _make_tool_call_response("_always_fails", {})
        final_response = LLMMessage(sender=None, content="The tool failed.")

        llm = MockedChat(responses=[tool_response, final_response])
        result = llm.prompt("Call the tool.", tools=[_always_fails])

        assert result == "The tool failed."

    def test_extra_arguments_filtered(self):
        """Extra arguments from the API (like 'signature') are filtered out."""
        tool_response = _make_tool_call_response(
            "_add", {"a": 5, "b": 10, "signature": "extra_field"}
        )
        final_response = LLMMessage(sender=None, content="15")

        llm = MockedChat(responses=[tool_response, final_response])
        result = llm.prompt("5 + 10?", tools=[_add])

        # Should succeed despite the extra 'signature' argument.
        assert result == "15"

    def test_assert_tool_was_invoked_in_forked_chat(self):
        """assert_tool_was_invoked finds tools in the forked subchat."""
        tool_response = _make_tool_call_response("_add", {"a": 1, "b": 2})
        final_response = LLMMessage(sender=None, content="3")

        llm = MockedChat(responses=[tool_response, final_response])
        llm.prompt("1+2?", tools=[_add])

        # The assertion should find the tool result in the nested fork.
        result = assertions.assert_tool_was_invoked(_add)
        assert result.passed

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
