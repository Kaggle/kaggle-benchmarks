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

from types import SimpleNamespace

import pydantic
import pytest

from kaggle_benchmarks import actors, assertions, chats
from kaggle_benchmarks.actors.llms import LLMResponse
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


def _make_tool_call_response(
    name: str,
    arguments: dict | None = None,
    call_id: str = "call_1",
) -> LLMMessage[str]:
    """Creates an LLMMessage that simulates a tool call from the LLM.

    Populates the typed `tool_calls` field with a ToolInvocation. We can't
    use _meta["tool_calls"] here: MockedChat hits respond()'s LLMMessage
    branch which does not normalize _meta, and LLMMessage's typed
    tool_calls field shadows the Message.tool_calls property shim.
    """
    return LLMMessage(
        sender=None,
        content="",
        tool_calls=[
            ToolInvocation(name=name, arguments=arguments or {}, call_id=call_id)
        ],
    )


# ---------------------------------------------------------------------------
# ToolInvocation.from_api_dict tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "api_dict, expected_name, expected_args, expected_id",
    [
        # Happy paths: dict and string-encoded dict arguments.
        pytest.param(
            {"id": "c1", "function": {"name": "add", "arguments": {"a": 1, "b": 2}}},
            "add",
            {"a": 1, "b": 2},
            "c1",
            id="dict_arguments",
        ),
        pytest.param(
            {"id": "c2", "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}},
            "add",
            {"a": 1, "b": 2},
            "c2",
            id="string_arguments",
        ),
        # Missing / null inputs → empty-args default.
        pytest.param(
            {"id": "c3", "function": {"name": "noop", "arguments": None}},
            "noop",
            {},
            "c3",
            id="none_arguments",
        ),
        pytest.param(
            {"id": "c4", "function": {"name": "noop"}},
            "noop",
            {},
            "c4",
            id="missing_arguments",
        ),
        pytest.param(
            {"function": {"name": "add", "arguments": {"a": 1}}},
            "add",
            {"a": 1},
            None,
            id="missing_id",
        ),
        # JSON "null" treated as no-args (parity with the missing-key default).
        pytest.param(
            {"id": "c5", "function": {"name": "f", "arguments": "null"}},
            "f",
            {},
            "c5",
            id="json_null_treated_as_no_args",
        ),
        # Malformed / non-dict JSON: kept as the raw string so invoke_tool
        # can surface a clear error instead of crashing.
        pytest.param(
            {"id": "c6", "function": {"name": "f", "arguments": '{"a":'}},
            "f",
            '{"a":',
            "c6",
            id="malformed_json_kept_as_string",
        ),
        pytest.param(
            {"id": "c7", "function": {"name": "f", "arguments": "[1, 2, 3]"}},
            "f",
            "[1, 2, 3]",
            "c7",
            id="json_list_kept_as_string",
        ),
        pytest.param(
            {"id": "c8", "function": {"name": "f", "arguments": "42"}},
            "f",
            "42",
            "c8",
            id="json_scalar_kept_as_string",
        ),
    ],
)
def test_from_api_dict(api_dict, expected_name, expected_args, expected_id):
    """ToolInvocation.from_api_dict handles various argument formats."""
    invocation = ToolInvocation.from_api_dict(api_dict)
    assert invocation.name == expected_name
    assert invocation.arguments == expected_args
    assert invocation.call_id == expected_id


@pytest.mark.parametrize(
    "api_dict, expected_signature, expected_thought",
    [
        pytest.param(
            {
                "id": "c1",
                "function": {"name": "f", "arguments": {}},
                "_thought_signature": b"sig-bytes",
                "_thought": True,
            },
            b"sig-bytes",
            True,
            id="present",
        ),
        pytest.param(
            {"id": "c2", "function": {"name": "f", "arguments": {}}},
            None,
            None,
            id="absent_defaults_to_none",
        ),
    ],
)
def test_from_api_dict_thought_fields(api_dict, expected_signature, expected_thought):
    """Gemini 3.x carriers round-trip through _thought_signature / _thought keys."""
    invocation = ToolInvocation.from_api_dict(api_dict)
    assert invocation.thought_signature == expected_signature
    assert invocation.thought is expected_thought


# ---------------------------------------------------------------------------
# ToolInvocationResult tests
# ---------------------------------------------------------------------------


def test_result_text_returns_output():
    result = ToolInvocationResult(name="f", arguments={}, output=42)
    assert result.text == "42"


def test_result_text_returns_error():
    result = ToolInvocationResult(name="f", arguments={}, error="Something went wrong")
    assert result.text == "Something went wrong"


def test_result_error_takes_precedence_in_text():
    result = ToolInvocationResult(name="f", arguments={}, output=42, error="oops")
    assert result.text == "oops"


def test_result_describe_success():
    result = ToolInvocationResult(name="add", arguments={"a": 1}, output=2)
    assert "add" in result.describe()
    assert "2" in result.describe()
    assert "Error" not in result.describe()


def test_result_describe_error():
    result = ToolInvocationResult(name="add", arguments={}, error="Error: boom")
    desc = result.describe()
    assert "Error" in desc
    assert "boom" in desc


# ---------------------------------------------------------------------------
# invoke_tool tests
# ---------------------------------------------------------------------------


def test_invoke_tool_not_found_sets_error():
    """When tool is not found, error field should be set (not output)."""
    invocation = ToolInvocation(name="nonexistent", arguments={})
    result = invoke_tool(invocation, [_add])

    assert result.error is not None
    assert "not found" in result.error
    assert result.output is None


def test_invoke_tool_exception_sets_error():
    """When tool raises, error field should be set (not output)."""
    invocation = ToolInvocation(name="_always_fails", arguments={})
    result = invoke_tool(invocation, [_always_fails])

    assert result.error is not None
    assert "Simulated error" in result.error
    assert result.output is None


def test_invoke_tool_success_sets_output():
    """Successful tool call sets output, not error."""
    invocation = ToolInvocation(name="_add", arguments={"a": 3, "b": 4})
    result = invoke_tool(invocation, [_add])

    assert result.output == 7.0
    assert result.error is None


def test_invoke_tool_string_arguments_set_error():
    """When arguments is a string (from from_api_dict's JSONDecodeError
    fallback), invoke_tool surfaces an error rather than crashing."""
    invocation = ToolInvocation(name="_add", arguments='{"a": 1, "b":')
    result = invoke_tool(invocation, [_add])

    assert result.error is not None
    assert "arguments" in result.error.lower()
    assert result.output is None


# ---------------------------------------------------------------------------
# Tool invocation loop tests
# ---------------------------------------------------------------------------


def test_tool_called_and_result_returned():
    """The LLM requests a tool call, the tool runs, and the final answer
    uses the result."""
    tool_response = _make_tool_call_response("_add", {"a": 3, "b": 4})
    final_response = LLMMessage(sender=None, content="The answer is 7.")

    llm = MockedChat(responses=[tool_response, final_response])
    result = llm.prompt("What is 3 + 4?", tools=[_add])

    assert result == "The answer is 7."


def test_tool_not_called_when_no_tool_calls():
    """When the LLM doesn't request any tools, prompt() returns directly."""
    response = LLMMessage(sender=None, content="The answer is 42.")
    llm = MockedChat(responses=[response])

    result = llm.prompt("What is 42?", tools=[_add])
    assert result == "The answer is 42."
    assert len(llm.invocations) == 1  # Only one call, no loop


def test_tool_error_sent_back_to_llm():
    """When a tool raises an exception, the error message is sent back."""
    tool_response = _make_tool_call_response("_always_fails", {})
    final_response = LLMMessage(sender=None, content="The tool failed.")

    llm = MockedChat(responses=[tool_response, final_response])
    result = llm.prompt("Call the tool.", tools=[_always_fails])

    assert result == "The tool failed."


def test_extra_arguments_filtered():
    """Extra arguments from the API (like 'signature') are filtered out."""
    tool_response = _make_tool_call_response(
        "_add", {"a": 5, "b": 10, "signature": "extra_field"}
    )
    final_response = LLMMessage(sender=None, content="15")

    llm = MockedChat(responses=[tool_response, final_response])
    result = llm.prompt("5 + 10?", tools=[_add])

    # Should succeed despite the extra 'signature' argument.
    assert result == "15"


def test_assert_tool_was_invoked_in_forked_chat():
    """assert_tool_was_invoked finds tools in the forked subchat."""
    tool_response = _make_tool_call_response("_add", {"a": 1, "b": 2})
    final_response = LLMMessage(sender=None, content="3")

    llm = MockedChat(responses=[tool_response, final_response])
    llm.prompt("1+2?", tools=[_add])

    # The assertion should find the tool result in the nested fork.
    result = assertions.assert_tool_was_invoked(_add)
    assert result.passed


def test_assert_tool_was_invoked_by_name():
    """assert_tool_was_invoked accepts a string tool name."""
    tool_response = _make_tool_call_response("_add", {"a": 1, "b": 2})
    final_response = LLMMessage(sender=None, content="3")

    llm = MockedChat(responses=[tool_response, final_response])
    llm.prompt("1+2?", tools=[_add])

    assert assertions.assert_tool_was_invoked("_add").passed
    assert not assertions.assert_tool_was_invoked("_multiply").passed


def test_no_fork_without_tools():
    """When no tools are provided, prompt() doesn't fork the chat."""
    response = LLMMessage(sender=None, content="Hello.")
    llm = MockedChat(responses=[response])

    result = llm.prompt("Hi")
    assert result == "Hello."

    # No nested chats in history.
    chat = chats.get_current_chat()
    nested = [item for item in chat.history if isinstance(item, chats.Chat)]
    assert len(nested) == 0


def test_tool_invocation_limit_exhausted():
    """ToolInvocationLimitExhausted is raised when max rounds are exceeded."""
    tool_response = _make_tool_call_response("_add", {"a": 1, "b": 2})
    llm = MockedChat(responses=[tool_response], cycle=True)

    with pytest.raises(ToolInvocationLimitExhausted):
        llm.prompt("Keep calling tools.", tools=[_add])


def test_max_tool_rounds_parameter():
    """max_tool_rounds can be set to 1 to limit iterations."""
    tool_response = _make_tool_call_response("_add", {"a": 1, "b": 2})
    llm = MockedChat(responses=[tool_response], cycle=True)

    with pytest.raises(ToolInvocationLimitExhausted):
        llm.prompt(
            "Keep calling tools.",
            tools=[_add],
            extra_api_params={"max_tool_rounds": 1},
        )
    # With max_tool_rounds=1, only 1 invocation should have been made.
    assert len(llm.invocations) == 1


def test_none_arguments_handled():
    """When the model returns None arguments, the tool still executes."""
    tool_response = _make_tool_call_response("_no_args", None)
    final_response = LLMMessage(sender=None, content="done")

    llm = MockedChat(responses=[tool_response, final_response])
    result = llm.prompt("Call the tool.", tools=[_no_args])

    assert result == "done"


def test_multiple_tool_calls_in_single_response():
    """When the LLM requests multiple tools at once, all are invoked."""
    msg = LLMMessage(
        sender=None,
        content="",
        tool_calls=[
            ToolInvocation(name="_add", arguments={"a": 1, "b": 2}, call_id="call_1"),
            ToolInvocation(
                name="_multiply", arguments={"a": 3, "b": 4}, call_id="call_2"
            ),
        ],
    )
    final_response = LLMMessage(sender=None, content="1+2=3, 3*4=12")

    llm = MockedChat(responses=[msg, final_response])
    result = llm.prompt("Calculate both.", tools=[_add, _multiply])

    assert result == "1+2=3, 3*4=12"
    # 2 invocations: first with tool calls, second with final answer.
    assert len(llm.invocations) == 2


def test_tool_not_found_through_loop():
    """When the LLM calls a nonexistent tool, the error is sent back."""
    tool_response = _make_tool_call_response("nonexistent_tool", {"x": 1})
    final_response = LLMMessage(sender=None, content="That tool doesn't exist.")

    llm = MockedChat(responses=[tool_response, final_response])
    result = llm.prompt("Call the tool.", tools=[_add])

    assert result == "That tool doesn't exist."


class _StreamingTruncatedToolLLM(actors.LLMChat):
    """Streams a single tool-call chunk whose arguments JSON is truncated —
    exercises the full defensive chain: stream() → _finalize_tool_calls →
    from_api_dict's JSONDecodeError fallback → invoke_tool's str-arg branch."""

    def __init__(self):
        super().__init__(name="StreamingTruncated")
        self.stream_responses = True

    def invoke(self, messages, system=None, **kwargs):
        def chunks():
            yield LLMResponse(
                content="",
                tool_calls=[
                    SimpleNamespace(
                        index=0,
                        id="call_1",
                        function=SimpleNamespace(
                            name="_add",
                            arguments='{"a": 1, "b":',  # truncated
                        ),
                    )
                ],
            )

        return chunks()


def test_truncated_streaming_json_surfaces_clean_error_end_to_end():
    """End-to-end: a model that streams malformed tool-call JSON does not
    crash the tool loop. The truncated string is preserved through
    _finalize_tool_calls, invoke_tool produces a ToolInvocationResult with
    `error` set, and that error message lands in the chat history.

    Guards against regressions across three layers — from_api_dict,
    invoke_tool, and the streaming finalize — by exercising the full chain
    in one assertion."""
    llm = _StreamingTruncatedToolLLM()

    with chats.new("Truncated JSON loop") as chat:
        # max_tool_rounds=1 keeps the test tight; the model keeps returning
        # the same broken call so the loop exhausts after one round.
        with pytest.raises(ToolInvocationLimitExhausted):
            llm.prompt(
                "Compute 1+2",
                tools=[_add],
                extra_api_params={"max_tool_rounds": 1},
            )

        # The forked tool-loop chat hangs off the parent chat. Walk into it
        # to find the tool-result message with the parse error.
        tool_loop = next(item for item in chat.history if isinstance(item, chats.Chat))
        tool_results = [
            m for m in tool_loop.history if isinstance(m.content, ToolInvocationResult)
        ]
        assert len(tool_results) == 1
        result = tool_results[0].content
        assert result.error is not None
        assert "could not be parsed as JSON" in result.error
        # The raw truncated string must round-trip through the error, so an
        # operator can see exactly what the model emitted.
        assert '{"a": 1, "b":' in result.error


# ---------------------------------------------------------------------------
# Two-phase tool loop tests (schema + tools)
# ---------------------------------------------------------------------------


class _CityInfo(pydantic.BaseModel):
    """Test schema for structured output."""

    name: str
    population: int


def _get_city(city_name: str) -> dict:
    """Returns city data."""
    return {"name": city_name, "population": 1_000_000}


def test_schema_not_passed_during_tool_rounds():
    """During tool rounds, respond() should NOT receive schema=.
    The schema-only call happens as a separate final invocation."""
    tool_response = _make_tool_call_response("_get_city", {"city_name": "Berlin"})
    text_response = LLMMessage(sender=None, content="Berlin has 1M people.")
    schema_response = LLMMessage(
        sender=None, content='{"name": "Berlin", "population": 1000000}'
    )

    llm = MockedChat(
        responses=[tool_response, text_response, schema_response],
        support_structured_outputs=True,
    )
    result = llm.prompt("Look up Berlin.", tools=[_get_city], schema=_CityInfo)

    assert isinstance(result, _CityInfo)
    # 3 invocations: tool call, text answer (no schema), schema-only call.
    assert len(llm.invocations) == 3

    # Tool rounds (invocations 0 and 1): should NOT have response_format.
    _, kwargs_round1 = llm.invocations[0]
    assert "response_format" not in kwargs_round1

    _, kwargs_round2 = llm.invocations[1]
    assert "response_format" not in kwargs_round2

    # Final call (invocation 2): should have response_format for the schema.
    _, kwargs_round3 = llm.invocations[2]
    assert "response_format" in kwargs_round3
    assert kwargs_round3["response_format"] == _CityInfo


def test_no_extra_call_when_schema_is_str():
    """When schema=str (default), no extra schema-only call is made."""
    tool_response = _make_tool_call_response("_get_city", {"city_name": "Berlin"})
    final_response = LLMMessage(sender=None, content="Berlin has 1M people.")

    llm = MockedChat(responses=[tool_response, final_response])
    result = llm.prompt("Look up Berlin.", tools=[_get_city])

    assert result == "Berlin has 1M people."
    # Only 2 invocations: tool call + final text answer. No extra schema call.
    assert len(llm.invocations) == 2


def test_schema_call_sees_tool_history():
    """The schema-only call should see the tool conversation history."""
    tool_response = _make_tool_call_response("_get_city", {"city_name": "Berlin"})
    text_response = LLMMessage(sender=None, content="Berlin has 1M people.")
    schema_response = LLMMessage(
        sender=None, content='{"name": "Berlin", "population": 1000000}'
    )

    llm = MockedChat(
        responses=[tool_response, text_response, schema_response],
        support_structured_outputs=True,
    )
    llm.prompt("Look up Berlin.", tools=[_get_city], schema=_CityInfo)

    # The third invocation (schema call) should include messages from the
    # tool loop — at minimum the tool result and the text answer.
    messages_round3, _ = llm.invocations[2]
    # Should have more messages than the first round (user prompt only)
    # because tool results and LLM responses were appended.
    messages_round1, _ = llm.invocations[0]
    assert len(messages_round3) > len(messages_round1)
