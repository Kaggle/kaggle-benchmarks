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
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from kaggle_benchmarks import actors, chats
from kaggle_benchmarks.actors.llms import LLMResponse, OpenAI, _parse_think_tags
from kaggle_benchmarks.prompting import handler


class MockedOpenAI(OpenAI):
    def __init__(self, model: str, **kwargs):
        super().__init__(client=None, model=model, **kwargs)
        self.support_temperature = False

    def _call_api(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return LLMResponse(content="{}")


@dataclass
class MockFunctionDelta:
    name: str | None = None
    arguments: str | None = None


@dataclass
class MockToolCallDelta:
    index: int
    id: str | None = None
    function: MockFunctionDelta | None = None
    type: str | None = "function"


class MockedOpenAIWithTokens(OpenAI):
    def __init__(self, **kwargs):
        # We pass a dummy client, as it's not used in the mocked _call_api
        super().__init__(client=None, model="mock_with_tokens", **kwargs)

    def _call_api(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs

        if self.stream_responses:

            def stream_generator():
                yield LLMResponse(content="stream", meta={"input_tokens": 10})
                yield LLMResponse(
                    content="ing", meta={"input_tokens": 10, "output_tokens": 5}
                )

            return stream_generator()
        return LLMResponse(
            content="non-streaming", meta={"input_tokens": 20, "output_tokens": 2}
        )


class MockedOpenAIWithToolCall(OpenAI):
    def __init__(self, **kwargs):
        super().__init__(client=None, model="mock-tool-caller", **kwargs)

    def _call_api(self, messages, **kwargs):
        # Mirrors real _call_api shape: dicts (post-model_dump), not SDK objects.
        return LLMResponse(
            content="",
            tool_calls=[
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"a": 1, "b": 2}',
                    },
                }
            ],
        )


class MockedOpenAIWithStreamingToolCall(OpenAI):
    def __init__(self, **kwargs):
        super().__init__(client=None, model="mock-streaming-tool-caller", **kwargs)

    def _call_api(self, messages, **kwargs):
        def stream_generator():
            yield LLMResponse(
                content="",
                tool_calls=[
                    MockToolCallDelta(
                        index=0,
                        id="call_123",
                        function=MockFunctionDelta(name="calculator"),
                    )
                ],
            )
            yield LLMResponse(
                content="",
                tool_calls=[
                    MockToolCallDelta(
                        index=0, function=MockFunctionDelta(arguments='{"a": 5,')
                    )
                ],
            )
            yield LLMResponse(
                content="Okay, ",
                tool_calls=[
                    MockToolCallDelta(
                        index=0, function=MockFunctionDelta(arguments=' "b": 10}')
                    )
                ],
            )
            yield LLMResponse(content="calculating...")

        return stream_generator()


class MockedOpenAIWithMultipleStreamingToolCalls(OpenAI):
    def __init__(self, **kwargs):
        super().__init__(
            client=None, model="mock-multi-streaming-tool-caller", **kwargs
        )

    def _call_api(self, messages, **kwargs):
        def stream_generator():
            yield LLMResponse(
                content="",
                tool_calls=[
                    MockToolCallDelta(
                        index=0,
                        id="call_calc_123",
                        function=MockFunctionDelta(name="calculator"),
                    )
                ],
            )
            yield LLMResponse(
                content="",
                tool_calls=[
                    MockToolCallDelta(
                        index=1,
                        id="call_weather_456",
                        function=MockFunctionDelta(name="get_weather"),
                    )
                ],
            )
            yield LLMResponse(
                content="",
                tool_calls=[
                    MockToolCallDelta(
                        index=0, function=MockFunctionDelta(arguments='{"a": 100,')
                    )
                ],
            )
            yield LLMResponse(
                content="",
                tool_calls=[
                    MockToolCallDelta(
                        index=1, function=MockFunctionDelta(arguments='{"city": "NYC"}')
                    )
                ],
            )
            yield LLMResponse(
                content="Okay, ",
                tool_calls=[
                    MockToolCallDelta(
                        index=0, function=MockFunctionDelta(arguments=' "b": 200}')
                    )
                ],
            )
            yield LLMResponse(content="processing requests...")

        return stream_generator()


def test_invoke():
    llm = MockedOpenAI(model="test-model")
    llm.prompt("Hi")
    assert llm.messages == [{"role": "user", "content": "Hi"}]
    assert llm.kwargs.get("response_format") is None


def test_prompt_reasoning_sets_effort_and_thinking_config():
    """Tests that reasoning sets reasoning_effort and thinking_config for Google models."""
    llm = MockedOpenAI(model="google/gemini-2.5-flash")
    llm.prompt("Think hard", reasoning="high")

    assert llm.kwargs["reasoning_effort"] == "high"
    extra = llm.kwargs["extra_body"]["extra_body"]["google"]["thinking_config"]
    assert extra["include_thoughts"] is True


def test_prompt_forwards_extra_api_params():
    llm = MockedOpenAI(model="test-model")
    llm.prompt("Hi", extra_api_params={"max_tokens": 500})
    assert llm.kwargs["max_tokens"] == 500


def test_reasoning_content_captured_in_response(mocker):
    """Tests that reasoning_content is captured as reasoning_traces.

    When reasoning_content is set, <think> tags in content should remain
    untouched since reasoning_content takes priority.
    """
    mock_client = mocker.MagicMock()
    mock_message = mocker.MagicMock()
    mock_message.content = "<think>\nstale\n</think>\n\nThere are 3 r's."
    mock_message.reasoning_content = (
        "Let me count: s-t-r-a-w-b-e-r-r-y. r appears at positions 3, 8, 9."
    )
    mock_message.tool_calls = None

    mock_usage = mocker.MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5

    mock_choice = mocker.MagicMock()
    mock_choice.message = mock_message

    mock_response = mocker.MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    mock_client.chat.completions.create.return_value = mock_response

    llm = OpenAI(client=mock_client, model="test-model")
    llm.stream_responses = False

    with chats.new("Test reasoning") as t:
        response = llm.prompt("How many r's in strawberry?")

    # Content should preserve <think> tags since reasoning_content takes priority.
    assert response == "<think>\nstale\n</think>\n\nThere are 3 r's."
    last_message = t.messages[-1]
    assert last_message.reasoning_traces == (
        "Let me count: s-t-r-a-w-b-e-r-r-y. r appears at positions 3, 8, 9."
    )


def test_reasoning_plumbed_through_respond():
    """Tests that LLMResponse.reasoning_traces is plumbed through respond() to the message."""

    class MockedOpenAIWithReasoning(OpenAI):
        def __init__(self):
            super().__init__(client=None, model="mock-reasoning")

        def _call_api(self, messages, **kwargs):
            return LLMResponse(
                content="The answer is 42.",
                reasoning_traces="I need to think about this carefully...",
            )

    llm = MockedOpenAIWithReasoning()

    with chats.new("Test reasoning plumbing") as t:
        response = llm.prompt("What is the answer?")

    assert response == "The answer is 42."
    last_message = t.messages[-1]
    assert last_message.reasoning_traces == "I need to think about this carefully..."


def test_last_reasoning_traces_accessor():
    """Tests that kbench.last_reasoning_traces() returns reasoning from the last message."""

    class MockedOpenAIWithReasoning(OpenAI):
        def __init__(self):
            super().__init__(client=None, model="mock-reasoning")

        def _call_api(self, messages, **kwargs):
            return LLMResponse(
                content="The answer is 42.",
                reasoning_traces="I need to think about this carefully...",
            )

    llm = MockedOpenAIWithReasoning()

    with chats.new("Test last_reasoning_traces"):
        llm.prompt("What is the answer?")
        assert (
            chats.last_reasoning_traces() == "I need to think about this carefully..."
        )


def test_last_reasoning_traces_returns_none_without_reasoning():
    """Tests that kbench.last_reasoning_traces() returns None when no reasoning traces exist."""
    llm = MockedOpenAI(model="test-model")

    with chats.new("Test no reasoning"):
        llm.prompt("Hi")
        assert chats.last_reasoning_traces() is None


def test_parse_think_tags_extracts_traces():
    """Tests that <think> tags in content are parsed into reasoning_traces."""
    content = "<think>\nLet me think step by step.\n</think>\n\nThe answer is 42."
    remaining, thinking = _parse_think_tags(content)
    assert remaining == "The answer is 42."
    assert thinking == "Let me think step by step."


def test_parse_think_tags_returns_none_without_tags():
    """Tests that content without <think> tags returns None for thinking."""
    content = "The answer is 42."
    remaining, thinking = _parse_think_tags(content)
    assert remaining == "The answer is 42."
    assert thinking is None


def test_parse_think_tags_extracts_multiple_blocks():
    """Tests that multiple <think> blocks are all extracted and joined."""
    content = (
        "<think>\nFirst thought.\n</think>\n\n"
        "Middle content.\n\n"
        "<think>\nSecond thought.\n</think>\n\n"
        "The answer is 42."
    )
    remaining, thinking = _parse_think_tags(content)
    assert remaining == "Middle content.\n\nThe answer is 42."
    assert thinking == "First thought.\n\nSecond thought."


def test_parse_think_tags_empty_block():
    """Tests that malformed empty <think></think> returns None for thinking."""
    content = "<think></think>\n\nThe answer is 42."
    remaining, thinking = _parse_think_tags(content)
    assert remaining == "The answer is 42."
    assert thinking is None


def test_think_tags_captured_in_response(mocker):
    """Tests that Model Proxy <think> tags are parsed into reasoning_traces."""
    mock_client = mocker.MagicMock()
    mock_message = mocker.MagicMock()
    mock_message.content = (
        "<think>\nCounting the letters...\n</think>\n\nThere are 3 r's."
    )
    mock_message.reasoning_content = None
    mock_message.tool_calls = None

    mock_usage = mocker.MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5

    mock_choice = mocker.MagicMock()
    mock_choice.message = mock_message

    mock_response = mocker.MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    mock_client.chat.completions.create.return_value = mock_response

    llm = OpenAI(client=mock_client, model="google/gemini-2.5-flash")
    llm.stream_responses = False

    with chats.new("Test think tags") as t:
        response = llm.prompt("How many r's in strawberry?", reasoning="high")

    assert response == "There are 3 r's."
    last_message = t.messages[-1]
    assert last_message.reasoning_traces == "Counting the letters..."


def test_extra_api_params_rejects_sdk_params():
    """Tests that extra_api_params rejects params already on prompt()/respond() signatures."""
    llm = MockedOpenAI(model="test-model")
    with pytest.raises(ValueError, match="cannot be set via extra_api_params"):
        llm.prompt("Hi", seed=42, extra_api_params={"seed": 99})


def test_call_api_handles_none_message(mocker, caplog):
    """Empty response when choices[0].message is None (see issue #191).

    Model Proxy has been observed to intermittently return a response where
    choices[0].message is None. The library must not raise AttributeError
    on message.tool_calls; it should degrade to an empty LLMResponse so the
    native tool loop can terminate cleanly.
    """
    mock_client = mocker.MagicMock()

    mock_usage = mocker.MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 0

    mock_choice = mocker.MagicMock()
    mock_choice.message = None

    mock_response = mocker.MagicMock(spec=[])
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    mock_client.chat.completions.create.return_value = mock_response

    llm = OpenAI(client=mock_client, model="anthropic/claude-opus-5@default")
    llm.stream_responses = False

    with chats.new("Test None message"), caplog.at_level("WARNING"):
        response = llm.prompt("Hi")

    assert response == ""
    assert any("choices[0].message=None" in record.message for record in caplog.records)


def test_call_api_handles_empty_choices(mocker):
    """Empty response when choices is empty (existing guard, now covered by a test)."""
    mock_client = mocker.MagicMock()

    mock_usage = mocker.MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 0

    mock_response = mocker.MagicMock(spec=[])
    mock_response.choices = []
    mock_response.usage = mock_usage

    mock_client.chat.completions.create.return_value = mock_response

    llm = OpenAI(client=mock_client, model="anthropic/claude-opus-5@default")
    llm.stream_responses = False

    with chats.new("Test empty choices"):
        response = llm.prompt("Hi")

    assert response == ""


def test_prompt_reasoning_none_sets_effort_without_thinking_config():
    """Tests that reasoning='none' sets reasoning_effort but skips thinking_config."""
    llm = MockedOpenAI(model="test-model")
    llm.prompt("Hi", reasoning="none")

    assert llm.kwargs["reasoning_effort"] == "none"
    assert "extra_body" not in llm.kwargs


def test_reasoning_extra_body_only_for_google_models():
    """Tests that google thinking_config extra_body is only added for google/ models."""
    google_llm = MockedOpenAI(model="google/gemini-2.5-flash")
    google_llm.prompt("Hi", reasoning="high")
    assert "extra_body" in google_llm.kwargs

    openai_llm = MockedOpenAI(model="openai/gpt-5.4-2026-03-05")
    openai_llm.prompt("Hi", reasoning="high")
    assert "extra_body" not in openai_llm.kwargs
    assert openai_llm.kwargs["reasoning_effort"] == "high"

    anthropic_llm = MockedOpenAI(model="anthropic/claude-sonnet-4-6@default")
    anthropic_llm.prompt("Hi", reasoning="high")
    assert "extra_body" not in anthropic_llm.kwargs
    assert anthropic_llm.kwargs["reasoning_effort"] == "high"


@pytest.mark.parametrize(
    "model,expected",
    [
        ("google/gemini-2.5-flash", True),
        ("openai/gpt-5.4-pro", True),
        ("openai/gpt-5.6", True),
        ("xai/grok-4.5", True),
        ("xai/grok-4.6", True),
        ("xai/grok-4.9-fast", True),
        ("xai/grok-4.10", True),
        ("xai/grok-4.42", True),
        ("xai/grok-5", True),
        ("xai/grok-7.1", True),
        ("xai/grok-10", True),
        ("xai/grok-10.0", True),
        ("xai/grok-4.4", False),
        ("xai/grok-4.0", False),
        ("xai/grok-4", False),
        ("xai/grok-3", False),
        ("xai/grok-3.9", False),
        ("xai/grok-beta", False),
        ("openai/gpt-5.5", False),
        ("anthropic/claude-sonnet-4-6@default", False),
    ],
)
def test_should_remove_seed(model, expected):
    assert MockedOpenAI(model=model)._should_remove_seed() is expected


def test_invoke_prompt():
    llm = MockedOpenAI(model="test-model")
    llm.support_structured_outputs = False

    class A:
        pass

    @handler(types=A)
    def type_a(_):
        v = yield "A"
        return v

    llm.prompt("Hi", schema=A)

    assert llm.messages == [
        {"role": "user", "content": "Hi"},
        {"role": "system", "content": "A"},
    ]
    assert llm.kwargs.get("response_format") is None


def test_pydantic_models():
    class Model(BaseModel):
        a: str = "a"
        b: int = 0

    llm = MockedOpenAI(model="test-model")
    llm.support_structured_outputs = True
    llm.prompt("Hi", schema=Model)
    assert llm.messages == [
        {"role": "user", "content": "Hi"},
    ]
    assert llm.kwargs.get("response_format") is Model

    llm.support_structured_outputs = False
    llm.prompt("Hi", schema=Model)
    assert llm.messages[0] == {"role": "user", "content": "Hi"}
    assert json.dumps(Model.model_json_schema()) in llm.messages[1]["content"]
    assert llm.kwargs.get("response_format") is None


@pytest.mark.parametrize("streaming", [True, False])
def test_invoke_with_token_counts(streaming):
    llm = MockedOpenAIWithTokens()
    llm.stream_responses = streaming

    with chats.new("Test Tokens") as t:
        response_content = llm.prompt("count my tokens")

        last_message = t.messages[-1]
        assert last_message.sender is llm
        if streaming:
            assert response_content == "streaming"
            assert last_message._meta["input_tokens"] == 10
            assert last_message._meta["output_tokens"] == 5
        else:
            assert response_content == "non-streaming"
            assert last_message._meta["input_tokens"] == 20
            assert last_message._meta["output_tokens"] == 2


def test_llm_extracts_tool_calls():
    llm = MockedOpenAIWithToolCall()

    with chats.new("test tools"):
        actors.user.send("call a tool")
        response_msg = llm.respond()

    assert response_msg.tool_calls is not None
    assert len(response_msg.tool_calls) == 1
    [tc] = response_msg.tool_calls
    assert tc.name == "calculator"
    assert tc.arguments == {"a": 1, "b": 2}
    assert tc.call_id == "call_123"


def test_streaming_accumulates_tool_calls():
    llm = MockedOpenAIWithStreamingToolCall()
    llm.stream_responses = True

    with chats.new("test streaming tools"):
        actors.user.send("What is 5 + 10?")
        response_msg = llm.respond()

    assert response_msg.content == "Okay, calculating..."

    final_tool_calls = response_msg.tool_calls
    assert final_tool_calls is not None
    assert len(final_tool_calls) == 1
    [tc] = final_tool_calls
    assert tc.call_id == "call_123"
    assert tc.name == "calculator"
    assert tc.arguments == {"a": 5, "b": 10}


def test_streaming_accumulates_multiple_tool_calls():
    llm = MockedOpenAIWithMultipleStreamingToolCalls()
    llm.stream_responses = True

    with chats.new("test multi-streaming tools"):
        actors.user.send("What is 100 + 200 and the weather in NYC?")
        response_msg = llm.respond()

    assert response_msg.content == "Okay, processing requests..."

    final_tool_calls = response_msg.tool_calls
    assert final_tool_calls is not None
    assert len(final_tool_calls) == 2

    calculator_call, weather_call = final_tool_calls
    assert calculator_call.call_id == "call_calc_123"
    assert calculator_call.name == "calculator"
    assert calculator_call.arguments == {"a": 100, "b": 200}

    assert weather_call.call_id == "call_weather_456"
    assert weather_call.name == "get_weather"
    assert weather_call.arguments == {"city": "NYC"}
