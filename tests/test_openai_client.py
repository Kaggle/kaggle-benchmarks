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


@dataclass
class MockFunction:
    name: str
    arguments: str


@dataclass
class MockToolCall:
    id: str
    function: MockFunction
    type: str = "function"


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
        tool_call = MockToolCall(
            id="call_123",
            function=MockFunction(name="calculator", arguments='{"a": 1, "b": 2}'),
        )
        return LLMResponse(content="", tool_calls=[tool_call])


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


@pytest.mark.parametrize(
    "model",
    ["google/gemini-2.5-flash", "anthropic/claude-sonnet", "openai/gpt-5.4"],
)
def test_prompt_reasoning_sets_effort_and_thinking_config(model):
    """Tests that reasoning sets reasoning_effort and include_thoughts for all models."""
    llm = MockedOpenAI(model=model)
    llm.prompt("Think hard", reasoning="high")

    assert llm.kwargs["reasoning_effort"] == "high"
    extra = llm.kwargs["extra_body"]["extra_body"]["google"]["thinking_config"]
    assert extra["include_thoughts"] is True


def test_prompt_forwards_api_params():
    llm = MockedOpenAI(model="test-model")
    llm.prompt("Hi", api_params={"max_tokens": 500})
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


def test_prompt_rejects_reasoning_effort_in_api_params():
    llm = MockedOpenAI(model="test-model")
    with pytest.raises(ValueError, match="reasoning_effort.*not allowed in api_params"):
        llm.prompt("Hi", api_params={"reasoning_effort": "high"})


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
    assert response_msg.tool_calls[0].function.name == "calculator"
    assert response_msg.tool_calls[0].function.arguments == '{"a": 1, "b": 2}'


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

    final_call_obj = MockToolCallDelta(index=0, **final_tool_calls[0])
    final_call_obj.function = MockFunctionDelta(**final_call_obj.function)

    assert final_call_obj.id == "call_123"
    assert final_call_obj.function.name == "calculator"
    assert final_call_obj.function.arguments == '{"a": 5, "b": 10}'


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

    calculator_call = final_tool_calls[0]
    assert calculator_call["id"] == "call_calc_123"
    assert calculator_call["function"]["name"] == "calculator"
    assert calculator_call["function"]["arguments"] == '{"a": 100, "b": 200}'

    weather_call = final_tool_calls[1]
    assert weather_call["id"] == "call_weather_456"
    assert weather_call["function"]["name"] == "get_weather"
    assert weather_call["function"]["arguments"] == '{"city": "NYC"}'
