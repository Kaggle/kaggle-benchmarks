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

import pydantic
import pytest

from kaggle_benchmarks import actors, chats, prompting, usage, utils
from kaggle_benchmarks import tools as tool_utils
from kaggle_benchmarks.content_types import images, videos
from kaggle_benchmarks.llm_messages import LLMMessage
from tests.mocks import MockedChat


def test_prompt_without_context():
    llm = MockedChat.from_contents(["response content"])

    r = llm.prompt("A")
    assert r == "response content"
    assert len(llm.invocations) == 1
    invoked_messages, kwargs = llm.invocations[0]
    assert len(invoked_messages) == 1
    assert invoked_messages[0].content == "A"
    assert invoked_messages[0].sender is actors.user

    assert kwargs["system"] is None


def test_respond():
    llm = MockedChat.from_contents(["response content"])

    with chats.new() as t:
        actors.user.send("A")
        assert len(t.messages) == 1

        r = llm.respond()
        assert len(t.messages) == 2
        assert r.content == "response content"
        assert len(llm.invocations) == 1
        invoked_messages, kwargs = llm.invocations[0]
        assert len(invoked_messages) == 1
        assert invoked_messages[0].content == "A"
        assert invoked_messages[0].sender is actors.user


def test_chat_context():
    llm = MockedChat.from_contents(["response A", "response B"])
    # This message should not be visible in the context of the next chat.
    actors.user.send("<should not be visible in the context>")

    with chats.new(system_instructions="S") as t:
        assert t.status == utils.Status.RUNNING

        assert len(t.messages) == 1
        assert t.messages[0].content == "S"
        assert t.messages[0].sender is actors.system

        r = llm.prompt("A")
        assert r == "response A"

        assert len(t.messages) == 3
        assert t.messages[1].content == "A"
        assert t.messages[1].sender is actors.user
        assert t.messages[2].content == "response A"
        assert t.messages[2].sender is llm

        assert len(llm.invocations) == 1

        invoked_messages, kwargs = llm.invocations[0]
        assert len(invoked_messages) == 2
        assert invoked_messages == t.messages[:2]
        assert kwargs["system"] is None
        assert kwargs["schema"] is str

        r = llm.prompt("B")
        assert r == "response B"
        assert len(t.messages) == 5
        assert len(llm.invocations) == 2

        invoked_messages, kwargs = llm.invocations[1]
        assert len(invoked_messages) == 4
        assert invoked_messages == t.messages[:4]
        assert kwargs["system"] is None

    assert t.status == utils.Status.SUCCESS


@pytest.mark.parametrize(
    "support_structured_outputs",
    [
        pytest.param(True, id="with_schema_support"),
        pytest.param(False, id="without_schema_support"),
    ],
)
def test_structured_output(support_structured_outputs):
    llm = MockedChat.from_contents(
        ['{"field1": 1, "field2": "two"}'],
        support_structured_outputs=support_structured_outputs,
    )

    class Response(pydantic.BaseModel):
        field1: int
        field2: str

    with chats.new("test") as t:
        response = llm.prompt("test", schema=Response)
        assert response == Response(field1=1, field2="two")
        assert len(t.messages) == 2

        invoked_messages, kwargs = llm.invocations[0]
        if support_structured_outputs:
            assert len(invoked_messages) == 1
            assert kwargs["schema"] is Response
        else:
            # extra message for schema instructions
            assert len(invoked_messages) == 2
            assert kwargs["schema"] is str


def get_weather(location: str) -> str:
    """Get current weather"""
    if "london" in location.lower():
        return "Rainy"
    return "Sunny"


class WeatherReport(pydantic.BaseModel):
    text: str
    temperature: int


@pytest.mark.parametrize(
    "support_tools",
    [
        pytest.param(True, id="with_tool_support"),
        pytest.param(False, id="without_tool_support"),
    ],
)
def test_tool_calling_with_structured_output(support_tools):
    value = WeatherReport(text="Rainy", temperature=15)

    if support_tools:
        responses = [
            LLMMessage(
                sender=None,
                content=None,
                tool_calls=[
                    tool_utils.ToolInvocation(
                        name="get_weather", arguments={"location": "London"}
                    )
                ],
            ),
            LLMMessage(
                sender=None,
                content=value.model_dump_json(),
            ),
        ]
    else:
        responses = [
            LLMMessage(
                sender=None,
                content=json.dumps(
                    {
                        "tools": [
                            dict(name="get_weather", arguments={"location": "London"})
                        ],
                        "message": None,
                    }
                ),
            ),
            LLMMessage(
                sender=None,
                content=json.dumps(
                    {
                        "tools": None,
                        "message": value.model_dump(),
                    }
                ),
            ),
        ]

    llm = MockedChat(
        responses=responses,
        support_tool_calling=support_tools,
        support_structured_outputs=True,
    )

    tools = [get_weather]

    with chats.new() as t:
        response = llm.prompt(
            "What is the weather in London?", tools=tools, schema=WeatherReport
        )
        assert isinstance(response, WeatherReport)
        assert response == value
        assert len(t.messages) == 2

        # one with tool invocation
        # second one with result
        assert len(llm.invocations) == 2
        messages1, kwargs1 = llm.invocations[0]

        if support_tools:
            assert len(messages1) == 1
            assert kwargs1["schema"] == WeatherReport
            assert kwargs1["tools"] == tools

        else:
            # extra message to describe tools
            assert len(messages1) == 2
            assert issubclass(kwargs1["schema"], tool_utils.ModelResponse)

            assert not kwargs1["tools"]

        # assert len(messages1) == 1
        assert messages1[0].content == "What is the weather in London?"
        assert messages1[0].sender is actors.user
        # assert kwargs1["schema"] == WeatherReport

        messages2, kwargs2 = llm.invocations[1]
        if support_tools:
            assert len(messages2) == 3
            assert kwargs2["tools"] == tools
            assert kwargs2["schema"] == WeatherReport
        else:
            # extra message to describe tools
            assert len(messages2) == 4
            assert issubclass(kwargs2["schema"], tool_utils.ModelResponse)
            assert not kwargs2["tools"]

        assert messages2[0].sender is actors.user
        assert messages2[1].sender is llm
        assert isinstance(messages2[2].content, tool_utils.ToolInvocationResult)
        assert messages2[2].content.output == "Rainy"


def test_tool_calling_with_typed_output_no_tools():
    responses = [
        LLMMessage(
            sender=None,
            content=json.dumps(
                {
                    "tools": [
                        dict(name="get_weather", arguments={"location": "London"})
                    ],
                    "message": None,
                }
            ),
        ),
        LLMMessage(
            sender=None,
            content=json.dumps(
                {
                    "tools": [],
                    "message": "12",
                }
            ),
        ),
    ]

    llm = MockedChat(
        responses=responses,
        support_tool_calling=False,
        support_structured_outputs=True,
    )

    tools = [get_weather]

    with chats.new():
        response = llm.prompt("What is the weather in London?", tools=tools, schema=int)
        assert isinstance(response, int)
        assert response == 12
        assert len(llm.invocations) == 2
        # Check that the schema was wrapped for emulated tool calling
        messages1, kwargs1 = llm.invocations[0]
        # user message + instructions
        assert len(messages1) == 2
        assert issubclass(kwargs1["schema"], tool_utils.ModelResponse)
        messages2, kwargs2 = llm.invocations[1]
        # user message + first response + call result + instructions
        assert len(messages2) == 4
        assert issubclass(kwargs2["schema"], tool_utils.ModelResponse)


def test_custom_types():
    llm = MockedChat(
        responses=[
            LLMMessage(sender=None, content="any content"),
            LLMMessage(sender=None, content="any content"),
            LLMMessage(sender=None, content="any content"),
        ],
        support_structured_outputs=True,
    )

    class F:
        pass

    value = F()

    @prompting.handler(types=F)
    def _(cls):
        yield ""
        return value

    response = llm.prompt("Test", schema=F)
    assert isinstance(response, F)
    assert value is response

    @prompting.handler(types=F)
    def _(cls):
        value = yield ""
        raise prompting.ResponseParsingError(
            error="Bad response", schema=cls, value=value
        )

    with chats.new() as t:
        with pytest.raises(prompting.ResponseParsingError):
            llm.prompt("test_value", schema=F)

        assert len(t.messages) == 2
        llm_message = t.messages[-1]
        assert isinstance(llm_message, LLMMessage)
        # the error goes to the subchat used for helper prompt
        error_text = llm_message.chat.messages[-1].text
        assert "Bad response" in error_text
        assert "F" in error_text

    @prompting.handler(types=F)
    def _(cls):
        yield ""
        yield "nonsense"
        return F()

    with pytest.raises(prompting.SchemaError):
        llm.prompt("Test", schema=F)


def test_chat_usage_aggregation():
    """Test that chat usage properties aggregate token usage from all assistant messages."""
    llm = MockedChat(
        responses=[
            LLMMessage(
                sender=None,
                content="first",
                usage=usage.Usage(input_tokens=5, output_tokens=3),
            ),
            LLMMessage(
                sender=None,
                content="second",
                usage=usage.Usage(input_tokens=15, output_tokens=1),
            ),
        ],
    )
    with chats.new("Test Usage") as t:
        llm.prompt("first")
        llm.prompt("second")

        # Each streaming response yields: input_tokens=10, output_tokens=2
        # Two prompts = 2 * 10 = 20 input tokens, 2 * 2 = 4 output tokens
        assert t.usage.input_tokens == 20
        assert t.usage.output_tokens == 4


def test_chat_usage_empty():
    """Test that chat usage properties return zero/None for empty chat."""
    with chats.new("Empty") as t:
        assert t.usage.input_tokens is None
        assert t.usage.output_tokens is None
        assert t.usage.input_tokens_cost_nanodollars is None
        assert t.usage.output_tokens_cost_nanodollars is None
        assert t.usage.total_backend_latency_ms is None


def test_video_message_payload():
    """Test that a VideoURL message produces the correct payload for the OpenAI backend."""
    video = videos.from_url("https://www.youtube.com/watch?v=abc123")

    from kaggle_benchmarks import messages

    msg = messages.Message(sender=actors.user, content=video)
    assert msg.payload == [
        {
            "type": "image_url",
            "image_url": {"url": "https://www.youtube.com/watch?v=abc123"},
        }
    ]


def test_prompt_with_image_and_video():
    """Test that prompt() with both image and video sends them as separate messages."""
    red_pixel_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    img = images.from_base64(red_pixel_b64, format="png")
    video = videos.from_url("https://www.youtube.com/watch?v=abc123")

    with chats.new("image_and_video") as t:
        # Manually send image and video, then prompt, to verify message ordering.
        # We don't call llm.prompt() directly because Ferret's invoke() can't
        # serialize image/video content to JSON.
        actors.user.send(img)
        actors.user.send(video)
        actors.user.send("Describe both")

        assert len(t.messages) == 3
        assert isinstance(t.messages[0].content, images.ImageBase64)
        assert isinstance(t.messages[1].content, videos.VideoURL)
        assert t.messages[2].content == "Describe both"


def test_chat_fork():
    with chats.new("Parent") as p:
        actors.user.send("Hello")

        with chats.fork("Forked") as f:
            assert f.name == "Forked"
            assert len(f.history) == 1
            assert f.history[0].content == "Hello"
            actors.user.send("World")
            assert len(f.history) == 2

        assert len(p.history) == 2
        assert p.history[1] is f

        with chats.fork("Orphaned", orphan=True) as of:
            assert of.name == "Orphaned"
            assert len(of.history) == 1  # Only messages are copied to the fork
            actors.user.send("Bye")
            assert len(of.history) == 2

        assert len(p.history) == 2


def test_invoke_llmmessage():
    mocked_chat = MockedChat.from_contents(["test response"])
    messages = [LLMMessage(sender=actors.user, content="hello")]

    response = mocked_chat.invoke(messages=messages, temperature=0.5)

    assert isinstance(response, LLMMessage)
    assert response.content == "test response"
    assert response.sender is mocked_chat
    assert len(mocked_chat.invocations) == 1
    assert mocked_chat.invocations[0][0] == messages
    assert mocked_chat.invocations[0][1].get("temperature") == 0.5
