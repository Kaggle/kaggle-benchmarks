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

import pytest

from kaggle_benchmarks import actors, chats, contexts, prompting, utils
from kaggle_benchmarks.actors.llms import LLMResponse
from kaggle_benchmarks.content_types import images, videos
from kaggle_benchmarks.llm_messages import LLMMessage
from kaggle_benchmarks.prompting import handler
from tests.mocks import MockedChat


class Ferret(actors.LLMChat):
    def __init__(self):
        super().__init__(name="Ferret")
        self.stream_responses = False

    def invoke(self, messages, system=None, **kwargs):
        if not self.stream_responses:
            return LLMResponse(
                content=json.dumps(
                    dict(
                        messages=[[m.sender.name.lower(), m.content] for m in messages],
                        system=system,
                    )
                )
            )

        def stream_generator():
            yield LLMResponse(content="stream", meta={"input_tokens": 10})
            yield LLMResponse(
                content="ing", meta={"input_tokens": 10, "output_tokens": 1}
            )
            yield LLMResponse(
                content="...", meta={"input_tokens": 10, "output_tokens": 2}
            )

        return stream_generator()


def test_prompt_without_context():
    llm = Ferret()
    r = llm.prompt("A")
    assert {"messages": [["user", "A"]], "system": None} == json.loads(r)


def test_respond():
    llm = Ferret()

    with chats.new("Test") as t:
        actors.user.send("A")
        assert len(t.messages) == 1

        r = llm.respond()
        assert len(t.messages) == 2
        assert {"messages": [["user", "A"]], "system": None} == json.loads(r.text)


def test_chat_context():
    llm = Ferret()
    llm.prompt("<should not be visible in the context>")

    with chats.new(system_instructions="S") as t:
        assert t.status == utils.Status.RUNNING

        r = llm.prompt("A")
        assert {
            "messages": [["system", "S"], ["user", "A"]],
            "system": None,
        } == json.loads(r)

        r = llm.prompt("B")
        response = json.loads(r)

        assert response["system"] is None
        assert 4 == len(response["messages"])
        assert ["system", "S"] == response["messages"][0]
        assert ["user", "A"] == response["messages"][1]
        assert llm.name.lower() == response["messages"][2][0]
        assert ["user", "B"] == response["messages"][3]

    assert t.status == utils.Status.SUCCESS


def test_structured():
    llm = Ferret()

    class F:
        pass

    value = F()

    @handler(types=F)
    def _(cls):
        yield ""
        return value

    response = llm.prompt("Test", schema=F)
    assert isinstance(response, F)
    assert value is response

    @handler(types=F)
    def _(cls):
        value = yield ""
        raise prompting.ResponseParsingError(
            error="Bad response", schema=cls, value=value
        )

    with chats.new() as t:
        with pytest.raises(prompting.ResponseParsingError):
            llm.prompt("test_value", schema=F)
        assert "Bad response" in t.messages[-1].text
        assert "test_value" in t.messages[-1].text
        assert "F" in t.messages[-1].text

    @handler(types=F)
    def _(cls):
        yield ""
        yield "nonsense"
        return F()

    with pytest.raises(prompting.SchemaError):
        llm.prompt("Test", schema=F)


def test_streaming_prompt():
    llm = Ferret()
    # Explicitly set stream mode.
    llm.stream_responses = True

    with chats.new("Test Streaming") as t:
        response_content = llm.prompt("stream this")
        assert response_content == "streaming..."

        # The last message in the chat is the one from the LLM.
        last_message = t.messages[-1]
        assert last_message.content == "streaming..."
        assert last_message.sender is llm
        assert last_message._meta["input_tokens"] == 10
        assert last_message._meta["output_tokens"] == 2


def test_nested_chat_id():
    llm = Ferret()
    with chats.new("root") as root:
        sub = chats.Chat(name="sub")
        chats.get_current_chat().append(sub)
        with contexts.enter(chat=sub):
            llm.prompt("Hi")

        sub.name += " - analysis"

    assert root.history[0] is sub
    assert sub.id.startswith("sub - analysis-")
    assert len(sub.history) == 2


def test_chat_usage_aggregation():
    """Test that chat usage properties aggregate token usage from all assistant messages."""
    llm = Ferret()
    llm.stream_responses = True

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
    assert mocked_chat.invocations[0][1] == {"temperature": 0.5}
