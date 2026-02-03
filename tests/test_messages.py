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

from kaggle_benchmarks import actors, chats, messages, user
from kaggle_benchmarks.actors.llms import LLMResponse


class Parrot(actors.LLMChat):
    """Repeats last user message."""

    def invoke(self, messages, **kwargs):
        return next(
            LLMResponse(content=m.payload)
            for m in reversed(messages)
            if m.sender == user
        )


def test_raw_payload():
    p = Parrot()
    with chats.new() as chat:
        raw_response = '{"value": 0.01}'
        m = p.prompt(raw_response, schema=float)
        assert m == 0.01
        assert chat.messages[-1].payload == raw_response

    r = p.prompt('{"value": true}', schema=bool)
    assert r


def test_dataclass_payload():
    @dataclass
    class Point:
        x: float
        y: float

    msg = messages.Message(Point(1, 2), sender=user)
    assert json.loads(msg.payload) == {"x": 1, "y": 2}


def test_class_payload():
    class Point:
        def __init__(self, x):
            self.x = x

    msg = messages.Message(Point(1), sender=user)
    with pytest.warns():
        payload = msg.payload

    assert json.loads(payload) == {"x": 1}


def test_streaming():
    text = "a b c d"

    def g():
        for chunk in text:
            yield chunk
            assert msg.content.endswith(chunk)

    msg = messages.Message(content="", sender=None)
    msg.stream(g())
    assert msg.content == text


def test_streaming_with_token_counts():
    """Tests that streaming correctly updates metadata like token counts."""

    def chunk_generator():
        yield LLMResponse(
            content="Hello ", meta={"input_tokens": 10, "output_tokens": 1}
        )
        yield LLMResponse(
            content="world", meta={"input_tokens": 10, "output_tokens": 2}
        )
        yield LLMResponse(content="!", meta={"input_tokens": 10, "output_tokens": 3})

    msg = messages.Message(content="", sender=None)
    msg.stream(chunk_generator())

    assert msg.content == "Hello world!"
    assert msg._meta["input_tokens"] == 10
    assert msg._meta["output_tokens"] == 3


def test_tool_calls_property():
    """Tests that the tool_calls property correctly retrieves data from _meta."""
    mock_tool_calls = [{"id": "call_abc", "type": "function"}]

    # Message with tool calls
    msg_with_tools = messages.Message(
        content="", sender=None, _meta={"tool_calls": mock_tool_calls}
    )

    msg_without_tools = messages.Message(content="Hello", sender=None)

    assert msg_with_tools.tool_calls == mock_tool_calls
    assert msg_without_tools.tool_calls is None
