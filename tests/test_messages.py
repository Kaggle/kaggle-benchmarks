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
from types import SimpleNamespace

import pydantic
import pytest

from kaggle_benchmarks import chats, messages, user
from kaggle_benchmarks.actors.llms import LLMResponse
from kaggle_benchmarks.tools.base import ToolInvocation
from tests.mocks import MockedChat


def test_raw_payload():
    float_response = '{"value": 0.01}'
    p = MockedChat.from_contents([float_response, '{"value": true}'])
    with chats.new() as chat:
        m = p.prompt(float_response, schema=float)
        assert m == 0.01
        assert chat.messages[-1].payload == float_response

    r = p.prompt('{"value": true}', schema=bool)
    assert r


def test_dataclass_payload():
    @dataclass
    class Point:
        x: float
        y: float

    msg = messages.Message(Point(1, 2), sender=user)
    assert json.loads(msg.payload) == {"x": 1, "y": 2}


def test_pydantic_payload():
    class Point(pydantic.BaseModel):
        x: float
        y: float

    msg = messages.Message(Point(x=1.5, y=2.5), sender=user)
    assert json.loads(msg.payload) == {"x": 1.5, "y": 2.5}


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
    """Tests that the tool_calls property correctly retrieves data from _meta.

    Intentionally uses raw dicts — verifies the property shim is agnostic to
    value type. Post-`_meta` normalization, real LLM responses store
    ToolInvocation objects here, but the property itself makes no assumptions.
    """
    mock_tool_calls = [{"id": "call_abc", "type": "function"}]

    # Message with tool calls
    msg_with_tools = messages.Message(
        content="", sender=None, _meta={"tool_calls": mock_tool_calls}
    )

    msg_without_tools = messages.Message(content="Hello", sender=None)

    assert msg_with_tools.tool_calls == mock_tool_calls
    assert msg_without_tools.tool_calls is None


def _make_tool_call_chunk(
    index, id_=None, name=None, arguments=None, thought_signature=None, thought=None
):
    """Builds an OpenAI-style streaming tool-call chunk delta. The thought
    fields mirror what GenAI streaming chunks carry (synthesized via
    SimpleNamespace in GoogleGenAI._get_stream_response)."""
    return SimpleNamespace(
        index=index,
        id=id_,
        function=SimpleNamespace(name=name, arguments=arguments),
        thought_signature=thought_signature,
        thought=thought,
    )


def test_streaming_no_tool_calls_leaves_meta_unset():
    """When the stream contains no tool-call chunks, _meta['tool_calls']
    is not touched (stays absent)."""

    def chunk_generator():
        yield LLMResponse(content="Hello ", meta={"input_tokens": 5})
        yield LLMResponse(content="world", meta={"input_tokens": 5, "output_tokens": 2})

    msg = messages.Message(content="", sender=None)
    msg.stream(chunk_generator())

    assert msg.content == "Hello world"
    assert "tool_calls" not in msg._meta


def test_streaming_content_and_tool_calls_finalized():
    """Realistic shape: chunks interleave delta.content and delta.tool_calls
    (OpenAI's actual streaming format) and may carry Gemini 3.x thought
    fields. The finalize pass converts accumulated dicts to typed
    ToolInvocation. Second _finalize call is a no-op (idempotency guard)."""

    def chunk_generator():
        # First chunk announces the call + opens args + carries thought fields
        # (mirrors how GenAI _get_stream_response synthesizes its delta).
        yield LLMResponse(
            content="Let me ",
            tool_calls=[
                _make_tool_call_chunk(
                    index=0,
                    id_="call_1",
                    name="add",
                    arguments='{"a":',
                    thought_signature=b"sig-bytes",
                    thought=True,
                )
            ],
        )
        yield LLMResponse(
            content="compute. ",
            tool_calls=[_make_tool_call_chunk(index=0, arguments=' 1, "b": 2}')],
        )
        yield LLMResponse(content="Done.")

    msg = messages.Message(content="", sender=None)
    msg.stream(chunk_generator())

    assert msg.content == "Let me compute. Done."
    [tc] = msg._meta["tool_calls"]
    assert isinstance(tc, ToolInvocation)
    assert tc.name == "add"
    assert tc.arguments == {"a": 1, "b": 2}
    assert tc.call_id == "call_1"
    # Thought fields captured during accumulation, preserved through finalize.
    assert tc.thought_signature == b"sig-bytes"
    assert tc.thought is True

    # Idempotency: re-running finalize must not re-wrap or replace the object.
    msg._finalize_tool_calls()
    [second] = msg._meta["tool_calls"]
    assert second is tc
