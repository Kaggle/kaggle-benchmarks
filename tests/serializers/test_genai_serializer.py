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

import dataclasses

import pydantic
import pytest
from google.genai import types

from kaggle_benchmarks import chats, llm_messages, messages
from kaggle_benchmarks.actors import base as actors
from kaggle_benchmarks.content_types import audios, videos
from kaggle_benchmarks.content_types.images import ImageBase64
from kaggle_benchmarks.serializers import genai as genai_serializer
from kaggle_benchmarks.tools import ToolInvocation, ToolInvocationResult


@dataclasses.dataclass
class DummyDataclass:
    foo: str
    bar: int


class DummyPydantic(pydantic.BaseModel):
    foo: str
    bar: int


# Pre-defined base64 string for reuse
B64_STRING = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

# A shared set of message formats and their expected outputs.


def _make_function_call_part(name: str, args: dict, call_id: str) -> types.Part:
    """Builds a function_call Part with the id field set."""
    part = types.Part.from_function_call(name=name, args=args)
    part.function_call.id = call_id
    return part


def _make_function_response_part(name: str, response: dict, call_id: str) -> types.Part:
    """Builds a function_response Part with the id field set."""
    part = types.Part.from_function_response(name=name, response=response)
    part.function_response.id = call_id
    return part


MESSAGE_FORMATS = [
    pytest.param(
        messages.Message(content="Hello", sender=actors.user),
        [types.Content(role="user", parts=[types.Part(text="Hello")])],
        id="text_message",
    ),
    pytest.param(
        messages.Message(
            content=ImageBase64(
                b64_string=B64_STRING, mime_type="image/png", caption="A cat"
            ),
            sender=actors.user,
        ),
        [
            types.Content(
                role="user",
                parts=[
                    types.Part(text="A cat"),
                    types.Part(
                        inline_data=types.Blob(data=B64_STRING, mime_type="image/png")
                    ),
                ],
            )
        ],
        id="image_message",
    ),
    pytest.param(
        messages.Message(
            content=videos.VideoURL(url="https://youtube.com/watch?v=dummy"),
            sender=actors.user,
        ),
        [
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        file_data=types.FileData(
                            file_uri="https://youtube.com/watch?v=dummy",
                            mime_type="video/*",
                        ),
                    ),
                ],
            )
        ],
        id="video_message",
    ),
    pytest.param(
        messages.Message(
            content=audios.AudioContent(b64_string="dGVzdA==", mime_type="audio/wav"),
            sender=actors.user,
        ),
        [
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=b"test", mime_type="audio/wav"),
                    ),
                ],
            )
        ],
        id="audio_message",
    ),
    pytest.param(
        messages.Message(
            content=audios.AudioContent(
                b64_string="dGVzdA==", mime_type="audio/mp3", caption="A speech clip"
            ),
            sender=actors.user,
        ),
        [
            types.Content(
                role="user",
                parts=[
                    types.Part(text="A speech clip"),
                    types.Part(
                        inline_data=types.Blob(data=b"test", mime_type="audio/mp3"),
                    ),
                ],
            )
        ],
        id="audio_message_with_caption",
    ),
    pytest.param(
        llm_messages.LLMMessage(
            content="",
            sender=actors.system,
            tool_calls=[ToolInvocation(name="test_tool", call_id="123", arguments={})],
        ),
        [
            types.Content(
                role="system",
                parts=[
                    _make_function_call_part("test_tool", {}, "123"),
                ],
            )
        ],
        id="llm_message_with_tool_calls",
    ),
    pytest.param(
        llm_messages.LLMMessage(
            content="",
            sender=actors.system,
            tool_calls=[
                ToolInvocationResult(
                    name="test_tool", call_id="123", arguments={}, output="result"
                )
            ],
        ),
        [
            types.Content(
                role="system",
                parts=[
                    _make_function_response_part(
                        "test_tool", {"result": "result"}, "123"
                    ),
                ],
            )
        ],
        id="llm_message_with_tool_invocation_result",
    ),
    pytest.param(
        messages.Message(
            content=DummyDataclass(foo="baz", bar=42),
            sender=actors.user,
        ),
        [
            types.Content(
                role="user",
                parts=[
                    types.Part(text='{"foo": "baz", "bar": 42}'),
                ],
            )
        ],
        id="dataclass_message",
    ),
    pytest.param(
        messages.Message(
            content=DummyPydantic(foo="baz", bar=42),
            sender=actors.user,
        ),
        [
            types.Content(
                role="user",
                parts=[
                    types.Part(text='{"foo": "baz", "bar": 42}'),
                ],
            )
        ],
        id="pydantic_message",
    ),
    pytest.param(
        messages.Message(
            content=ToolInvocationResult(
                name="test_tool", call_id="123", arguments={}, output="result"
            ),
            sender=actors.user,
        ),
        [
            types.Content(
                role="user",
                parts=[
                    _make_function_response_part(
                        "test_tool", {"result": "result"}, "123"
                    ),
                ],
            )
        ],
        id="message_with_tool_invocation_result",
    ),
]


# Mirrors GoogleGenAI's roles_mapping. The bare parametrized batch above uses
# the default mapping (passthrough); these tests need assistant→model so the
# `dump_text_message` tool_calls branch fires (it checks for role == "model").
_GENAI_ROLES = {"assistant": "model", "system": "user", "tool": "user"}
_assistant_actor = actors.Actor(name="LLM", role="assistant", avatar="🤖")


def test_dump_text_message_with_meta_tool_invocation():
    """Plain Message with normalized ToolInvocation in _meta — the path
    taken by built-in LLM responses after respond()'s normalization."""
    serializer = genai_serializer.GenAISerializer(roles_mapping=_GENAI_ROLES)
    msg = messages.Message(
        content="",
        sender=_assistant_actor,
        _meta={
            "tool_calls": [
                ToolInvocation(name="test_tool", call_id="123", arguments={"x": 1})
            ]
        },
    )
    expected = [
        types.Content(
            role="model",
            parts=[_make_function_call_part("test_tool", {"x": 1}, "123")],
        )
    ]
    actual = [c.model_dump() for c in serializer.dump_message(msg)]
    assert actual == [c.model_dump() for c in expected]


def test_dump_text_message_carries_thought_signature():
    """thought_signature (bytes) / thought (bool) round-trip from
    ToolInvocation into the GenAI Part — required by Gemini 3.x multi-turn
    tool conversations. Types mirror google.genai types.Part."""
    serializer = genai_serializer.GenAISerializer(roles_mapping=_GENAI_ROLES)
    msg = messages.Message(
        content="",
        sender=_assistant_actor,
        _meta={
            "tool_calls": [
                ToolInvocation(
                    name="test_tool",
                    call_id="123",
                    arguments={},
                    thought_signature=b"sig-bytes",
                    thought=True,
                )
            ]
        },
    )
    actual = [c.model_dump() for c in serializer.dump_message(msg)]
    # Inspect the single Part we expect to see.
    [content] = actual
    assert content["role"] == "model"
    [part] = content["parts"]
    assert part["function_call"]["name"] == "test_tool"
    assert part["function_call"]["id"] == "123"
    assert part["thought_signature"] == b"sig-bytes"
    assert part["thought"] is True


def test_dump_text_message_with_string_arguments_logs_warning(caplog):
    """If arguments is a string (JSONDecodeError fallback from streaming),
    the serializer logs a warning and emits empty args rather than crashing."""
    import logging

    serializer = genai_serializer.GenAISerializer(roles_mapping=_GENAI_ROLES)
    msg = messages.Message(
        content="",
        sender=_assistant_actor,
        _meta={
            "tool_calls": [
                ToolInvocation(name="test_tool", call_id="123", arguments='{"a":')
            ]
        },
    )
    with caplog.at_level(logging.WARNING):
        actual = [c.model_dump() for c in serializer.dump_message(msg)]

    assert any("non-dict arguments" in rec.message for rec in caplog.records)
    [content] = actual
    [part] = content["parts"]
    assert part["function_call"]["name"] == "test_tool"
    # Empty args fallback.
    assert part["function_call"]["args"] in (None, {})


@pytest.mark.parametrize("message, expected_raw_messages", MESSAGE_FORMATS)
def test_dump_message(message, expected_raw_messages):
    serializer = genai_serializer.GenAISerializer()
    actual = [c.model_dump() for c in serializer.dump_message(message)]
    expected = [c.model_dump() for c in expected_raw_messages]
    assert actual == expected


def test_dump_image_message_with_extra_api_params():
    serializer = genai_serializer.GenAISerializer()
    image_content = ImageBase64(
        b64_string=B64_STRING,
        mime_type="image/png",
        extra_api_params={"media_resolution": {"level": "MEDIA_RESOLUTION_LOW"}},
    )
    message = messages.Message(content=image_content, sender=actors.user)
    actual = [c.model_dump() for c in serializer.dump_message(message)]
    expected = [
        types.Content(
            role="user",
            parts=[
                types.Part(
                    inline_data=types.Blob(data=B64_STRING, mime_type="image/png"),
                    media_resolution={"level": "MEDIA_RESOLUTION_LOW"},
                )
            ],
        ).model_dump()
    ]
    assert actual == expected


def test_dump_video_message_with_extra_api_params():
    serializer = genai_serializer.GenAISerializer()
    video_content = videos.VideoURL(
        url="https://youtube.com/watch?v=dummy",
        extra_api_params={
            "video_metadata": {"fps": 1.0, "start_offset": "0s", "end_offset": "10s"}
        },
    )
    message = messages.Message(content=video_content, sender=actors.user)
    actual = [c.model_dump() for c in serializer.dump_message(message)]
    expected = [
        types.Content(
            role="user",
            parts=[
                types.Part(
                    file_data=types.FileData(
                        file_uri="https://youtube.com/watch?v=dummy",
                        mime_type="video/*",
                    ),
                    video_metadata={
                        "fps": 1.0,
                        "start_offset": "0s",
                        "end_offset": "10s",
                    },
                )
            ],
        ).model_dump()
    ]
    assert actual == expected


def test_dump_audio_message_with_extra_api_params():
    serializer = genai_serializer.GenAISerializer()
    audio_content = audios.AudioContent(
        b64_string="dGVzdA==",
        mime_type="audio/wav",
        extra_api_params={"media_resolution": {"level": "MEDIA_RESOLUTION_LOW"}},
    )
    message = messages.Message(content=audio_content, sender=actors.user)
    actual = [c.model_dump() for c in serializer.dump_message(message)]
    expected = [
        types.Content(
            role="user",
            parts=[
                types.Part(
                    inline_data=types.Blob(data=b"test", mime_type="audio/wav"),
                    media_resolution={"level": "MEDIA_RESOLUTION_LOW"},
                )
            ],
        ).model_dump()
    ]
    assert actual == expected


def test_dump_image_filters_unsupported_extra_api_params():
    """Verifies that provider-specific params like 'detail' are dropped with a warning."""
    serializer = genai_serializer.GenAISerializer()
    image_content = ImageBase64(
        b64_string=B64_STRING,
        mime_type="image/png",
        extra_api_params={"detail": "low"},
    )
    message = messages.Message(content=image_content, sender=actors.user)
    with pytest.warns(UserWarning, match="Ignoring unsupported extra_api_params"):
        actual = [c.model_dump() for c in serializer.dump_message(message)]
    expected = [
        types.Content(
            role="user",
            parts=[
                types.Part(
                    inline_data=types.Blob(data=B64_STRING, mime_type="image/png"),
                )
            ],
        ).model_dump()
    ]
    assert actual == expected


def test_dump_messages():
    serializer = genai_serializer.GenAISerializer()
    msgs = [
        messages.Message(content="Hello", sender=actors.user),
        messages.Message(content="World", sender=actors.user),
    ]
    # In GenAI serializer, consecutive messages with the same role get grouped
    expected_raw_messages = [
        types.Content(
            role="user",
            parts=[types.Part(text="Hello"), types.Part(text="World")],
        )
    ]
    actual = [c.model_dump() for c in serializer.dump_messages(msgs)]
    expected = [c.model_dump() for c in expected_raw_messages]
    assert actual == expected


def test_dump_chat():
    serializer = genai_serializer.GenAISerializer()
    msgs = [
        messages.Message(content="Hello", sender=actors.user),
        messages.Message(content="World", sender=actors.user),
    ]
    chat = chats.Chat(history=msgs)
    expected_raw_messages = [
        types.Content(
            role="user",
            parts=[types.Part(text="Hello"), types.Part(text="World")],
        )
    ]
    actual = [c.model_dump() for c in serializer.dump_chat(chat)]
    expected = [c.model_dump() for c in expected_raw_messages]
    assert actual == expected
