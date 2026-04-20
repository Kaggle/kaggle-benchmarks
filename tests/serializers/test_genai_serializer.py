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
                    types.Part.from_function_call(name="test_tool", args={}),
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
                    types.Part.from_function_response(
                        name="test_tool", response={"result": "result"}
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
                    types.Part.from_function_response(
                        name="test_tool", response={"result": "result"}
                    ),
                ],
            )
        ],
        id="message_with_tool_invocation_result",
    ),
]


@pytest.mark.parametrize("message, expected_raw_messages", MESSAGE_FORMATS)
def test_dump_message(message, expected_raw_messages):
    serializer = genai_serializer.GenAISerializer()
    actual = [c.model_dump() for c in serializer.dump_message(message)]
    expected = [c.model_dump() for c in expected_raw_messages]
    assert actual == expected


def test_dump_image_message_with_api_params():
    serializer = genai_serializer.GenAISerializer()
    image_content = ImageBase64(
        b64_string=B64_STRING,
        mime_type="image/png",
        api_params={"media_resolution": {"level": "MEDIA_RESOLUTION_LOW"}},
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


def test_dump_video_message_with_api_params():
    serializer = genai_serializer.GenAISerializer()
    video_content = videos.VideoURL(
        url="https://youtube.com/watch?v=dummy",
        api_params={
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


def test_dump_audio_message_with_api_params():
    serializer = genai_serializer.GenAISerializer()
    audio_content = audios.AudioContent(
        b64_string="dGVzdA==",
        mime_type="audio/wav",
        api_params={"media_resolution": {"level": "MEDIA_RESOLUTION_LOW"}},
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


def test_dump_image_filters_unsupported_api_params():
    """Verifies that provider-specific params like 'detail' are dropped with a warning."""
    serializer = genai_serializer.GenAISerializer()
    image_content = ImageBase64(
        b64_string=B64_STRING,
        mime_type="image/png",
        api_params={"detail": "low"},
    )
    message = messages.Message(content=image_content, sender=actors.user)
    with pytest.warns(UserWarning, match="Ignoring unsupported api_params"):
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
