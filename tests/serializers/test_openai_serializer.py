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

from kaggle_benchmarks import chats, llm_messages, messages
from kaggle_benchmarks.actors import base as actors
from kaggle_benchmarks.content_types import videos
from kaggle_benchmarks.content_types.images import ImageBase64, ImageURL
from kaggle_benchmarks.serializers import openai as openai_serializer
from kaggle_benchmarks.tools import ToolInvocation, ToolInvocationResult


@dataclasses.dataclass
class DummyDataclass:
    foo: str
    bar: int


class DummyPydantic(pydantic.BaseModel):
    foo: str
    bar: int


# A shared set of message formats and their expected outputs.
# Each tuple is: (message, expected_raw_messages)
MESSAGE_FORMATS = [
    pytest.param(
        messages.Message(content="Hello", sender=actors.user),
        [{"role": "user", "content": "Hello"}],
        id="text_message",
    ),
    pytest.param(
        messages.Message(
            content=ImageURL(url="http://example.com/a.png", caption="A cat"),
            sender=actors.user,
        ),
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "A cat"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://example.com/a.png"},
                    },
                ],
            }
        ],
        id="image_message",
    ),
    pytest.param(
        messages.Message(
            content=videos.VideoURL(url="https://youtube.com/watch?v=dummy"),
            sender=actors.user,
        ),
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "https://youtube.com/watch?v=dummy"},
                ],
            }
        ],
        id="video_message",
    ),
    pytest.param(
        llm_messages.LLMMessage(
            content="",
            sender=actors.system,
            tool_calls=[ToolInvocation(name="test_tool", call_id="123", arguments={})],
        ),
        [
            {
                "call_id": "123",
                "arguments": "{}",
                "name": "test_tool",
                "type": "function_call",
            },
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
            {
                "call_id": "123",
                "arguments": "{}",
                "name": "test_tool",
                "type": "function_call",
            },
            {
                "call_id": "123",
                "output": "result",
                "type": "function_call_output",
            },
        ],
        id="llm_message_with_tool_invocation_result",
    ),
    pytest.param(
        messages.Message(
            content=DummyDataclass(foo="baz", bar=42),
            sender=actors.user,
        ),
        [
            {
                "role": "user",
                "content": '{"foo": "baz", "bar": 42}',
            },
        ],
        id="dataclass_message",
    ),
    pytest.param(
        messages.Message(
            content=DummyPydantic(foo="baz", bar=42),
            sender=actors.user,
        ),
        [
            {
                "role": "user",
                "content": '{"foo": "baz", "bar": 42}',
            },
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
            {
                "call_id": "123",
                "arguments": "{}",
                "name": "test_tool",
                "type": "function_call",
            },
            {
                "call_id": "123",
                "output": "result",
                "type": "function_call_output",
            },
        ],
        id="message_with_tool_invocation_result",
    ),
]


@pytest.mark.parametrize("message, expected_raw_messages", MESSAGE_FORMATS)
def test_dump_message(message, expected_raw_messages):
    serializer = openai_serializer.OpenAICompletionSerializer(roles_mapping={})
    assert list(serializer.dump_message(message)) == expected_raw_messages


def test_dump_messages():
    serializer = openai_serializer.OpenAICompletionSerializer(roles_mapping={})
    msgs = [
        messages.Message(content="Hello", sender=actors.user),
        messages.Message(
            content=ImageURL(url="http://example.com/a.png", caption="A cat"),
            sender=actors.user,
        ),
    ]
    expected_raw_messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "A cat"},
                {"type": "image_url", "image_url": {"url": "http://example.com/a.png"}},
            ],
        },
    ]
    assert list(serializer.dump_messages(msgs)) == expected_raw_messages


def test_dump_chat():
    serializer = openai_serializer.OpenAICompletionSerializer(roles_mapping={})
    msgs = [
        messages.Message(content="Hello", sender=actors.user),
        messages.Message(
            content=ImageURL(url="http://example.com/a.png", caption="A cat"),
            sender=actors.user,
        ),
    ]
    chat = chats.Chat(history=msgs)
    expected_raw_messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "A cat"},
                {"type": "image_url", "image_url": {"url": "http://example.com/a.png"}},
            ],
        },
    ]
    assert list(serializer.dump_chat(chat)) == expected_raw_messages


def test_dump_image_message_model_proxy():
    serializer = openai_serializer.ModelProxyOpenAISerializer(roles_mapping={})
    image_content = ImageBase64(b64_string="...", mime_type="image/png")
    message = messages.Message(content=image_content, sender=actors.user)
    raw_messages = list(serializer.dump_message(message))
    assert raw_messages == [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,..."},
                },
            ],
        }
    ]
