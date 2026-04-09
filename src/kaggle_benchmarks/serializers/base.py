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
import copy
import dataclasses
import itertools
import json

import pydantic

from kaggle_benchmarks import actors, chats, llm_messages, tools
from kaggle_benchmarks import messages as msg
from kaggle_benchmarks.content_types import audio, images, videos


class UnsupportedMessageFormat(ValueError):
    pass


def _copy_replace(message, **new_fields):
    new = copy.copy(message)
    for k, v in new_fields.items():
        setattr(new, k, v)
    return new


class BaseSerializer:
    """Base class for all message serializers.

    Provides the core logic to map generic benchmark messages to provider-specific
    formats. Subclasses must implement the specific `dump_*` methods.
    """

    def __init__(self, roles_mapping: dict[str, str] | None = None):
        self.roles_mapping = roles_mapping or {}

    def get_role(self, sender: actors.Actor):
        """Resolves the provider-specific role for a given sender using roles_mapping."""
        return self.roles_mapping.get(sender.role, sender.role)

    def dump_chat(self, chat: chats.Chat):
        """Serializes an entire chat history into a provider-specific format."""
        return self.dump_messages(chat.messages)

    def dump_messages(self, messages: list[msg.Message]):
        """Serializes a list of messages into a provider-specific format."""
        return itertools.chain(*(self.dump_message(message) for message in messages))

    def dump_message(self, message: msg.Message):
        """Dynamically dispatches serialization based on the message content type."""
        if isinstance(message, llm_messages.LLMMessage):
            try:
                yield from self.dump_llm_message(message)
                return
            except NotImplementedError:
                # Fallback if the subclass doesn't support explicit LLM messages
                pass

        content = message.content
        if isinstance(content, str):
            yield from self.dump_text_message(message)
        elif isinstance(content, images.ImageContent):
            yield from self.dump_image(message)
        elif isinstance(content, videos.VideoContent):
            yield from self.dump_video(message)
        elif isinstance(content, audio.AudioContent):
            yield from self.dump_audio(message)
        elif isinstance(content, dict):
            yield from self.dump_json_message(message)
        elif isinstance(content, tools.ToolInvocationResult):
            yield from self.dump_tool_invocation(message)
        elif isinstance(content, pydantic.BaseModel):
            yield from self.dump_json_message(
                _copy_replace(message, content=message.content.model_dump())
            )
        elif dataclasses.is_dataclass(content) and not isinstance(content, type):
            yield from self.dump_json_message(
                _copy_replace(message, content=dataclasses.asdict(content))
            )
        else:
            yield from self._dump_message(message)

    def _dump_message(self, message: msg.Message):
        """Fallback method for unsupported message types. Override in subclass to handle."""
        raise NotImplementedError(
            f"Unsupported message format for: {type(message.content)}"
        )

    def dump_tool_invocation(self, message: msg.Message[tools.ToolInvocationResult]):
        raise NotImplementedError()

    def dump_llm_message(self, message: llm_messages.LLMMessage):
        """Serializes an LLM message containing tools and complex outputs."""
        raise NotImplementedError()

    def dump_text_message(self, message: msg.Message[str]):
        """Serializes a standard text message."""
        raise NotImplementedError()

    def dump_json_message(self, message: msg.Message[dict]):
        """Serializes a JSON dictionary message by stringifying it as text by default."""
        yield from self.dump_text_message(
            _copy_replace(message, content=json.dumps(message.content))
        )

    def dump_image(self, message: msg.Message[images.ImageContent]):
        """Serializes an image message."""
        raise NotImplementedError()

    def dump_video(self, message: msg.Message[videos.VideoContent]):
        """Serializes a video message."""
        raise NotImplementedError()

    def dump_audio(self, message: msg.Message[audio.AudioContent]):
        """Serializes an audio message."""
        raise NotImplementedError()
