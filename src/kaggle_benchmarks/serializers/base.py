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
import itertools
import json

import pydantic

from kaggle_benchmarks import actors, chats, llm_messages, tools
from kaggle_benchmarks import messages as msg
from kaggle_benchmarks.content_types import images, videos


class UnsupportedMessageFormat(ValueError):
    pass


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
        elif isinstance(content, dict):
            yield from self.dump_json_message(message)
        elif isinstance(content, pydantic.BaseModel):
            msg = copy.copy(message)
            msg.content = content.model_dump()
            yield from self.dump_json_message(msg)
        elif isinstance(content, tools.ToolInvocationResult):
            yield from self.dump_tool_invocation(message)
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
            message.copy(new_content=json.dumps(message.content))
        )

    def dump_image(self, image: msg.Message[images.ImageContent]):
        """Serializes an image content object."""
        raise NotImplementedError()

    def dump_video(self, video: msg.Message[videos.VideoContent]):
        """Serializes a video content object."""
        raise NotImplementedError()
