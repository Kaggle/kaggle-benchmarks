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

import base64
import logging
from typing import Any

from google.genai import types

from kaggle_benchmarks import llm_messages, messages
from kaggle_benchmarks import tools as tool_utils
from kaggle_benchmarks.content_types import audios, images, videos
from kaggle_benchmarks.serializers.base import BaseSerializer

_PART_FIELDS = set(types.Part.model_fields.keys())


def _filter_part_params(api_params: dict[str, Any]) -> dict[str, Any]:
    """Filters api_params to only valid Part fields, warning on unsupported ones."""
    unsupported = set(api_params) - _PART_FIELDS
    if unsupported:
        logging.warning(
            "Ignoring unsupported api_params for GenAI Part: %s. "
            "Supported fields: %s",
            unsupported,
            _PART_FIELDS,
        )
    return {k: v for k, v in api_params.items() if k in _PART_FIELDS}


class GenAISerializer(BaseSerializer):
    """Serializer mapping generic messages to the Google GenAI SDK (Gemini) format."""

    def dump_messages(self, messages: list[messages.Message]):
        """Serializes messages and merges consecutive elements with the same role.

        Grouping contiguous parts by role is important; otherwise, structured output
        simulation and instructions passed across distinct messages may be ignored.
        """
        result = []
        prev = None
        chunks = list(super().dump_messages(messages))
        for chunk in chunks:
            if prev and prev.role == chunk.role:
                prev.parts.extend(chunk.parts)
            else:
                result.append(chunk)
                prev = chunk
        return result

    def dump_text_message(self, message: messages.Message[str]):
        """Serializes a standard textual payload into a Part object."""
        yield types.Content(
            role=self.get_role(message.sender), parts=[types.Part(text=message.content)]
        )

    def dump_image(self, message: messages.Message[images.ImageContent]):
        """Serializes images natively as inline data Blobs for the GenAI client."""
        parts = []
        image = message.content
        if image.caption:
            parts = [types.Part.from_text(text=image.caption)]
        yield types.Content(
            role=self.get_role(message.sender),
            parts=parts
            + [
                types.Part(
                    inline_data=types.Blob(
                        # The API expects the raw base64 string, not bytes.
                        data=image.b64_string,
                        mime_type=image.mime_type,
                    ),
                    **_filter_part_params(image.api_params),
                )
            ],
        )

    def dump_video(self, message: messages.Message[videos.VideoContent]):
        """Serializes video URLs natively as FileData parts for the GenAI client."""
        video = message.content
        yield types.Content(
            role=self.get_role(message.sender),
            parts=[types.Part(
                file_data=types.FileData(file_uri=video.url, mime_type=video.mime_type),
                **_filter_part_params(video.api_params),
            )],
        )

    def dump_audio(self, message: messages.Message[audios.AudioContent]):
        """Serializes audio as inline bytes for the GenAI client."""
        audio_content = message.content
        parts = []
        if audio_content.caption:
            parts.append(types.Part.from_text(text=audio_content.caption))
        parts.append(
            types.Part(
                inline_data=types.Blob(
                    data=base64.b64decode(audio_content.b64_string),
                    mime_type=audio_content.mime_type,
                ),
                **_filter_part_params(audio_content.api_params),
            )
        )
        yield types.Content(
            role=self.get_role(message.sender),
            parts=parts,
        )

    def dump_llm_message(self, message: llm_messages.LLMMessage):
        """Serializes LLM Messages into appropriate GenAI Tool parts."""
        parts = []

        for call in message.tool_calls or []:
            parts.extend(self._dump_invocation(call))

        if message.content:
            parts.append(types.Part.from_text(text=message.payload))

        if not parts:
            parts.append(types.Part.from_text(text=""))

        yield types.Content(role=self.get_role(message.sender), parts=parts)

    def _dump_message(self, message: messages.Message):
        """Fallback method, rendering unrecognized message objects safely."""
        logging.warning(
            "Unrecognized message format encountered: %s", type(message.content)
        )
        yield from self.dump_text_message(
            messages.Message(sender=message.sender, content=message.payload)
        )

    def dump_tool_invocation(
        self, message: messages.Message[tool_utils.ToolInvocationResult]
    ):
        yield types.Content(
            role=self.get_role(message.sender),
            parts=list(self._dump_invocation(message.content)),
        )

    def _dump_invocation(
        self, call: tool_utils.ToolInvocationResult | tool_utils.ToolInvocation
    ):
        if isinstance(call, tool_utils.ToolInvocationResult):
            yield types.Part.from_function_response(
                name=call.name, response={"result": call.output}
            )

        else:
            args = call.arguments
            yield types.Part.from_function_call(name=call.name, args=args)
