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

import json
import logging
import warnings

from kaggle_benchmarks import llm_messages, messages
from kaggle_benchmarks import tools as tool_utils
from kaggle_benchmarks.content_types import audios, images, videos
from kaggle_benchmarks.serializers.base import BaseSerializer, UnsupportedMessageFormat


def _merge_with_warning(base: dict, extra_api_params: dict, content_type: str) -> dict:
    """Merges extra_api_params into base dict, warning on overlapping keys."""
    overlap = set(extra_api_params) & set(base)
    if overlap:
        warnings.warn(
            f"extra_api_params {overlap} will overwrite core {content_type} fields. "
            f"This may produce unexpected results.",
            stacklevel=3,
        )
    return {**base, **extra_api_params}


class OpenAICompletionSerializer(BaseSerializer):
    """Serializer mapping generic messages to the OpenAI Chat Completions API format."""

    def dump_image(self, message: messages.Message[images.ImageContent]):
        """Serializes an image content object into the API's image_url format."""
        image = message.content
        caption = [{"type": "text", "text": image.caption}] if image.caption else []
        image_url = _merge_with_warning(
            {"url": image.url}, image.extra_api_params, "image_url"
        )
        yield {
            "role": self.get_role(message.sender),
            "content": caption + [{"type": "image_url", "image_url": image_url}],
        }

    def dump_llm_message(self, message: llm_messages.LLMMessage):
        """Extracts tool calls and results, injecting them into the sequence before standard text output."""
        if message.tool_calls:
            tool_calls_payload = []
            tool_results = []
            for call in message.tool_calls:
                tool_calls_payload.append(
                    {
                        "id": str(call.call_id),
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments)
                            if not isinstance(call.arguments, str)
                            else call.arguments,
                        },
                    }
                )
                if isinstance(call, tool_utils.ToolInvocationResult):
                    tool_results.append(
                        {
                            "role": "tool",
                            "content": call.text,
                            "tool_call_id": str(call.call_id),
                        }
                    )
            yield {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls_payload,
            }
            yield from tool_results

        if message.content is not None and message.content != "":
            yield from self.dump_message(
                messages.Message(content=message.content, sender=message.sender)
            )

    def dump_text_message(self, message: messages.Message):
        """Serializes a standard textual payload."""
        msg = {
            "role": self.get_role(message.sender),
            "content": message.content or None,
        }

        # Convert normalized ToolInvocation back to Chat Completions wire format.
        tool_calls = message.tool_calls
        if tool_calls and self.get_role(message.sender) == "assistant":
            msg["tool_calls"] = [
                {
                    # `id` is required by the spec; `or ""` defends against test fixtures.
                    "id": tc.call_id or "",
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": (
                            json.dumps(tc.arguments)
                            if isinstance(tc.arguments, dict)
                            else tc.arguments
                        ),
                    },
                }
                for tc in tool_calls
            ]

        yield msg

    def dump_tool_invocation(
        self, message: messages.Message[tool_utils.ToolInvocationResult]
    ):
        """Serializes a tool result as a Chat Completions tool message."""
        result = message.content
        yield {
            "role": "tool",
            "content": result.text,
            "tool_call_id": str(result.call_id) if result.call_id else "",
        }

    _SUPPORTED_AUDIO_FORMATS = {"mp3", "wav"}

    def dump_audio(self, message: messages.Message[audios.AudioContent]):
        """Serializes audio as input_audio for the OpenAI Chat Completions API."""
        audio_content = message.content
        fmt = audio_content._format
        if fmt not in self._SUPPORTED_AUDIO_FORMATS:
            raise UnsupportedMessageFormat(
                f"OpenAI API only supports {self._SUPPORTED_AUDIO_FORMATS} audio formats, "
                f"got '{fmt}' (from mime_type='{audio_content.mime_type}')"
            )
        caption = (
            [{"type": "text", "text": audio_content.caption}]
            if audio_content.caption
            else []
        )
        input_audio = _merge_with_warning(
            {"data": audio_content.b64_string, "format": fmt},
            audio_content.extra_api_params,
            "input_audio",
        )
        yield {
            "role": self.get_role(message.sender),
            "content": caption + [{"type": "input_audio", "input_audio": input_audio}],
        }


class ModelProxyOpenAISerializer(OpenAICompletionSerializer):
    """Specialized OpenAI serializer that maps constructs like videos and images
    specifically for the Kaggle Model Proxy format.
    """

    def dump_image(self, message: messages.Message[images.ImageContent]):
        """Serializes images natively as base64-encoded data URLs compatible with Model Proxy."""
        image = message.content
        caption = [{"type": "text", "text": image.caption}] if image.caption else []
        image_url = _merge_with_warning(
            {"url": f"data:{image.mime_type};base64,{image.b64_string}"},
            image.extra_api_params,
            "image_url",
        )
        yield {
            "role": self.get_role(message.sender),
            "content": caption + [{"type": "image_url", "image_url": image_url}],
        }

    def dump_video(self, message: messages.Message[videos.VideoContent]):
        """Serializes videos leveraging the bespoke `video_url` supported by Model Proxy."""
        video = message.content
        image_url = _merge_with_warning(
            {"url": video.url}, video.extra_api_params, "image_url"
        )
        yield {
            "role": self.get_role(message.sender),
            "content": [{"type": "image_url", "image_url": image_url}],
        }

    def _dump_message(self, message: messages.Message):
        """Fallback method, rendering unrecognized message objects safely."""
        logging.warning(
            "Unrecognized message format encountered: %s", type(message.content)
        )
        yield from self.dump_text_message(
            messages.Message(sender=message.sender, content=str(message.content))
        )
