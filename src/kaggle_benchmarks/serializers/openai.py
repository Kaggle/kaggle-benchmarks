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

from kaggle_benchmarks import llm_messages, messages
from kaggle_benchmarks import tools as tool_utils
from kaggle_benchmarks.content_types import images, videos
from kaggle_benchmarks.serializers.base import BaseSerializer


class OpenAICompletionSerializer(BaseSerializer):
    """Serializer mapping generic messages to the OpenAI Chat Completions API format."""

    def dump_image(self, message: messages.Message[images.ImageContent]):
        """Serializes an image content object into the API's image_url format."""
        image = message.content
        caption = [{"type": "text", "text": image.caption}] if image.caption else []
        yield {
            "role": self.get_role(message.sender),
            "content": caption
            + [
                {"type": "image_url", "image_url": {"url": image.url}},
            ],
        }

    def dump_llm_message(self, message: llm_messages.LLMMessage):
        """Extracts tool calls and results, injecting them into the sequence before standard text output."""
        for call in message.tool_calls or []:
            yield from self._dump_invocation(call)

        if message.content is not None and message.content != "":
            yield from self.dump_message(
                messages.Message(content=message.content, sender=message.sender)
            )

    def dump_text_message(self, message: messages.Message):
        """Serializes a standard textual payload."""
        yield {
            "role": self.get_role(message.sender),
            "content": message.content,
        }

    def dump_tool_invocation(
        self, message: messages.Message[tool_utils.ToolInvocationResult]
    ):
        yield from self._dump_invocation(message.content)

    def _dump_invocation(
        self, call: tool_utils.ToolInvocationResult | tool_utils.ToolInvocation
    ):
        yield {
            "call_id": call.call_id,
            "arguments": json.dumps(call.arguments)
            if not isinstance(call.arguments, str)
            else call.arguments,
            "name": call.name,
            "type": "function_call",
        }
        if isinstance(call, tool_utils.ToolInvocationResult):
            yield {
                "type": "function_call_output",
                "output": str(call.output),
                "call_id": str(call.call_id),
            }


class ModelProxyOpenAISerializer(OpenAICompletionSerializer):
    """Specialized OpenAI serializer that maps constructs like videos and images
    specifically for the Kaggle Model Proxy format.
    """

    def dump_image(self, message: messages.Message[images.ImageContent]):
        """Serializes images natively as base64-encoded data URLs compatible with Model Proxy."""
        image = message.content
        caption = [{"type": "text", "text": image.caption}] if image.caption else []
        yield {
            "role": self.get_role(message.sender),
            "content": caption
            + [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image.mime_type};base64,{image.b64_string}"
                    },
                },
            ],
        }

    def dump_video(self, message: messages.Message[videos.VideoContent]):
        """Serializes videos leveraging the bespoke `video_url` supported by Model Proxy."""
        video = message.content
        yield {
            "role": self.get_role(message.sender),
            "content": [
                {"type": "image_url", "image_url": {"url": video.url}},
            ],
        }

    def _dump_message(self, message: messages.Message):
        """Fallback method, rendering unrecognized message objects safely."""
        logging.warning(
            "Unrecognized message format encountered: %s", type(message.content)
        )
        yield from self.dump_text_message(
            messages.Message(sender=message.sender, content=str(message.content))
        )
