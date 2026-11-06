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

"""An agent that calls the Gemini API directly.

Implements :meth:`LLMActor.invoke` against ``google-genai``: no Model Proxy, and
no per-model workarounds — Gemini handles structured output and tool calls
natively, so the call is just a translation of the conversation into GenAI's
types and of its reply back into a message.
"""

from __future__ import annotations

import os
from typing import Callable

from google import genai
from google.genai import types

from kaggle_benchmarks.agents.base import LLMAgent
from kaggle_benchmarks.llm_messages import LLMMessage
from kaggle_benchmarks.messages import Message
from kaggle_benchmarks.serializers import genai as genai_serializer
from kaggle_benchmarks.tools import functions
from kaggle_benchmarks.tools.base import ToolInvocation

_ROLES = {"assistant": "model", "system": "user", "tool": "user"}


def direct_client(api_key: str | None = None) -> genai.Client:
    """Builds a client for the Gemini API itself, bypassing Model Proxy.

    Args:
        api_key: The Google API key. Defaults to ``GOOGLE_API_KEY``, which the
            SDK reads on its own. Vertex AI works too, through the SDK's
            ``GOOGLE_GENAI_USE_VERTEXAI`` / ``GOOGLE_CLOUD_*`` variables.

    Raises:
        ValueError: If no key is given and none is configured for the SDK.
    """
    if api_key is None and not os.getenv("GOOGLE_API_KEY"):
        # Vertex configuration is a valid alternative to an API key.
        if not os.getenv("GOOGLE_GENAI_USE_VERTEXAI"):
            raise ValueError(
                "No Google API key found. Pass api_key=..., or set the "
                "GOOGLE_API_KEY environment variable."
            )
    return genai.Client(api_key=api_key)


class GenAIAgent(LLMAgent):
    """An agent answering through the Gemini API.

    Args:
        model: The Gemini model id.
        api_key: Passed to :func:`direct_client`; read from the environment
            when omitted.
        client: A ready-made client, for tests or a custom endpoint. Takes
            precedence over ``api_key``.
        **kwargs: Forwarded to ``LLMAgent`` (``tools``, ``subagents``,
            ``instructions``, ``description``, …).
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        api_key: str | None = None,
        client: genai.Client | None = None,
        **kwargs,
    ):
        kwargs.setdefault("name", model)
        super().__init__(**kwargs)
        self.model = model
        self.client = client or direct_client(api_key)
        self.serializer = genai_serializer.GenAISerializer(roles_mapping=_ROLES)

    def invoke(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        schema: type = str,
        tools: list[Callable] | None = None,
        **kwargs,
    ) -> LLMMessage:
        config = types.GenerateContentConfig(
            system_instruction=system,
            **kwargs,
        )
        if schema is not str:
            config.response_mime_type = "application/json"
            config.response_schema = schema
        if tools:
            config.tools = [
                types.Tool(function_declarations=[functions.function_to_genai_tool(t)])
                for t in tools
            ]

        response = self.client.models.generate_content(
            model=self.model,
            contents=self.serializer.dump_messages(messages),
            config=config,
        )
        return self._to_message(response)

    def _to_message(self, response) -> LLMMessage:
        """Turns a GenAI response into a message, text and tool calls together."""
        text, tool_calls = "", []
        for candidate in response.candidates or []:
            for part in getattr(candidate.content, "parts", None) or []:
                if getattr(part, "function_call", None):
                    call = part.function_call
                    tool_calls.append(
                        ToolInvocation(
                            name=call.name,
                            arguments=dict(call.args or {}),
                            call_id=getattr(call, "id", None) or call.name,
                        )
                    )
                elif getattr(part, "text", None):
                    text += part.text

        usage = response.usage_metadata
        return LLMMessage(
            content=text,
            sender=self,
            tool_calls=tool_calls or None,
            _meta={
                "input_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None),
            },
        )
