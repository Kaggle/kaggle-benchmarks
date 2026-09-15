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

"""The Google GenAI LLM backend."""

import enum
import json
import typing
import uuid
from types import SimpleNamespace
from typing import Any, Callable, Iterator

from google import genai
from google.genai import types

from kaggle_benchmarks import messages
from kaggle_benchmarks.actors.llms import (
    LLMChat,
    LLMResponse,
    ReasoningLevel,
    _extract_extra_usage_metadata,
)
from kaggle_benchmarks.serializers import genai as genai_serializer
from kaggle_benchmarks.tools import functions


class GoogleGenAI(LLMChat):
    def __init__(self, client: genai.Client, model: str, **kwargs):
        kwargs.setdefault("name", model)
        super().__init__(**kwargs)
        self.model = model
        self.client = client
        self.serializer = genai_serializer.GenAISerializer(
            roles_mapping={"assistant": "model", "system": "user", "tool": "user"}
        )

    def _get_usage_meta(self, usage: types.UsageMetadata | None) -> dict[str, Any]:
        if usage is None:
            return {}
        return {
            "input_tokens": usage.prompt_token_count,
            "output_tokens": usage.candidates_token_count,
            **_extract_extra_usage_metadata(usage),
        }

    # "medium" exists for OpenAI's `reasoning_effort`; GenAI's ThinkingLevel
    # enum only has LOW/HIGH, so "MEDIUM" triggers a UserWarning from the
    # SDK. Left as-is to surface the asymmetry rather than silently round.
    _REASONING_LEVEL_MAP = {
        "none": None,
        "low": "LOW",
        "medium": "MEDIUM",
        "high": "HIGH",
    }

    def _split_response(
        self,
        response: types.GenerateContentResponse,
    ) -> tuple[str, str | None]:
        """Splits a response into content text and thinking text."""
        if not response.candidates or not response.candidates[0].content:
            return "", None
        parts = response.candidates[0].content.parts or []
        content_segments = []
        thinking_segments = []
        for part in parts:
            if not part.text:
                continue
            if getattr(part, "thought", False):
                thinking_segments.append(part.text)
            else:
                content_segments.append(part.text)
        content = "".join(content_segments) if content_segments else ""
        thinking = "".join(thinking_segments) if thinking_segments else None
        return content, thinking

    def _extract_text(self, response: types.GenerateContentResponse) -> str:
        """Extracts non-thought text content from a response."""
        content, _ = self._split_response(response)
        return content

    def _get_stream_response(
        self, response_stream: Iterator[types.GenerateContentResponse]
    ) -> Iterator[LLMResponse]:
        # TODO: Streaming does not capture reasoning_traces. Thought parts
        # are filtered out by _extract_text, so last_reasoning_traces() will
        # return None when streaming is enabled.
        for chunk in response_stream:
            tool_calls = None
            if chunk.candidates and chunk.candidates[0].content:
                parts = chunk.candidates[0].content.parts or []
                fn_parts = [p for p in parts if p.function_call]
                if fn_parts:
                    tool_calls = []
                    # Per-chunk enumerate assumes GenAI emits each call atomically.
                    # If calls ever span chunks, switch to a per-stream counter.
                    for i, part in enumerate(fn_parts):
                        fc = part.function_call
                        tc_chunk = SimpleNamespace(
                            index=i,
                            id=fc.id or f"call_{uuid.uuid4().hex[:8]}",
                            function=SimpleNamespace(
                                name=fc.name,
                                arguments=json.dumps(dict(fc.args))
                                if fc.args
                                else "{}",
                            ),
                            thought_signature=getattr(part, "thought_signature", None),
                            thought=getattr(part, "thought", None),
                        )
                        tool_calls.append(tc_chunk)

            yield LLMResponse(
                content=self._extract_text(chunk)
                if chunk.candidates
                else (chunk.text or ""),
                tool_calls=tool_calls,
                meta=self._get_usage_meta(chunk.usage_metadata),
            )

    def invoke(
        self,
        messages: list[messages.Message],
        system: str | None,
        reasoning: ReasoningLevel | None = None,
        tools: list[Callable] | None = None,
        **kwargs,
    ) -> LLMResponse | Iterator[LLMResponse]:
        raw_messages = list(self.serializer.dump_messages(messages))

        config_params = {}
        if system:
            config_params["system_instruction"] = system

        # Convert callables to GenAI FunctionDeclarations.
        if tools:
            config_params["tools"] = [
                types.Tool(
                    function_declarations=[
                        functions.function_to_genai_tool(t) for t in tools
                    ]
                )
            ]

        if reasoning is not None and "thinking_config" not in kwargs:
            # Only set thinking_config if not already overwritten by the caller.
            level = self._REASONING_LEVEL_MAP[reasoning]
            if level is None:
                config_params["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=0,
                )
            else:
                config_params["thinking_config"] = types.ThinkingConfig(
                    thinking_level=level,
                    include_thoughts=True,
                )

        if "response_format" in kwargs:
            schema = kwargs.pop("response_format")
            config_params["response_schema"] = schema

            # Determine the correct MIME type based on the schema's type
            is_enum = isinstance(schema, type) and issubclass(schema, enum.Enum)
            is_literal = typing.get_origin(schema) is typing.Literal

            if is_enum or is_literal:
                config_params["response_mime_type"] = "text/x.enum"
            else:
                # Assume any other schema (like a Pydantic model) is for JSON
                config_params["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**kwargs, **config_params)

        return self._call_api(contents=raw_messages, config=config)

    def _call_api(
        self, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> LLMResponse | Iterator[LLMResponse]:
        if self.stream_responses:
            response_stream = self.client.models.generate_content_stream(
                model=self.model, contents=contents, config=config
            )
            return self._get_stream_response(response_stream)
        else:
            response = self.client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
            # Handle cases where the model refuses to respond
            if not response.candidates:
                return LLMResponse(
                    content="",
                    meta=self._get_usage_meta(response.usage_metadata),
                )

            content, thinking = self._split_response(response)

            # Extract function calls from response parts and normalize to
            # OpenAI-style dicts so native_tool_agent() can use
            # ToolInvocation.from_api_dict() uniformly for both backends.
            parts = response.candidates[0].content.parts or []
            fn_parts = [p for p in parts if p.function_call]
            tool_calls = None
            if fn_parts:
                tool_calls = []
                for part in fn_parts:
                    fc = part.function_call
                    tc: dict[str, Any] = {
                        "id": fc.id or f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": dict(fc.args) if fc.args else {},
                        },
                    }
                    # TODO(genai-sdk): thought_signature is SDK-internal,
                    # required for Gemini 3.x round-tripping. Revisit once
                    # the GenAI SDK stabilizes the thought API.
                    if getattr(part, "thought_signature", None):
                        tc["_thought_signature"] = part.thought_signature
                    if getattr(part, "thought", None):
                        tc["_thought"] = part.thought
                    tool_calls.append(tc)

            return LLMResponse(
                content=content,
                reasoning_traces=thinking,
                tool_calls=tool_calls,
                meta=self._get_usage_meta(response.usage_metadata),
            )
