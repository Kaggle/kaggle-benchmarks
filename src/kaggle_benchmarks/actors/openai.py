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

"""The OpenAI (Model Proxy) LLM backend."""

import json
import logging
import re
from typing import Any, Callable, Iterator

import openai

from kaggle_benchmarks import actors, messages, utils
from kaggle_benchmarks.actors.llms import (
    LLMChat,
    LLMResponse,
    ReasoningLevel,
    _extract_extra_usage_metadata,
)
from kaggle_benchmarks.serializers import openai as openai_serializer
from kaggle_benchmarks.tools import functions

logger = logging.getLogger(__name__)

_THINK_TAG_PATTERN = re.compile(r"<think>\n?(.*?)\n?</think>\n*", re.DOTALL)

_GROK_VERSION_RE = re.compile(r"^xai/grok-(\d+)(?:\.(\d+))?")
# Grok 4.5 is the first release to reject the `seed` parameter through Model Proxy.
_MIN_SEEDLESS_GROK_VERSION = (4, 5)


def _is_seedless_grok(model: str) -> bool:
    """Reports whether a Grok model is new enough to reject `seed`."""
    match = _GROK_VERSION_RE.match(model)
    if not match:
        return False
    major, minor = match.groups()
    return (int(major), int(minor or 0)) >= _MIN_SEEDLESS_GROK_VERSION


def _parse_think_tags(content: str) -> tuple[str, str | None]:
    """Extracts all <think>...</think> blocks from content.

    Model Proxy wraps reasoning traces in <think> tags inside content
    because the chat completions spec doesn't have a dedicated field
    for reasoning traces.
    """
    segments = _THINK_TAG_PATTERN.findall(content)
    if not segments:
        return content, None
    remaining = _THINK_TAG_PATTERN.sub("", content).strip()
    thinking = "\n\n".join(s.strip() for s in segments if s.strip())
    return remaining, thinking or None


class OpenAI(LLMChat):
    def __init__(self, client: openai.OpenAI, model: str, **kwargs):
        kwargs.setdefault("name", model)
        super().__init__(**kwargs)
        self.model = model
        self.client = client
        self.serializer = openai_serializer.ModelProxyOpenAISerializer(
            roles_mapping={"tool": "system"}
        )

    def _get_usage_meta(
        self, usage: openai.types.CompletionUsage | None
    ) -> dict[str, Any]:
        """Extracts token usage metadata from an OpenAI response object."""
        if usage is None:
            return {}
        return {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            **_extract_extra_usage_metadata(usage),
        }

    def _should_remove_seed(self) -> bool:
        unsupported_prefixes = (
            "google/",
            "openai/gpt-5.4-pro",
            "openai/gpt-5.6",
        )
        if any(self.model.startswith(prefix) for prefix in unsupported_prefixes):
            return True
        return _is_seedless_grok(self.model)

    def invoke(
        self,
        messages: list[messages.Message],
        system: str | None,
        reasoning: ReasoningLevel | None = None,
        tools: list[Callable] | None = None,
        **kwargs,
    ) -> LLMResponse | Iterator[LLMResponse]:
        if system:
            from kaggle_benchmarks.messages import Message

            messages = [Message(sender=actors.system, content=system)] + messages

        raw_messages = list(self.serializer.dump_messages(messages))

        # Convert callables to OpenAI tool schemas.
        if tools:
            kwargs["tools"] = [functions.function_to_openai_tool(t) for t in tools]

        if self._should_remove_seed():
            # TODO(b/430112500): Remove once model proxy supports it for AIS backends.
            # Temporarily do not send "seed" parameter for models not supporting it in Model Proxy.
            kwargs.pop("seed", None)

        if reasoning is not None:
            # extra_api_params takes precedence if reasoning_effort was
            # already set by the caller.
            kwargs.setdefault("reasoning_effort", reasoning)
            # The double-nested extra_body is intentional: the outer one is
            # consumed by the OpenAI SDK (merged into the request body), the
            # inner one arrives at Model Proxy as a top-level field where it
            # reads google.thinking_config.  Without include_thoughts, the
            # frontend drops thinking traces from the response.
            if kwargs["reasoning_effort"] != "none" and self.model.startswith(
                "google/"
            ):
                kwargs.setdefault("extra_body", {})
                kwargs["extra_body"].setdefault("extra_body", {})
                kwargs["extra_body"]["extra_body"].setdefault("google", {})
                kwargs["extra_body"]["extra_body"]["google"].setdefault(
                    "thinking_config", {"include_thoughts": True}
                )

        return self._call_api(raw_messages, **kwargs)

    def _get_stream_response(
        self, response_stream: openai.Stream
    ) -> Iterator[LLMResponse]:
        """Yields LLMResponse objects from a streaming response."""
        # TODO: Streaming does not capture reasoning_traces. Think tags in
        # streamed chunks are not parsed, so last_reasoning_traces() will
        # return None when streaming is enabled.
        for chunk in response_stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Guard against chunks where 'delta' is None
            if not delta:
                continue

            yield LLMResponse(
                content=delta.content or "",
                tool_calls=delta.tool_calls,
                meta=self._get_usage_meta(chunk.usage),
            )

    def _call_api(
        self, messages: list[dict[str, str]], **kwargs
    ) -> LLMResponse | Iterator[LLMResponse]:
        if self.support_structured_outputs and "response_format" in kwargs:
            # quickfix for nested models in ModelProxy API
            if utils.has_nested_models(kwargs["response_format"]):
                method = self.client.chat.completions.create
                response_format = kwargs.pop("response_format")
                json_schema = json.dumps(response_format.model_json_schema())
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The output must be a valid JSON object that strictly adheres to the following JSON schema:\n"
                            f"{json_schema}"
                        ),
                    }
                )
            else:
                method = self.client.beta.chat.completions.parse
        else:
            if self.stream_responses:
                kwargs["stream"] = True

            method = self.client.chat.completions.create

        response = method(
            model=self.model,
            messages=messages,
            **kwargs,
        )

        if isinstance(response, openai.Stream):
            return self._get_stream_response(response)
        else:
            # Handle cases where the API returns no choices (e.g.
            # Anthropic models proxied through the OpenAI endpoint can
            # return choices=None on multi-turn tool conversations).
            if not response.choices:
                return LLMResponse(
                    content="",
                    meta=self._get_usage_meta(response.usage),
                )
            # Handle choices[0].message being None (observed intermittently
            # from Model Proxy — see issue #191).
            if response.choices[0].message is None:
                logger.warning(
                    "Model Proxy returned choices[0].message=None for "
                    "model %s; treating as empty response.",
                    self.model,
                )
                return LLMResponse(
                    content="",
                    meta=self._get_usage_meta(response.usage),
                )
            message = response.choices[0].message
            tool_calls = message.tool_calls
            content = message.content or ""
            # The OpenAI chat completions spec doesn't support a dedicated
            # reasoning_content field, so Model Proxy embeds thinking traces
            # in content using <think> tags.  Only parse when reasoning was
            # requested to avoid stripping literal <think> tags from content.
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning is None and kwargs.get("reasoning_effort") not in (
                None,
                "none",
            ):
                content, reasoning = _parse_think_tags(content)
            return LLMResponse(
                content=content,
                reasoning_traces=reasoning,
                tool_calls=[t.model_dump() for t in tool_calls] if tool_calls else None,
                meta=self._get_usage_meta(response.usage),
            )
