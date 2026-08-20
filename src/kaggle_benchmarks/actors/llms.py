# Copyright 2025 Kaggle Inc.
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

"""
Defines a chat agent that interacts with a Large Language Model (LLM).

Design Note:
    LLMChat is stateless. No system instructions or temperature settings are
    managed within the class itself. All state, including instructions and
    chat history, is maintained within the `chats.Chat` object. This allows
    for clean separation of concerns and enables nested threads to encapsulate
    inner chat history, preventing it from being visible or sent to the LLM
    in outer threads.


Example:

class Goose(LLMChat):
    def __init__(self, sound):
        super().__init__(name="goose", avatar="🪿")
        self.sound = sound

    def invoke(self, messages, system: str = ""):
        return LLMResponse(content=self.sound if not system else system)

goose = Goose('honk')


print(goose.send("Hi"))
# 🪿 [goose]: Hi


with chats.new(system_instructions="quack") as t:
    goose.send("Hi!")
    goose.respond()
    goose.send("What's up?")
    goose.respond()

print(t)
# 🧵Chat:
#   ⚙️ [System]: quack
#   🪿 [goose]: Hi!
#   🪿 [goose]: honk
#   🪿 [goose]: What's up?
#   🪿 [goose]: quack quack

# system message is separately managed by chats module, so goose doesn't use system message of the Chat obj unless explicitly passed in to respond()

with chats.new(name="Outer") as outer_t:
    goose.send("Outer message 1")
    goose.respond()
    with chats.new(name="Inner", system_instructions="inner") as inner_t:
        goose.send("Inner message 1")
        goose.respond()
        goose.send("Inner message 2")
        goose.respond()
    goose.send("Outer message 2")
    goose.respond()


# Inner messages are not part of the outer chat's history.
print(outer_t)
# 🧵Outer:
#   🪿 [goose]: Outer message 1
#   🪿 [goose]: honk
#   🧵Inner:
#     ⚙️ [System]: inner
#     🪿 [goose]: Inner message 1
#     🪿 [goose]: honk
#     🪿 [goose]: Inner message 2
#     🪿 [goose]: honk
#   🪿 [goose]: Outer message 2
#   🪿 [goose]: honk

"""

import dataclasses
import enum
import inspect
import json
import logging
import re
import typing
import uuid
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, TypeVar

import openai
from google import genai
from google.genai import types

from kaggle_benchmarks import actors, chats, messages, prompting, utils
from kaggle_benchmarks._config import config
from kaggle_benchmarks.content_types import audios, images, videos
from kaggle_benchmarks.serializers import genai as genai_serializer
from kaggle_benchmarks.serializers import openai as openai_serializer
from kaggle_benchmarks.tools import base as tool_utils
from kaggle_benchmarks.tools import functions, native

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from kaggle_benchmarks import llm_messages

T = TypeVar("T")
ReasoningLevel = Literal["none", "low", "medium", "high"]

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


# TODO: Figure out a more robust way to handle extra fields.
def _extract_extra_usage_metadata(usage: Any) -> dict[str, Any]:
    """Extracts cost metadata from a usage object augmented by Model Proxy."""
    cost = getattr(usage, "cost", None) or {}
    return {
        "input_tokens_cost_nanodollars": cost.get("input_tokens_cost_nanodollars"),
        "output_tokens_cost_nanodollars": cost.get("output_tokens_cost_nanodollars"),
        "total_backend_latency_ms": getattr(usage, "total_backend_latency_ms", None),
    }


@dataclasses.dataclass(frozen=True)
class LLMResponse:
    content: str
    reasoning_traces: str | None = None
    tool_calls: list[Any] | None = None
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)


class LLMChat(actors.Actor):
    """A chat agent that interacts with a Large Language Model (LLM)."""

    def __init__(
        self,
        *,
        support_structured_outputs: bool = False,
        support_temperature: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("role", "assistant")
        kwargs.setdefault("avatar", "🤖")
        super().__init__(**kwargs)
        self.support_structured_outputs = support_structured_outputs
        self.support_temperature = support_temperature
        self.stream_responses = config.interactive_mode

    def invoke(
        self,
        messages: list[messages.Message],
        system: str | None,
        reasoning: ReasoningLevel | None = None,
        tools: list[Callable] | None = None,
        **kwargs,
    ) -> LLMResponse | Iterator[LLMResponse] | "llm_messages.LLMMessage[str]":
        """Invokes the LLM with the given messages and system instructions."""
        raise NotImplementedError

    def prompt(
        self,
        message: str,
        schema: type[T] = str,
        seed: int = 0,
        temperature: float = 0,
        tools: list[Callable] | None = None,
        image: images.ImageContent | None = None,
        video: videos.VideoContent | None = None,
        audio: audios.AudioContent | None = None,
        reasoning: ReasoningLevel | None = None,
        extra_api_params: dict[str, Any] | None = None,
    ) -> T:
        """Sends a message to the LLM and returns the parsed response.

        Args:
            extra_api_params: Additional provider-specific API parameters
                (e.g. top_p, max_tokens, max_output_tokens). Cannot include
                parameters already on prompt() or respond() signatures
                (seed, temperature, schema, system, etc.).
        """
        # Guard: prompt() bypasses perspective projection — it must not
        # be called when a ChatRoom is the active context.
        from kaggle_benchmarks.rooms import ChatRoom

        current = chats.get_current_chat()
        if isinstance(current, ChatRoom):
            raise RuntimeError(
                f"LLMChat.prompt() cannot be called inside an active ChatRoom "
                f"('{current.name}'). For a participant's turn, use "
                f"participant.reply(). For a side query outside the room "
                f"conversation, exit the room context (or use chats.new() "
                f"for an isolated side-chat)."
            )
        if image is not None:
            match image:
                case images.ImageURL():
                    image_to_send = images.from_image_url(image)
                case images.ImageBase64():
                    image_to_send = image
                case _:
                    raise ValueError(f"Unsupported image type: {type(image)}")

            actors.user.send(image_to_send)

        if video is not None:
            if not isinstance(video, videos.VideoContent):
                raise ValueError(f"Unsupported video type: {video!r}")
            actors.user.send(video)

        if audio is not None:
            if not isinstance(audio, audios.AudioContent):
                raise ValueError(f"Unsupported audio type: {audio!r}")
            actors.user.send(audio)

        actors.user.send(message)

        extra = extra_api_params or {}
        _reserved = (
            set(inspect.signature(type(self).prompt).parameters)
            | set(inspect.signature(type(self).respond).parameters)
        ) - {"self", "message", "extra_api_params", "kwargs"}
        conflicts = set(extra) & _reserved
        if conflicts:
            raise ValueError(
                f"{conflicts} cannot be set via extra_api_params. "
                f"Use the corresponding prompt() parameter instead."
            )

        kwargs = {
            "seed": seed,
            "temperature": temperature if self.support_temperature else None,
            "reasoning": reasoning,
        }

        if tools:
            response = native.native_tool_agent(
                self, tools, schema=schema, **kwargs, **extra
            )
        else:
            response = self.respond(schema=schema, **kwargs, **extra)

        return response.content

    @chats.emits_message
    def respond(
        self,
        system: str | None = None,
        schema: type[T] = str,
        *,
        input_messages: list[messages.Message] | None = None,
        sender: actors.Actor | None = None,
        **kwargs,
    ) -> messages.Message[T]:
        """Generate a response from the active chat's history.

        Args:
            system: System prompt for this call. Replaces any system
                prompt already in the chat.
            schema: Output schema. Defaults to ``str``.
            input_messages: Override the message history sent to the LLM.
                For ChatRoom use only — it passes the viewer-projected
                history here. Benchmark code should not set this; the
                active chat's history is used by default.
            sender: Override the ``sender`` recorded on the response
                message. For ChatRoom use only — it passes the speaking
                ``Participant`` here so the message carries the right
                identity from construction. Defaults to ``self`` (the
                ``LLMChat``).
        """
        from kaggle_benchmarks import contexts, llm_messages

        ctx = contexts.get_current()
        chat = ctx.chat

        h = prompting.process_schema(schema)

        temp_messages = []

        schema_instructions = next(h)
        match schema_instructions:
            case [msg, schema]:
                if self.support_structured_outputs:
                    kwargs["response_format"] = schema
                else:
                    temp_messages.append(
                        messages.Message(sender=actors.system, content=msg)
                    )
            case None:
                pass
            case _:
                temp_messages.append(
                    messages.Message(sender=actors.system, content=schema_instructions)
                )

        response = messages.Message(
            sender=sender or self,
            content="",
            _status=utils.Status.RUNNING,
        )

        raw_messages = [
            msg
            for msg in (input_messages if input_messages is not None else chat.messages)
            if msg.is_visible_to_llm
        ] + temp_messages

        invoke_response = self.invoke(
            raw_messages,
            system=system,
            **kwargs,
        )
        if isinstance(invoke_response, LLMResponse):
            # A response can have either content, tool_calls, or both in some cases.
            response.content = invoke_response.content or ""
            # Normalize provider dicts → typed ToolInvocation for downstream.
            if invoke_response.tool_calls:
                response._meta["tool_calls"] = [
                    tool_utils.ToolInvocation.from_api_dict(tc)
                    for tc in invoke_response.tool_calls
                ]
            else:
                response._meta["tool_calls"] = None
            response._meta.update(invoke_response.meta)
            response._meta["reasoning_traces"] = invoke_response.reasoning_traces
            response.status = utils.Status.SUCCESS
            chat.append(response)
        elif isinstance(invoke_response, Iterator):
            # Append before streaming so UIs see new_message (with empty
            # content + RUNNING status) before chunks arrive — lets them
            # render the message header before tokens stream in.
            chat.append(response)
            response.stream(invoke_response)
            response.status = utils.Status.SUCCESS
        elif isinstance(invoke_response, llm_messages.LLMMessage):
            response = invoke_response
            # Set sender before append: chat.append fires the new_message
            # event, so the sender must already be the Participant (not the
            # backing LLMChat) by then.
            response.sender = sender or self
            chat.append(response)
        else:
            raise TypeError("Unknown response type from LLM.")

        answer = response.content
        response._meta.update(chat=chat, schema=schema, raw_content=answer, **kwargs)

        try:
            h.send(answer)  # must raise StopIteration by returning the parsed value
            raise prompting.SchemaError(
                f"Generator for {schema!r} yielded multiple values, expected only one."
            )
        except prompting.ResponseParsingError as e:
            chat.append(
                messages.Message(
                    str(e),
                    sender=actors.system,
                )
            )
            response.status = utils.Status.FAILED
            raise e

        except StopIteration as e:
            # StopIteration is expected as this is how you get returned value
            # from a generator. The message has already been appended and its
            # status set above; here we just refine .content to the parsed
            # schema value (may differ from the raw text the UI rendered).
            response.content = e.value

        return response

    def __repr__(self):
        name = self.name
        return f"{type(self).__name__}({name=})"


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
