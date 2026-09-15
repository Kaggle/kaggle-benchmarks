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

The abstract ``LLMChat`` lives here. The concrete provider backends live in
sibling modules — ``OpenAI`` in :mod:`kaggle_benchmarks.actors.openai` and
``GoogleGenAI`` in :mod:`kaggle_benchmarks.actors.genai` — and are re-exported
from this module for backward compatibility.

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
import inspect
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, TypeVar

from kaggle_benchmarks import actors, chats, messages, prompting, utils
from kaggle_benchmarks._config import config
from kaggle_benchmarks.content_types import audios, images, videos
from kaggle_benchmarks.tools import base as tool_utils
from kaggle_benchmarks.tools import native

if TYPE_CHECKING:
    from kaggle_benchmarks import llm_messages

T = TypeVar("T")
ReasoningLevel = Literal["none", "low", "medium", "high"]


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


# Backward-compatible re-exports: the concrete provider chats now live in their
# own modules. Imported at the bottom so LLMChat/LLMResponse are already defined
# when those modules import from here.
from kaggle_benchmarks.actors.genai import GoogleGenAI  # noqa: E402
from kaggle_benchmarks.actors.openai import OpenAI  # noqa: E402

__all__ = [
    "GoogleGenAI",
    "LLMChat",
    "LLMResponse",
    "OpenAI",
    "ReasoningLevel",
]
