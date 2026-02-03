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

import base64
import dataclasses
import enum
import mimetypes
import typing
from typing import Any, Iterator, TypeVar

import openai
from google import genai
from google.genai import types

from kaggle_benchmarks import actors, chats, messages, prompting, utils
from kaggle_benchmarks._config import config
from kaggle_benchmarks.content_types import images

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class LLMResponse:
    content: str
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
        self, messages: list[messages.Message], system: str | None, **kwargs
    ) -> LLMResponse | Iterator[LLMResponse]:
        """Invokes the LLM with the given messages and system instructions."""
        raise NotImplementedError

    def prompt(
        self,
        message: str,
        schema: type[T] = str,
        seed: int = 0,
        temperature: float = 0,
        tools: list[Any] | None = None,
        image: images.ImageContent | None = None,
    ) -> T:
        if image is not None:
            match image:
                case images.ImageURL():
                    image_to_send = images.from_image_url(image)
                case images.ImageBase64():
                    image_to_send = image
                case _:
                    raise ValueError(f"Unsupported image type: {type(image)}")

            actors.user.send(image_to_send)

        actors.user.send(message)
        return self.respond(
            schema=schema,
            seed=seed,
            temperature=temperature if self.support_temperature else None,
            tools=tools if tools is not None else [],
        ).content

    @chats.emits_message
    def respond(
        self,
        system: str | None = None,
        schema: type[T] = str,
        **kwargs,
    ) -> messages.Message[T]:
        from kaggle_benchmarks import contexts

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
            sender=self,
            content="",
            _status=utils.Status.RUNNING,
        )

        raw_messages = [
            msg for msg in chat.messages if msg.is_visible_to_llm
        ] + temp_messages

        invoke_response = self.invoke(
            raw_messages,
            system=system,
            **kwargs,
        )
        if isinstance(invoke_response, LLMResponse):
            # A response can have either content, tool_calls, or both in some cases.
            response.content = invoke_response.content or ""
            response._meta["tool_calls"] = invoke_response.tool_calls
            response._meta.update(invoke_response.meta)
        elif isinstance(invoke_response, Iterator):
            response.stream(invoke_response)
        else:
            raise TypeError("Unknown response type from LLM.")

        answer = response.content
        response._meta.update(chat=chat, schema=schema, raw_content=answer, **kwargs)

        try:
            h.send(answer)  # must raise StopIteration by returning the parsed value
            raise prompting.SchemaProcessingError(
                f"Generator for {schema!r} yielded multiple values, expected only one."
            )
        except prompting.ResponseParsingError as e:
            chat.append(
                messages.Message(
                    f"Error processing response {answer}:\n{e}",
                    sender=actors.system,
                )
            )
            response.status = utils.Status.FAILED
            raise e

        except StopIteration as e:
            # StopIteration is expected as this is how you get returned value from a generator
            response.content = e.value
            response.status = utils.Status.SUCCESS
            chat.append(response)

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

    def _get_usage_meta(
        self, usage: openai.types.CompletionUsage | None
    ) -> dict[str, Any]:
        """Extracts token usage metadata from an OpenAI response object."""
        if usage is None:
            return {}
        return {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
        }

    def invoke(
        self, messages: list[messages.Message], system: str | None, **kwargs
    ) -> LLMResponse | Iterator[LLMResponse]:
        raw_messages = self._get_raw_messages(messages)
        if system:
            raw_messages = [{"role": "system", "content": system}] + raw_messages

        if self.model.startswith("google/"):
            # TODO(b/430112500): Remove once model proxy supports it for AIS backends.
            # Temporarily do not send "seed" parameter for google models when using Model Proxy.
            kwargs.pop("seed", None)

        return self._call_api(raw_messages, **kwargs)

    def _get_raw_messages(self, messages: list[messages.Message]):
        return [
            {
                "role": message.sender.role
                if message.sender.role != "tool"
                else "system",  # TODO: Remove this renaming once ModelProxy supports tools
                "content": message.payload,
            }
            for message in messages
        ]

    def _get_stream_response(
        self, response_stream: openai.Stream
    ) -> Iterator[LLMResponse]:
        """Yields LLMResponse objects from a streaming response."""
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
            message = response.choices[0].message
            tool_calls = message.tool_calls
            return LLMResponse(
                content=message.content or "",
                tool_calls=[t.model_dump() for t in tool_calls] if tool_calls else None,
                meta=self._get_usage_meta(response.usage),
            )


class GoogleGenAI(LLMChat):
    def __init__(self, client: genai.Client, model: str, **kwargs):
        kwargs.setdefault("name", model)
        super().__init__(**kwargs)
        self.model = model
        self.client = client

    def _get_usage_meta(self, usage: types.UsageMetadata | None) -> dict[str, Any]:
        if usage is None:
            return {}
        return {
            "input_tokens": usage.prompt_token_count,
            "output_tokens": usage.candidates_token_count,
        }

    def _get_raw_messages(self, messages: list[messages.Message]):
        """Converts benchmark messages to Google GenAI's Content format."""
        raw_messages = []
        for message in messages:
            role = "model" if message.sender.role == "assistant" else "user"
            payload = message.payload

            parts = []
            if isinstance(payload, str):
                parts.append(types.Part(text=payload))

            # Note: The Gemini API is smart enough to process image data URLs even when they are passed as part of a plain text string.
            elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
                for item in payload:
                    if item.get("type") == "image_url":
                        url = item["image_url"]["url"]

                        image_bytes = None
                        mime_type = "image/jpeg"
                        if url.startswith("data:"):
                            # Handle base64 data URLs
                            header, b64_string = url.split(",", 1)
                            mime_type = header.split(";")[0].split(":")[1]
                            image_bytes = base64.b64decode(b64_string)
                        else:
                            # Handle remote http/https URLs
                            b64_string = images.image_url_to_base64(url)
                            image_bytes = base64.b64decode(b64_string)
                            mime_type = mimetypes.guess_type(url)[0] or "image/jpeg"

                        if image_bytes:
                            parts.append(
                                types.Part.from_bytes(
                                    data=image_bytes, mime_type=mime_type
                                )
                            )
            else:
                # Fallback for any other unexpected payload types
                parts.append(types.Part(text=str(payload)))

            raw_messages.append(types.Content(role=role, parts=parts))

        return raw_messages

    def _get_stream_response(
        self, response_stream: Iterator[types.GenerateContentResponse]
    ) -> Iterator[LLMResponse]:
        # We currently only support text outputs
        for chunk in response_stream:
            yield LLMResponse(
                content=chunk.text or "",
                meta=self._get_usage_meta(chunk.usage_metadata),
            )

    def invoke(
        self, messages: list[messages.Message], system: str | None, **kwargs
    ) -> LLMResponse | Iterator[LLMResponse]:
        raw_messages = self._get_raw_messages(messages)

        config_params = {}
        if system:
            config_params["system_instruction"] = system
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

            return LLMResponse(
                content=response.text,
                meta=self._get_usage_meta(response.usage_metadata),
            )
