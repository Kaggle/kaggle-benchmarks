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
"""Defines a chat agent that interacts with a Large Language Model (LLM).

The core class is `LLMChat`, which provides a unified interface for sending
messages, handling structured outputs, managing tool calls, and processing
multimodal inputs.

The primary entry point for interaction is the `prompt` method, which handles the
conversation loop, including:
1.  Sending user input (text and optional images).
2.  Invoking the LLM.
3.  Executing requested tools and feeding results back to the LLM.
4.  Parsing the final response into a requested schema (str, Pydantic model, etc.).

Design Note:
    The `LLMChat` class is designed to be stateless. It does not hold any
    conversation history or configuration like temperature settings internally.
    Instead, all state, including system instructions and the sequence of
    messages, is managed within the current `chats.Chat` context.

    Methods like `prompt()` are stateful in their interaction with this context.
    They append messages to the current chat history and trigger LLM responses,
    effectively advancing the conversational state. This design allows for clean
    separation of concerns and enables features like nested conversation threads.


Examples:

    # 1. Basic Text Interaction
    >>> llm.prompt("What is the capital of France?")
    'Paris'

    # 2. Structured Output
    >>> class Sentiment(pydantic.BaseModel):
    ...     score: float
    ...     label: str
    >>> llm.prompt("I love this library!", schema=Sentiment, system="...")
    Sentiment(score=0.9, label='positive')

    # 3. Tool Calling
    >>> def roll_dice(sides: int) -> int:
    ...     return 4  # chosen by fair dice roll
    >>> llm.prompt("Roll a dice", tools=[roll_dice])
    'You rolled a 4.'

    # 4. Multimodal Input
    >>> image = images.from_url("https://example.com/cat.jpg")
    >>> llm.prompt("What animal is this?", image=image)
    'It is a cat.'

"""

from typing import Any, TypeVar

from kaggle_benchmarks import actors, chats, messages, prompting, utils
from kaggle_benchmarks.content_types import audios, images, videos
from kaggle_benchmarks.llm_messages import LLMMessage, Usage

if TYPE_CHECKING:
    from kaggle_benchmarks import llm_messages

T = TypeVar("T")


class APIError(Exception):
    pass


class ToolInvocationLimitExhausted(Exception):
    pass


class LLMChat(actors.Actor):
    """Base class for chat actors that interact with a Large Language Model API."""

    roles_mapping = {}

    def __init__(
        self,
        *,
        support_structured_outputs: bool = False,
        support_temperature: bool = False,
        support_tool_calling: bool = True,
        support_vision: bool = True,
        postprocessor=lambda x: x,
        **kwargs,
    ):
        kwargs.setdefault("role", "assistant")
        kwargs.setdefault("avatar", "🤖")
        super().__init__(**kwargs)
        self.support_structured_outputs = support_structured_outputs
        self.support_temperature = support_temperature
        self.support_tool_calling = support_tool_calling
        self.support_vision = support_vision
        self.postprocessor = postprocessor

    def prompt(
        self,
        message: str,
        schema: type[T] = str,
        seed: int | None = None,
        temperature: float | None = 0,
        tools: list[Any] | None = None,
        image: images.ImageContent | None = None,
        video: videos.VideoContent | None = None,
        audio: audios.AudioContent | None = None,
        max_tool_calls: int = 5,
    ) -> T:
        """Sends a user message to the LLM and returns the structured response.

        This convenience method handles the entire conversation loop, including sending
        the initial message, managing tool calls, and parsing the final response into
        the desired schema.

        Args:
            message: The user's message.
            schema: The expected Pydantic model or type of the response.
            seed: A random seed for the LLM.
            temperature: The sampling temperature for the LLM.
            tools: A list of tools available to the LLM.
            image: An optional image to include with the message.

        Returns:
            The processed and validated response from the LLM, matching the `schema`.
        """
        from kaggle_benchmarks import tools as tool_utils

        if image is not None:
            if not isinstance(image, images.ImageContent):
                raise TypeError(f"Unsupported image type: {type(image)}")
            if not self.support_vision:
                raise ValueError(f"Vision not supported by {self.name}")
            image.caption = message
            actors.user.send(image)

        elif video is not None:
            if not isinstance(video, videos.VideoContent):
                raise ValueError(f"Unsupported video type: {video!r}")
            actors.user.send(video)
            actors.user.send(message)

        if audio is not None:
            if not isinstance(audio, audios.AudioContent):
                raise ValueError(f"Unsupported audio type: {audio!r}")
            audio.caption = message
            actors.user.send(audio)
        else:
            actors.user.send(message)

        final_response = LLMMessage(
            sender=self, content=None, usage=Usage(0, 0), tool_calls=[]
        )

        try:
            # Fork the chat to isolate the tool-calling loop from the main
            # conversation. This prevents format instructions and tool invocations
            # from appearing in the primary chat history.
            with chats.fork() as subchat:
                final_response.chat = subchat

                for _ in range(max_tool_calls):
                    response = self.respond(
                        schema=schema,
                        seed=seed,
                        temperature=temperature,
                        tools=tools if tools is not None else [],
                    )

                    # final_response.tool_calls.extend(response.tool_calls or [])
                    final_response.content = response.content
                    final_response.usage += response.usage

                    if tools and response.tool_calls:
                        for call in response.tool_calls:
                            result = tool_utils.invoke_tool(call, tools)
                            final_response.tool_calls.append(result)
                            actors.Tool(name=call.name).send(result)
                    else:
                        break
                else:
                    raise ToolInvocationLimitExhausted()
        finally:
            chats.send(final_response)

        return final_response.content

    @chats.emits_message
    def respond(
        self,
        *,
        system: str | None = None,
        schema: type[T] = str,
        temperature: float | None = 0,
        seed: int | None = None,
        tools: list[Any] | None = None,
    ) -> LLMMessage[T]:
        """Generates a response from the LLM, handling schema processing and tool calls."""
        from kaggle_benchmarks import contexts

        if tools and not self.support_tool_calling:
            return self._simulate_tool_calling(
                tools=tools,
                schema=schema,
                system=system,
                temperature=temperature,
                seed=seed,
            )

        ctx = contexts.get_current()
        chat = ctx.chat

        h = prompting.process_schema(schema)
        schema_instructions = next(h)
        if isinstance(schema_instructions, tuple):
            schema_instructions, schema = schema_instructions

        response = self.invoke(
            messages=chat.messages,
            schema_instructions=schema_instructions,
            system=system,
            schema=schema,
            temperature=temperature,
            seed=seed,
            tools=tools,
        )

        response._meta.update(
            chat=chat,
            schema=schema,
            raw_content=response.content,
            temperature=temperature,
            seed=seed,
            tools=tools,
        )

        if not response.content:
            # e.g., waiting for tool invocation or an error occurred.
            return response

        try:
            h.send(
                response.content
            )  # must raise StopIteration by returning the parsed value
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
            raise
        except StopIteration as e:
            # StopIteration is the expected signal for a successful parse.
            response.content = e.value
            response.status = utils.Status.SUCCESS
            chat.append(response)

        return response  # type: ignore

    def invoke(
        self,
        messages: list[messages.Message],
        *,
        schema_instructions: str | None = None,
        schema: type[T] = str,
        system: str | None = None,
        temperature: float | None = 0,
        seed: int | None = None,
        tools: list[Any] | None = None,
    ) -> LLMMessage[str]:
        """Invokes the LLM with a given context, handling structured output simulation."""
        if schema is not str and not self.support_structured_outputs:
            result = self._simulate_structured_response(
                messages=messages,
                system=system,
                temperature=temperature,
                seed=seed,
                tools=tools,
                schema_instructions=schema_instructions,
            )
        else:
            result = self._invoke(
                messages=messages,
                system=system,
                schema=schema,
                temperature=temperature,
                seed=seed,
                tools=tools,
            )
        return self.postprocessor(result)

    def _invoke(
        self,
        messages: list[messages.Message],
        *,
        schema: type[T | str] = str,
        system: str | None = None,
        temperature: float | None = 0,
        seed: int | None = None,
        tools: list[Any] | None = None,
    ) -> LLMMessage[str]:
        """Abstract method for native LLM invocation."""
        raise NotImplementedError

    def __repr__(self):
        name = self.name
        arguments = ", ".join(
            f"{k}={v!r}" for k, v in self.__dict__.items() if k.startswith("support")
        )
        return f"{type(self).__name__}({name=}, {arguments})"

    def _simulate_tool_calling(
        self,
        tools: list[Any],
        schema: type[T],
        system: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> LLMMessage[T]:
        """Simulates tool calling for models that do not support it natively."""
        from kaggle_benchmarks.tools import simulate

        return simulate.simulate_respond_with_tools(
            self,
            tools=tools,
            output_schema=schema,
            system=system,
            temperature=temperature,
            seed=seed,
        )

    def _simulate_structured_response(
        self,
        messages: list[messages.Message],
        *,
        system: str | None = None,
        schema_instructions: str | None = None,
        temperature: float | None = 0,
        seed: int | None = None,
        tools: list[Any] | None = None,
    ) -> LLMMessage[str]:
        """Simulates structured output generation for text-only models."""
        if schema_instructions:
            messages.append(actors.system.send(schema_instructions))

        return self.invoke(
            messages=messages,
            system=system,
            schema=str,
            temperature=temperature,
            seed=seed,
            tools=tools,
        )
