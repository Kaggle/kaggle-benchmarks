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
import enum
import logging
import mimetypes
import typing
from typing import Any, TypeVar

import pydantic
from google import genai
from google.genai import types

from kaggle_benchmarks import actors, chats, messages, utils
from kaggle_benchmarks import tools as tool_utils
from kaggle_benchmarks.actors import llms
from kaggle_benchmarks.content_types import images

# Define a generic type variable for the output schema
T = TypeVar("T", bound=pydantic.BaseModel)


def _get_usage_meta(
    usage: types.GenerateContentResponseUsageMetadata | None,
) -> llms.Usage | None:
    if usage is None:
        return None
    return llms.Usage(
        input_tokens=usage.prompt_token_count,
        output_tokens=usage.candidates_token_count,
    )


class GoogleGenAI(llms.LLMChat):
    """An actor that interacts with the Google GenAI API (e.g., Gemini)."""

    def __init__(self, client: genai.Client, model: str, **kwargs):
        kwargs.setdefault("name", model)

        super().__init__(**kwargs)
        self.model = model
        self.client = client

    def _convert_to_genai_types(
        self, messages: list[messages.Message]
    ) -> list[types.Content]:
        """Converts internal messages to Google GenAI's `Content` format."""
        raw_messages = []
        for message in messages:
            role = "model" if message.sender.role == "assistant" else "user"

            parts = []
            if isinstance(message.content, str):
                parts.append(types.Part(text=message.content))
            elif isinstance(message.content, images.ImageContent):
                image = message.content
                if image.caption:
                    parts.append(types.Part.from_text(text=image.caption))
                parts.append(
                    types.Part(
                        inline_data=types.Blob(
                            # The API expects the raw base64 string, not bytes.
                            data=image.b64_string,
                            mime_type=image.mime_type,
                        )
                    )
                )

            # Note: The Gemini API is smart enough to process image data URLs even when they are passed as part of a plain text string.
            elif (
                isinstance(message.content, list)
                and message.content
                and isinstance(message.content[0], dict)
            ):
                for item in message.content:
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
                parts.append(types.Part(text=message.text))

            raw_messages.append(types.Content(role=role, parts=parts))

        return raw_messages

    def respond(
        self,
        system: str | None = None,
        schema: type[T | str] = str,
        temperature: float | None = 0,
        seed: int | None = None,
        tools: list[Any] | None = None,
    ) -> llms.LLMMessage[T]:
        if tools and schema is not str:
            # GenAI doesn't support both tools and response_schema simultaneously.
            # As a workaround, we ask model to generate a json and parse it manually.
            if not isinstance(schema, pydantic.BaseModel):
                schema = pydantic.create_model("Response", value=(schema, ...))

            # Temporarily disable structured output support to force tool emulation.
            flag = self.support_structured_outputs
            try:
                self.support_structured_outputs = False
                response = super().respond(
                    system=system,
                    schema=schema,
                    temperature=temperature,
                    seed=seed,
                    tools=tools,
                )
            finally:
                self.support_structured_outputs = flag

            if response.content:
                response.content = response.content.value
            return response

        return super().respond(
            system=system,
            schema=schema,
            temperature=temperature,
            seed=seed,
            tools=tools,
        )

    def _invoke(
        self,
        messages: list[messages.Message],
        *,
        schema: type[T | str] = str,
        system: str | None = None,
        temperature: float | None = 0,
        seed: int | None = None,
        tools: list[Any] | None = None,
    ) -> llms.LLMMessage[str]:
        """Prepares and executes the GenAI API call."""
        raw_messages = self._convert_to_genai_types(messages)

        config_params = {}
        if tools and schema is not str:
            return self.invoke(
                messages=messages,
                schema_instructions=None,
                schema=str,
                system=system,
                temperature=temperature,
                seed=seed,
                tools=tools,
            )

        if system:
            config_params["system_instruction"] = system

        if schema is not str and self.support_structured_outputs:
            config_params["response_json_schema"] = schema.model_json_schema()

            # Determine the correct MIME type based on the schema's type
            is_enum = isinstance(schema, type) and issubclass(schema, enum.Enum)
            is_literal = typing.get_origin(schema) is typing.Literal

            if is_enum or is_literal:
                config_params["response_mime_type"] = "text/x.enum"
            else:
                # Assume any other schema (like a Pydantic model) is for JSON
                config_params["response_mime_type"] = "application/json"

        tools_declaration = None
        if tools and self.support_tool_calling:
            tools_declaration = tools

        config = types.GenerateContentConfig(
            temperature=temperature,
            seed=seed,
            tools=tools_declaration,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False
            ),
            **config_params,
        )

        return self._call_api(contents=raw_messages, config=config)

    def _call_api(
        self, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> llms.LLMMessage[str]:
        response = self.client.models.generate_content(
            model=self.model, contents=contents, config=config
        )
        # Handle cases where the model refuses to respond
        if not response.candidates or not response.candidates[0].content.parts:
            logging.warning(
                "API failed to produce a response for the following request:\n"
                f"model: {self.model}\ncontents: {contents}\nconfig: {config}"
            )
            raise llms.APIError(
                "API failed to produce a response for the following request:\n"
                f"model: {self.model}\ncontents: {contents}\nconfig: {config}"
            )

        tool_calls = self.extract_tool_calls(response)

        for tool_invocation in self._iter_tool_calls(response):
            chats.send(messages.Message(sender=actors.Tool(), content=tool_invocation))

        return llms.LLMMessage(
            sender=self,
            content=response.text,
            tool_calls=tool_calls if tool_calls else None,
            usage=_get_usage_meta(response.usage_metadata),
        )

    def extract_tool_calls(self, response):
        tool_calls = list(self._iter_tool_calls(response))
        for part in response.candidates[0].content.parts:
            if part.function_call:
                tool_calls.append(
                    tool_utils.ToolInvocation(
                        name=part.function_call.name,
                        call_id=f"call_{part.function_call.name}",
                        arguments=part.function_call.args,
                    )
                )
        return tool_calls

    def _iter_tool_calls(self, response):
        # TODO: review this function for potentiall issues
        calls = []
        if response.automatic_function_calling_history:
            for item in response.automatic_function_calling_history:
                for part in item.parts:
                    if part.function_call:
                        calls.append(part.function_call)
                    if part.function_response:
                        yield tool_utils.ToolInvocationResult(
                            name=part.function_response.name,
                            call_id=f"call_{part.function_response.name}",
                            arguments=calls.pop(0).args,
                            output=part.function_response.response["result"],
                        )

                    if not part.function_call and not part.function_response:
                        logging.warning(f"Unknown part {part}")


class StreamingGoogleGenAI(GoogleGenAI):
    """A `GoogleGenAI` actor that handles streaming responses."""

    def _call_api(
        self, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> llms.LLMMessage:
        response_stream = self.client.models.generate_content_stream(
            model=self.model, contents=contents, config=config
        )
        msg = llms.LLMMessage(sender=self, content="")
        usage = None
        tool_calls = None
        for chunk in response_stream:
            if isinstance(chunk.text, str):
                msg.add_chunk(chunk.text)
            usage = _get_usage_meta(chunk.usage_metadata)

            if isinstance(chunk, types.GenerateContentResponse):
                tool_calls = self.extract_tool_calls(chunk)

        msg.usage = usage
        msg.tool_calls = tool_calls
        return msg


class ModelProxyGenAI(GoogleGenAI):
    """A `GoogleGenAI` actor variant for use with a model proxy.

    This class may include workarounds for specific proxy behaviors.
    """

    def __init__(self, client: genai.Client, model: str, **kwargs):
        if "gemini" in model:
            kwargs.setdefault("support_structured_outputs", True)

            if "gemini-2.5-flash" in model:
                # The proxy returns a 400 error if tools are set with this model.
                kwargs["support_tool_calling"] = False

        elif "deepseek" in model:
            kwargs["postprocessor"] = utils.extract_thinking_tag
            kwargs.setdefault("support_structured_outputs", False)
            kwargs.setdefault("support_tool_calling", False)
            kwargs.setdefault("support_vision", "r1" not in model)

        elif "anthropic" in model:
            # kwargs.setdefault("support_structured_outputs", False)
            kwargs["support_structured_outputs"] = False
            kwargs.setdefault("support_tool_calling", False)
            kwargs["postprocessor"] = utils.extract_json_tag

        elif "gemma" in model:
            kwargs.setdefault("support_vision", True)
        else:
            kwargs.setdefault("support_structured_outputs", False)
            kwargs.setdefault("support_tool_calling", False)
            kwargs.setdefault("support_vision", False)

        super().__init__(client, model, **kwargs)
