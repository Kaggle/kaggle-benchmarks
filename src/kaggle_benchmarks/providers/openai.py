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
from typing import Any, TypeVar

import openai
import pydantic
from openai.types import responses as responses_types

from kaggle_benchmarks import messages, utils
from kaggle_benchmarks.actors import llms
from kaggle_benchmarks.serializers import openai as openai_serializer

T = TypeVar("T")


def parse_usage(
    usage: openai.types.CompletionUsage | responses_types.ResponseUsage,
) -> llms.Usage:
    """Converts an OpenAI usage object to the internal `llms.Usage` format."""
    if isinstance(usage, openai.types.CompletionUsage):
        return llms.Usage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            input_tokens_cost_nanodollars=usage.cost.get(
                "input_tokens_cost_nanodollars"
            ),
            output_tokens_cost_nanodollars=usage.cost.get(
                "output_tokens_cost_nanodollars"
            ),
            total_backend_latency_ms=usage.total_backend_latency_ms,
        )
    return llms.Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )


class OpenAIResponsesAPI(llms.LLMChat):
    """An actor that interacts with an OpenAI-compatible API."""

    def __init__(self, client: openai.OpenAI, model: str, **kwargs):
        kwargs.setdefault("name", model)
        if model in ["gpt-3.5-turbo"]:
            kwargs["support_structured_outputs"] = False
            kwargs["support_vision"] = False
            kwargs["support_temperature"] = False
        super().__init__(**kwargs)
        self.model = model
        self.client = client
        self.serializer = openai_serializer.OpenAIResponsesSerializer()

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
        raw_messages = list(self.serializer.dump_messages(messages))

        api_kwargs = {"tools": tools, "response_format": schema}
        if self.support_temperature:
            api_kwargs["temperature"] = temperature
        # if seed:
        #     api_kwargs["seed"] = seed
        if system:
            api_kwargs["instructions"] = system

        return self._call_api(raw_messages, **api_kwargs)

    def dump_tools(self, tools: list[Any]) -> list[dict]:
        """Converts a list of functions to the OpenAI tool specification."""
        from kaggle_benchmarks import tools as tool_utils

        return [tool_utils.functions.function_to_openai_tool(tool) for tool in tools]

    def _call_api(
        self,
        messages: list,
        tools: list[Any] | None = None,
        response_format: Any = str,
        **kwargs,
    ) -> llms.LLMMessage[str]:
        """Makes the API call to the OpenAI-compatible endpoint."""

        if tools:
            kwargs["tools"] = self.dump_tools(tools)

        try:
            if response_format is str:
                response = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    **kwargs,
                )
            else:
                response_format.__name__ = "Response"
                response = self.client.responses.parse(
                    model=self.model,
                    input=messages,
                    text_format=response_format,
                    **kwargs,
                )
        except openai.BadRequestError as e:
            # logging.warning(
            #     f"encounter {e}. Trying out disabling structured output."
            # )
            raise llms.APIError(
                f"{self!r} encountered an API invocation error. "
                f"input: {messages!r}"
                f"arguments: {kwargs!r}"
                f"error: {e}"
            )

        return self.process_response(response, tools=tools)

    def process_response(
        self, response, message: llms.LLMMessage | None = None, tools=()
    ) -> llms.LLMMessage:
        """Processes the API response to extract content and tool calls."""
        from kaggle_benchmarks import tools as tool_utils

        tool_calls = []
        content = ""
        for item in response.output:
            if item.type == "function_call":
                tool_calls.append(
                    tool_utils.invoke_tool(
                        tool_utils.ToolInvocation(
                            name=item.name,
                            call_id=item.call_id,
                            arguments=json.loads(item.arguments),
                        ),
                        tools,
                    )
                )
            elif item.type == "message":
                content += "".join(x.text for x in item.content)

        if message is None:
            return llms.LLMMessage(
                sender=self,
                content=content,
                tool_calls=tool_calls,
                usage=parse_usage(response.usage),
            )

        message.content = content
        message.tool_calls = tool_calls
        message.usage = parse_usage(response.usage)
        return message


class StreamingOpenAIResponsesAPI(OpenAIResponsesAPI):
    """An actor that handles streaming responses."""

    def _call_api(
        self,
        messages: list[dict[str, str]],
        tools: list[Any] | None = None,
        response_format: Any = str,
        **kwargs,
    ) -> llms.LLMMessage:
        """Makes a streaming API call."""
        from kaggle_benchmarks import tools as tool_utils

        tools_definition = self.dump_tools(tools) if tools else []
        result = llms.LLMMessage(sender=self, content="", tool_calls=[], usage=None)

        stream_kwargs = {
            "model": self.model,
            "input": messages,
            "tools": tools_definition,
            **kwargs,
        }
        if response_format and response_format is not str:
            stream_kwargs["text_format"] = response_format

        with self.client.responses.stream(**stream_kwargs) as response:
            for chunk in response:
                if hasattr(chunk, "usage") and chunk.usage:
                    result.usage = parse_usage(chunk.usage)
                if isinstance(chunk, responses_types.ResponseTextDeltaEvent):
                    result.add_chunk(chunk.delta)
                elif isinstance(
                    chunk, responses_types.ResponseFunctionCallArgumentsDeltaEvent
                ):
                    result.add_chunk(chunk.delta)
                elif isinstance(chunk, responses_types.ResponseOutputItemDoneEvent):
                    if isinstance(chunk.item, responses_types.ResponseFunctionToolCall):
                        result.tool_calls.append(
                            tool_utils.ToolInvocation(
                                name=chunk.item.name,
                                call_id=chunk.item.call_id,
                                arguments=json.loads(chunk.item.arguments),
                            )
                        )
                elif isinstance(chunk, responses_types.ResponseCompletedEvent):
                    return self.process_response(chunk.response, result)
        return result


class ModelProxyOpenAI(OpenAIResponsesAPI):
    """An OpenAI-compatible actor for routing requests through a model proxy.

    This class includes workarounds for inconsistencies observed with various
    proxied models (e.g., Gemini, Meta, Gemma, DeepSeek).
    """

    def __init__(self, client: openai.OpenAI, model: str, **kwargs):
        if "gemini" in model:
            kwargs["support_structured_outputs"] = True
        elif "meta" in model:
            kwargs["support_structured_outputs"] = False
            kwargs["support_tool_calling"] = False
        elif "gemma" in model:
            kwargs["support_structured_outputs"] = False
            kwargs["support_vision"] = False
            self.roles_mapping = {
                "system": "user",
            }
        elif "deepseek" in model:
            kwargs["support_vision"] = False
            kwargs["support_structured_outputs"] = True
            kwargs["postprocessor"] = utils.extract_thinking_tag
        elif "qwen" in model:
            kwargs["support_vision"] = False
            # kwargs["support_structured_outputs"] = True
        elif "anthropic" in model:
            kwargs["postprocessor"] = utils.extract_json_tag

        kwargs.setdefault("support_tool_calling", False)
        super().__init__(client, model, **kwargs)
        self.serializer = openai_serializer.ModelProxyOpenAISerializer(
            roles_mapping=self.roles_mapping,
        )

    def _invoke(
        self,
        messages: list[messages.Message],
        *,
        schema: type[T] | type[str] = str,
        system: str | None = None,
        temperature: float | None = 0,
        seed: int | None = None,
        tools: list[Any] | None = None,
    ) -> llms.LLMMessage[str]:
        """Invokes the model, with a fallback for complex nested schemas.

        The model proxy can struggle with deeply nested Pydantic models. This
        method detects that and falls back to providing the schema via a system
        prompt instead of using the native structured output feature.
        """
        if issubclass(schema, pydantic.BaseModel) and has_nested_models(schema):
            return self._simulate_structured_response(
                messages=messages,
                schema_instructions=json.dumps(schema.model_json_schema()),
                temperature=temperature,
                seed=seed,
                tools=tools,
            )
        return super()._invoke(
            messages,
            schema=schema,
            system=system,
            temperature=temperature,
            seed=seed,
            tools=tools,
        )

    def _call_api(
        self,
        messages: list[dict[str, str]],
        tools: list[Any] | None = None,
        response_format: Any = str,
        **kwargs,
    ) -> llms.LLMMessage:
        """Calls the proxy using the `chat.completions` endpoint.

        The proxy does not handle the `responses` API correctly, so this method
        uses the deprecated `chat.completions` endpoint instead.
        """
        if self.support_structured_outputs and response_format is not str:
            method = self.client.chat.completions.parse
            kwargs["response_format"] = response_format
        else:
            method = self.client.chat.completions.create

        try:
            response = method(
                model=self.model,
                messages=messages,
                tools=tools or [],
                **kwargs,
            )
        except TypeError as e:
            # This can happen due to API quota or other proxy-side issues.
            raise RuntimeError(
                "API call failed, possibly due to an exhausted quota."
            ) from e

        message = response.choices[0].message
        return llms.LLMMessage(
            sender=self,
            content=message.content or "",
            usage=parse_usage(response.usage),
        )


def has_nested_models(model: type[pydantic.BaseModel]) -> bool:
    """Checks if a Pydantic model's schema contains nested definitions."""
    schema = model.model_json_schema()
    return bool(schema.get("$defs"))
