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

import dataclasses
import inspect
import json
from typing import Any, Callable, Generic, TypeVar

import pydantic

T = TypeVar("T")


class ToolInvocationLimitExhausted(Exception):
    """Raised when the tool invocation loop exceeds the maximum number of rounds."""


@dataclasses.dataclass
class ToolInvocation:
    """Represents a tool invocation requested by the LLM."""

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None

    @classmethod
    def from_api_dict(cls, call_data: dict) -> "ToolInvocation":
        """Creates a ToolInvocation from a normalised tool call dict.

        Both backends normalise their tool call responses to the same
        dict schema::

            {"id": ..., "function": {"name": ..., "arguments": ...}}

        The OpenAI backend produces this natively from the Chat
        Completions response, while the GenAI backend converts
        ``function_call`` Parts to this format in ``GoogleGenAI._call_api``.

        Handles edge cases from various backends:
        - ``arguments`` may be a JSON string (OpenAI) or a dict (GenAI).
        - ``arguments`` may be ``None`` for parameterless tools.
        """
        func = call_data["function"]
        arguments = func.get("arguments") or {}
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return cls(
            name=func["name"],
            arguments=arguments,
            call_id=call_data.get("id"),
        )


@dataclasses.dataclass
class ToolInvocationResult:
    """Represents the result of a tool invocation."""

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None
    output: Any = None
    error: str | None = None

    @property
    def text(self) -> str:
        """Returns a string representation of the result or error."""
        if self.error:
            return self.error
        return str(self.output)

    def describe(self):
        if self.error:
            return f"{self.name}({self.arguments}): Error: {self.error}"
        return f"{self.name}({self.arguments}) -> {self.output}"


class ToolCallModel(pydantic.BaseModel):
    """Represents a tool call in a structured response."""

    name: str
    arguments: dict[str, Any]


class ModelResponse(pydantic.BaseModel, Generic[T]):
    """A structured response from the LLM that may contain tool calls or a message."""

    tools: list[ToolCallModel] | None = None
    message: T | None = None


def describe_tools(tools: list[Callable]) -> str:
    """Generates a plain English description of the available tools."""
    descriptions = []
    for tool in tools:
        sig = inspect.signature(tool)
        params = []
        for param in sig.parameters.values():
            param_str = param.name
            if param.annotation != inspect.Parameter.empty:
                try:
                    param_str += f": {param.annotation.__name__}"
                except AttributeError:
                    param_str += f": {str(param.annotation)}"
            if param.default != inspect.Parameter.empty:
                param_str += f" = {param.default!r}"
            params.append(param_str)

        param_list_str = ", ".join(params)

        return_annotation = ""
        if sig.return_annotation != inspect.Parameter.empty:
            try:
                return_annotation = f" -> {sig.return_annotation.__name__}"
            except AttributeError:
                return_annotation = f" -> {str(sig.return_annotation)}"

        docstring = (tool.__doc__ or "").strip()
        description = (
            f"- `{tool.__name__}({param_list_str}){return_annotation}`: {docstring}"
        )
        descriptions.append(description)

    if not descriptions:
        return "No tools available."

    return "\n".join(descriptions)


def invoke_tool(call: ToolInvocation, tools: list[Callable]) -> ToolInvocationResult:
    """Invokes a tool and returns the result."""
    tool = next((t for t in tools if t.__name__ == call.name), None)
    if tool is None:
        return ToolInvocationResult(
            name=call.name,
            arguments=call.arguments,
            error=f"Error: Tool '{call.name}' not found.",
            call_id=call.call_id,
        )
    try:
        # Filter arguments to only those the function accepts.
        # Some backends (e.g. Model Proxy) inject extra fields like
        # 'signature' into tool call arguments that the actual function
        # does not expect.
        sig = inspect.signature(tool)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        args = call.arguments or {}
        if has_var_keyword:
            filtered_args = args
        else:
            accepted = set(sig.parameters.keys())
            filtered_args = {k: v for k, v in args.items() if k in accepted}

        output = tool(**filtered_args)
        return ToolInvocationResult(
            name=call.name,
            arguments=call.arguments,
            output=output,
            call_id=call.call_id,
        )
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return ToolInvocationResult(
            name=call.name,
            arguments=call.arguments,
            error=f"Error invoking tool '{call.name}': {e}",
            call_id=call.call_id,
        )
