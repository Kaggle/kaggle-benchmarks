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
from typing import Any, Callable, Generic, TypeVar

import pydantic

T = TypeVar("T")
F = TypeVar("F")
Name = TypeVar("Name", bound=str)


@dataclasses.dataclass
class ToolInvocation:
    """Represents a tool invocation requested by the LLM."""

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None

    # Allows serialization as content message
    def get_payload(self):
        return dataclasses.asdict(self)


UNKNOWN = object()


@dataclasses.dataclass
class ToolInvocationResult:
    """Represents the result of a tool invocation."""

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None
    output: Any = UNKNOWN
    error: str | None = None

    def describe(self):
        if self.error:
            return f"{self.name}({self.arguments}): Error: {self.error}"
        return f"{self.name}({self.arguments}) -> {self.output}"

    def get_payload(self):
        return dataclasses.asdict(self)

    @property
    def text(self):
        return str(self.output if self.output is not UNKNOWN else self.error)


class ToolCallModel(pydantic.BaseModel, Generic[Name, T]):
    # Represents a tool call in a structured response.
    # Generic Name allows for overwriting str with Literal['tool_name']
    name: Name
    arguments: T


class ModelResponse(pydantic.BaseModel, Generic[T, F]):
    # A structured response from the LLM that may contain tool calls or a message.

    tools: list[F] | None = None
    message: T | None = None

    model_config = pydantic.ConfigDict(
        title="Response",
        extra="forbid",
        arbitrary_types_allowed=False,
    )


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
        error_message = f"Error: Tool '{call.name}' not found."
        return ToolInvocationResult(
            name=call.name,
            arguments=call.arguments,
            output=error_message,
            call_id=call.call_id,
        )
    try:
        output = tool(**(call.arguments or {}))
        return ToolInvocationResult(
            name=call.name,
            arguments=call.arguments,
            output=output,
            call_id=call.call_id,
        )
    except KeyboardInterrupt:
        raise
    except Exception as e:
        error_message = f"Error invoking tool '{call.name}': {e}"
        return ToolInvocationResult(
            name=call.name,
            arguments=call.arguments,
            error=error_message,
            call_id=call.call_id,
        )


def iter_invocations(chat):
    from kaggle_benchmarks import llm_messages

    for item in chat.messages:
        if isinstance(item.content, ToolInvocationResult):
            yield item.content
        elif isinstance(item, llm_messages.LLMMessage):
            yield from item.tool_calls or []
