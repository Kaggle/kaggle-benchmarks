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

"""Native tool calling agent for multi-turn function calling.

Provides a tool invocation loop for models that support native function
calling (e.g., Gemini, GPT, Claude).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar

from kaggle_benchmarks.tools.base import (
    ToolInvocation,
    ToolInvocationLimitExhausted,
    invoke_tool,
)

if TYPE_CHECKING:
    from kaggle_benchmarks.actors.llms import LLMChat
    from kaggle_benchmarks.messages import Message

T = TypeVar("T")


def native_tool_agent(
    llm: LLMChat,
    tools: list[Callable],
    schema: type[T] = str,
    max_tool_rounds: int = 10,
    **respond_kwargs: Any,
) -> Message:
    """Runs a multi-turn tool calling loop.

    Forks the chat to isolate tool-calling round-trips from the main
    conversation, then loops: call ``llm.respond()`` → check for tool_calls
    → invoke tools → send results → repeat until the LLM responds without
    tool calls or ``max_tool_rounds`` is exhausted.

    Args:
        llm: The LLM chat actor to use.
        tools: List of Python callables available as tools.
        schema: The output schema for the final response.
        max_tool_rounds: Maximum number of tool-calling rounds before raising
            ``ToolInvocationLimitExhausted``.
        **respond_kwargs: Additional keyword arguments forwarded to
            ``llm.respond()`` (e.g. ``seed``, ``temperature``, ``reasoning``).

    Returns:
        The final response message from the LLM (a ``messages.Message``).

    Raises:
        ToolInvocationLimitExhausted: If the LLM keeps requesting tool calls
            beyond ``max_tool_rounds`` iterations.
    """
    # Lazy imports to avoid circular dependencies (actors → tools → actors).
    from kaggle_benchmarks import actors, chats

    with chats.fork(name="Tool loop"):
        for _ in range(max_tool_rounds):
            # TODO: Pass schema= only on a final call without tools, not on
            # every round. Use a two-phase loop: tools-only rounds, then a
            # schema-only call once the model stops requesting tool calls.
            response = llm.respond(schema=schema, tools=tools, **respond_kwargs)

            # TODO: Use response.tool_calls once respond() returns LLMMessage.
            tool_calls = response._meta.get("tool_calls")
            if not tool_calls:
                return response

            for call_data in tool_calls:
                invocation = ToolInvocation.from_api_dict(call_data)
                result = invoke_tool(invocation, tools)
                actors.Tool(name=invocation.name).send(result)

    raise ToolInvocationLimitExhausted(
        f"Tool invocation limit of {max_tool_rounds} rounds exhausted"
    )
