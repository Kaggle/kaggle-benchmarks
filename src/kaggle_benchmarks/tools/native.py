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

"""Native tool calling agent for models with API-level function calling support.

This module provides a tool invocation loop for models that support function
calling natively (e.g., Gemini, GPT, Claude).  It complements the simulated
tool calling in ``tools/simulate.py`` (PR #12), which uses structured output
to emulate tool calling for models that lack native support.
"""

from typing import Any, Callable, TypeVar

from kaggle_benchmarks.tools.base import (
    ToolInvocationLimitExhausted,
    invoke_tool,
    parse_tool_call,
)

T = TypeVar("T")

DEFAULT_MAX_TOOL_ROUNDS = 10


def native_agent(
    llm,
    tools: list[Callable],
    schema: type[T] = str,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    **respond_kwargs: Any,
):
    """Runs a native tool calling loop for models that support function calling.

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

    with chats.fork(name="Tool loop") as _subchat:
        for _ in range(max_tool_rounds):
            response = llm.respond(schema=schema, tools=tools, **respond_kwargs)

            tool_calls = response._meta.get("tool_calls")
            if not tool_calls:
                return response

            for call_data in tool_calls:
                invocation = parse_tool_call(call_data)
                result = invoke_tool(invocation, tools)
                actors.Tool(name=invocation.name).send(result)

    raise ToolInvocationLimitExhausted(
        f"Tool invocation limit of {max_tool_rounds} rounds exhausted"
    )
