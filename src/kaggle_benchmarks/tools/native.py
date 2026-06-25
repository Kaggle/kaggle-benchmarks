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
    conversation. The loop runs in two phases:

    1. **Tool rounds** — calls ``llm.respond(tools=…)`` without ``schema=``
       to avoid backend conflicts (OpenAI requires ``strict=True`` on tools
       when ``response_format`` is set; GenAI models return tool_calls instead
       of schema-formatted content). Repeats until the model stops requesting
       tools or ``max_tool_rounds`` is exhausted.

    2. **Schema formatting** — if ``schema`` is not ``str``, makes one final
       ``llm.respond(schema=…)`` call (no tools) so the response conforms to
       the requested output type.

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
    from kaggle_benchmarks import actors, chats

    response = None
    exhausted = True

    with chats.fork(name="Tool loop"):
        for _ in range(max_tool_rounds):
            response = llm.respond(tools=tools, **respond_kwargs)

            # tool_calls is a list[ToolInvocation] post-normalization:
            #  - plain Message responses: normalized in respond()'s LLMResponse
            #    branch (Step 2) or in stream()'s finalize pass (Step 3)
            #  - LLMMessage responses (e.g., MockedChat): typed field set
            #    directly by the producer
            tool_calls = response.tool_calls
            if not tool_calls:
                exhausted = False
                break

            for invocation in tool_calls:
                result = invoke_tool(invocation, tools)
                actors.Tool(name=invocation.name).send(result)

        if not exhausted and schema is not str:
            # User message required: some models (e.g. Claude) reject requests
            # where the conversation ends with an assistant message.
            actors.user.send(
                "Now format your previous answer using the requested schema."
            )
            response = llm.respond(schema=schema, **respond_kwargs)

    # Raised outside `with chats.fork()` because the context manager may
    # swallow exceptions when no parent run is active (see contexts.enter).
    if exhausted:
        raise ToolInvocationLimitExhausted(
            f"Tool invocation limit of {max_tool_rounds} rounds exhausted"
        )

    return response
