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
from typing import Iterable, Self, TypeVar

from kaggle_benchmarks import chats, messages, utils
from kaggle_benchmarks import tools as tool_utils

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class Usage:
    """Represents the token usage, cost, and latency of an LLM invocation."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    input_tokens_cost_nanodollars: int | None = None
    output_tokens_cost_nanodollars: int | None = None
    total_backend_latency_ms: int | None = None

    @property
    def total_cost_nanodollars(self) -> int | None:
        """Total cost in nanodollars (input + output)."""
        if (
            self.input_tokens_cost_nanodollars is None
            and self.output_tokens_cost_nanodollars is None
        ):
            return None
        return (self.input_tokens_cost_nanodollars or 0) + (
            self.output_tokens_cost_nanodollars or 0
        )

    def __add__(self, other: Self | None) -> Self:
        if other is None:
            return self

        def add_optional(a: int | None, b: int | None) -> int | None:
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        return Usage(
            input_tokens=add_optional(self.input_tokens, other.input_tokens),
            output_tokens=add_optional(self.output_tokens, other.output_tokens),
            input_tokens_cost_nanodollars=add_optional(
                self.input_tokens_cost_nanodollars,
                other.input_tokens_cost_nanodollars,
            ),
            output_tokens_cost_nanodollars=add_optional(
                self.output_tokens_cost_nanodollars,
                other.output_tokens_cost_nanodollars,
            ),
            total_backend_latency_ms=add_optional(
                self.total_backend_latency_ms, other.total_backend_latency_ms
            ),
        )


@dataclasses.dataclass
class LLMMessage(messages.Message[T]):
    """Represents a message from an LLM, including content, status, and metadata."""

    content: T
    _status: utils.Status = utils.Status.RUNNING
    thinking: str | None = None
    tool_calls: (
        list[tool_utils.ToolInvocation | tool_utils.ToolInvocationResult] | None
    ) = None
    usage: Usage | None = None
    # Used to keep the history that was sent to the LLM and may include response
    # format and tool invocation instructions.
    chat: chats.Chat | None = None

    def add_chunk(self, chunk: str):
        from kaggle_benchmarks import events

        self.content += chunk
        events.manager.dispatch("new_chunk", self, chunk)

    @classmethod
    def from_chunks(
        cls,
        chunks: Iterable[
            str | tool_utils.ToolInvocation | tool_utils.ToolInvocationResult | Usage
        ],
        **kwargs,
    ) -> Self:
        """Constructs an LLMMessage iteratively from a stream of chunks."""
        from kaggle_benchmarks import events

        tool_calls = []
        obj = cls(content="", tool_calls=tool_calls, **kwargs)
        events.manager.dispatch("new_message", chats.get_current_chat(), obj)
        events.manager.dispatch("start_streaming", obj)
        for chunk in chunks:
            if isinstance(chunk, str):
                obj.add_chunk(chunk)
            elif isinstance(chunk, tool_utils.ToolInvocation):
                tool_calls.append(chunk)
                events.manager.dispatch("new_tool_call", obj, chunk)
            elif isinstance(chunk, tool_utils.ToolInvocationResult):
                if chunk.call_id:
                    tool_calls[:] = [
                        tc
                        for tc in tool_calls
                        if not (
                            isinstance(tc, tool_utils.ToolInvocation)
                            and tc.call_id == chunk.call_id
                        )
                    ]
                tool_calls.append(chunk)
                events.manager.dispatch("new_tool_call", obj, chunk)
            elif isinstance(chunk, Usage):
                if obj.usage is None:
                    obj.usage = Usage()
                obj.usage += chunk
        obj.status = utils.Status.SUCCESS
        events.manager.dispatch("end_content", obj)
        return obj

    def __panel__(self):
        from kaggle_benchmarks import ui

        return ui.panel.render_llm_message(self)
