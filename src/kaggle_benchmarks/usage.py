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
from typing import Self


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

    def __str__(self) -> str:
        """Returns a human-readable string representation of the usage."""
        lines = [
            f"Input tokens: {self.input_tokens}",
            f"Output tokens: {self.output_tokens}",
            f"Input cost (nanodollars): {self.input_tokens_cost_nanodollars}",
            f"Output cost (nanodollars): {self.output_tokens_cost_nanodollars}",
            f"Total cost (nanodollars): {self.total_cost_nanodollars}",
            f"Backend latency (ms): {self.total_backend_latency_ms}",
        ]
        return "\n".join(lines)

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
