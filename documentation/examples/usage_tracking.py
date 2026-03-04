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

# %% [markdown]
# ---
# title: Usage Tracking
# ---
# %%
from kaggle_benchmarks import assert_true, chats, llm, task


@task(name="Usage tracking")
def usage_tracking(llm, question: str) -> str:
    """Demonstrates accessing token usage, costs, and latency metrics."""
    with chats.new("Conversation") as chat:
        answer = llm.prompt(question)

        # Access usage from individual messages
        for msg in chat.messages:
            print(f"Input tokens: {msg.usage.input_tokens}")
            print(f"Output tokens: {msg.usage.output_tokens}")
            print(
                f"Input cost (nanodollars): {msg.usage.input_tokens_cost_nanodollars}"
            )
            print(
                f"Output cost (nanodollars): {msg.usage.output_tokens_cost_nanodollars}"
            )
            print(f"Backend latency (ms): {msg.usage.total_backend_latency_ms}")

        # Assert total cost is within a reasonable range ($0 to $100)
        assert_true(
            0 <= chat.usage.total_cost_nanodollars <= 100 * 1e9,
            "Cost is between $0 and $100",
        )

        # Access aggregated usage from chat
        return (
            f"Answer: {answer}\n"
            f"Total input tokens: {chat.usage.input_tokens}\n"
            f"Total output tokens: {chat.usage.output_tokens}\n"
            f"Total input cost (nanodollars): {chat.usage.input_tokens_cost_nanodollars}\n"
            f"Total output cost (nanodollars): {chat.usage.output_tokens_cost_nanodollars}\n"
            f"Total latency (ms): {chat.usage.total_backend_latency_ms}"
        )


result = usage_tracking.run(llm, "What is machine learning?")
result

# %%
