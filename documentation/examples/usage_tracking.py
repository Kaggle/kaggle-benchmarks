# Copyright 2025 Kaggle Inc.
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
from kaggle_benchmarks import chats, llm, task


@task(name="Usage tracking")
def usage_tracking(llm, question: str) -> dict:
    """Demonstrates accessing token usage, costs, and latency metrics."""
    with chats.new("Conversation") as chat:
        answer = llm.prompt(question)

        # Access usage from individual messages
        for msg in chat.messages:
            if msg.sender.role == "assistant":
                print(f"Input tokens: {msg.input_tokens}")
                print(f"Output tokens: {msg.output_tokens}")
                print(f"Input cost (nanodollars): {msg.input_tokens_cost_nanodollars}")
                print(
                    f"Output cost (nanodollars): {msg.output_tokens_cost_nanodollars}"
                )
                print(f"Backend latency (ms): {msg.total_backend_latency_ms}")

        # Access aggregated usage from chat
        return {
            "answer": answer,
            "total_input_tokens": chat.total_input_tokens,
            "total_output_tokens": chat.total_output_tokens,
            "total_input_cost_nanodollars": chat.total_input_tokens_cost_nanodollars,
            "total_output_cost_nanodollars": chat.total_output_tokens_cost_nanodollars,
            "total_latency_ms": chat.total_backend_latency_ms,
        }


result = usage_tracking.run(llm, "What is machine learning?")
result

# %%
