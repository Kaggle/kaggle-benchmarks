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
from kaggle_benchmarks import actors, chats, llm, task
from kaggle_benchmarks.assertions import assert_true


@task(name="Usage tracking")
def usage_tracking(llm, question: str):
    """Demonstrates accessing token usage, costs, and latency metrics."""
    with chats.new("Conversation") as chat:
        answer = llm.prompt(question)

        # Access usage from individual messages
        for msg in chat.messages:
            if msg.usage.input_tokens is not None:
                print(msg.usage)

        # Assert total cost is within a reasonable range ($0 to $100)
        total_cost = chat.usage.total_cost_nanodollars or 0
        assert_true(
            0 <= total_cost <= 100 * 1e9,
            "Cost is between $0 and $100",
        )

        # Access aggregated usage from chat
        actors.system.send(f"Answer: {answer}\n\n{chat.usage}")


result = usage_tracking.run(llm, "What is machine learning?")
result

# %%
