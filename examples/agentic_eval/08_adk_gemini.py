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
# # ADK + Gemini — a real agent as an eval Agent
#
# Runs a **Google ADK** agent (backed by the **Gemini API**) through the eval
# `ADKAgent` adapter, which maps ADK's event stream into a `Trajectory`
# (design doc §9). Unlike the other notebooks this one makes **live** Gemini
# calls, so it's nondeterministic.
#
# **Requirements**
# - `pip install google-adk`
# - a Gemini API key in the environment: `GOOGLE_API_KEY=...`
#   (we set `GOOGLE_GENAI_USE_VERTEXAI=FALSE` below to use the API key, not Vertex)

# %%
import os

from kaggle_benchmarks import actors
from kaggle_benchmarks.agentic import answer_mentions, called_tool, run_analyzers
from kaggle_benchmarks.agentic.adk import ADKAgent
from kaggle_benchmarks.messages import Message

# Use the Gemini API (an API key), not Vertex. Read at call time, so setting it
# here — before agent.act() — is enough.
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")


# %% [markdown]
# ## Tools (plain Python functions — ADK builds tool schemas from the signatures)


# %%
def get_weather(city: str, date: str) -> str:
    """Get the weather forecast for a city on a given date (YYYY-MM-DD)."""
    forecast = {
        "2025-10-11": "sunny, 24°C",
        "2025-10-18": "sunny, 25°C",
        "2025-10-25": "rainy, 19°C",
    }
    return forecast.get(date, "no forecast available")


def get_local_events(city: str, month: str) -> list[str]:
    """List notable local events in a city for a month (YYYY-MM)."""
    if city.strip().lower() == "barcelona" and month == "2025-10":
        return [
            "El Clásico (FC Barcelona vs Real Madrid) on 2025-10-18 "
            "— expect big crowds and hotel price spikes"
        ]
    return []


# %% [markdown]
# ## Build the ADK agent and run it through the adapter

# %%
agent = ADKAgent(
    model="gemini-2.5-flash",
    tools=[get_weather, get_local_events],
    instruction=(
        "You are a travel assistant. Before recommending a weekend, check the "
        "weather AND local events for the candidate dates, and flag any trade-offs "
        "(e.g. an event that spikes prices) rather than silently avoiding them."
    ),
    name="travel_adk",
)

conversation = [
    Message(
        content=(
            "I want to visit Barcelona for a weekend in October 2025 — the 11th, "
            "18th, or 25th. Which do you recommend? Check the weather and any local "
            "events first."
        ),
        sender=actors.user,
    )
]

response = agent.act(conversation)
print(response.answer)

# %% [markdown]
# The trajectory captures the live tool calls + results + final answer.

# %%
response.trajectory

# %% [markdown]
# ## Analyze what the agent actually did

# %%
for r in run_analyzers(
    response.trajectory,
    [
        called_tool("get_weather"),
        called_tool("get_local_events"),
        answer_mentions("clásico", "clasico", "event"),
    ],
):
    mark = "✅" if r.passed else "❌"
    print(mark, (r.details or {}).get("analyzer", r.expectation))

# %%
