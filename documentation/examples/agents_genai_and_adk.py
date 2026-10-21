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
# title: "Agents: Gemini and ADK"
# ---
# `LLMAgent` is a second way to write a benchmark, alongside `LLMChat`. An agent
# *owns* what it works with — instructions, tools, and other agents it can hand
# work to — rather than being given them one call at a time.
#
# This example builds the same two-agent setup twice, once on each backend:
#
# * `GenAIAgent` calls the Gemini API directly, so it needs a `GOOGLE_API_KEY`
#   and no Kaggle credentials.
# * `ADKAgent` runs the same agent on Google ADK, which drives its own tool
#   loop. It needs `pip install google-adk`.
#
# Both produce the same kind of transcript, which is the point: the backend is
# an implementation detail of `invoke()`.

# %%
import kaggle_benchmarks as kbench
from kaggle_benchmarks.agents import ADKAgent, GenAIAgent

# %% [markdown]
# ## Tools are plain functions
#
# The docstring and type hints are what the model sees, so they are part of the
# interface rather than decoration.

# %%
POPULATIONS = {"Tokyo": 14_000_000, "Zurich": 430_000, "Lisbon": 545_000}


def lookup_population(city: str) -> int:
    """Returns the population of a city."""
    return POPULATIONS.get(city, 0)


def convert_currency(amount: float, rate: float) -> float:
    """Converts an amount using the given exchange rate."""
    return amount * rate


# %% [markdown]
# ## A subagent is an actor, not a function
#
# `researcher` below could be a tool — it answers a question and returns text.
# The difference is that it keeps a conversation: ask it twice and the second
# question lands in the same session as the first, so it still knows what was
# already discussed. Its `description` is what the parent's model reads when
# deciding whether to delegate, so it should say what the agent is *for*.

# %%
researcher = GenAIAgent(
    "gemini-2.5-flash",
    name="researcher",
    description="Looks up facts about cities, including population.",
    instructions="Answer with the fact and nothing else.",
    tools=[lookup_population],
)

planner = GenAIAgent(
    "gemini-2.5-flash",
    name="planner",
    instructions="Plan trips. Delegate factual lookups to the researcher.",
    tools=[convert_currency],
    subagents=[researcher],
)

# The model is offered both: its own tool, and the subagent as one more call.
[c.__name__ for c in planner.callables]

# %% [markdown]
# ## Running it
#
# `run()` answers in the agent's own session. Anything the researcher is asked
# becomes a nested session inside the planner's, so the transcript shows the
# delegation as a branch rather than as flattened text.

# %%
answer = planner.run("Is Tokyo or Lisbon bigger, and by how much?")
print(answer)

# %%
# The planner's session, with the researcher's turns nested inside it.
planner._session

# %% [markdown]
# Ask the researcher again and it still has the earlier exchange in view —
# this is the part a plain tool could not do.

# %%
researcher.run("And how does that compare to Zurich?")
researcher._session

# %% [markdown]
# ## The same agent on ADK
#
# ADK runs the tool loop itself, so `ADKAgent` overrides `run()` and translates
# the events ADK emits back into the session. The definition is otherwise
# identical — same tools, same subagent, same `run()`.

# %%
adk_researcher = ADKAgent(
    "gemini-2.5-flash",
    name="researcher",
    description="Looks up facts about cities, including population.",
    instructions="Answer with the fact and nothing else.",
    tools=[lookup_population],
)

adk_planner = ADKAgent(
    "gemini-2.5-flash",
    name="planner",
    instructions="Plan trips. Delegate factual lookups to the researcher.",
    tools=[convert_currency],
    subagents=[adk_researcher],
)

adk_planner.run("Is Tokyo or Lisbon bigger, and by how much?")

# %% [markdown]
# ## Using one in a benchmark
#
# An agent is an `Actor`, so a task takes it exactly as it would take an
# `LLMChat` — `task.run(agent)` rather than `task.run(llm)`.


# %%
@kbench.task(name="Bigger city")
def bigger_city(agent):
    answer = agent.run("Is Tokyo or Lisbon bigger? Reply with just the city name.")
    kbench.assertions.assert_in("tokyo", answer.lower())


bigger_city.run(planner)
