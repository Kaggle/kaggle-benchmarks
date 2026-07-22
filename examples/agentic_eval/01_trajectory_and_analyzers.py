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
# # Trajectory + Analyzers
#
# The [`kaggle_benchmarks.agentic`](../../src/kaggle_benchmarks/agentic) prototype
# for design doc §3.2–3.3. We run a (mocked) agent through the travel scenario,
# get a `Trajectory` — built from the library's own `Message` / `LLMMessage` /
# `ToolInvocation` types — and run analyzers over it (which return the library's
# `AssertionResult`).

# %%
import panel as pn

from kaggle_benchmarks.agentic import (
    answer_mentions,
    called_tool,
    error_class_of,
    judge,
    reasoning_mentions,
    run_analyzers,
    simulate,
)
from kaggle_benchmarks.agentic.demo import (
    FOOTBALL_JUDGE,
    TRAVEL,
    thorough_agent,
    travel_tools,
)

pn.extension()

# %% [markdown]
# ## Run the agent and inspect the trajectory
#
# `simulate` drives the emulated tools and records every step. The `Trajectory`
# renders itself (`__panel__`) — display it as the last expression in the cell.

# %%
traj = simulate(TRAVEL, thorough_agent(), travel_tools(TRAVEL))
print(traj.render())
traj

# %% [markdown]
# ## Analyze the trajectory
#
# Structural (`called_tool`), reasoning (`reasoning_mentions`), answer, and
# judge-based checks. Each returns an `AssertionResult`.

# %%
analyzers = [
    called_tool("web_search"),
    called_tool("get_events"),
    reasoning_mentions("clásico", "football"),
    answer_mentions("clásico", "real madrid", "match"),
    judge("Did the agent flag the local event as a price/crowd risk?", FOOTBALL_JUDGE),
]
for r in run_analyzers(traj, analyzers):
    mark = "✅" if r.passed else "❌"
    name = (r.details or {}).get("analyzer", r.expectation)
    extra = "" if r.passed else f"  [{error_class_of(r)}]"
    print(f"{mark} {name}{extra}")

# %%
