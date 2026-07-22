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
# # Evaluate + leaderboard + error taxonomy
#
# Design doc §4.3. Grade both agents with the `Examiner` (analyzers + judge),
# then compare. Each check is a library `AssertionResult`; the lazy agent should
# fail with `missed_hidden_constraint`. `Report` renders itself (`__panel__`).

# %%
import panel as pn

from kaggle_benchmarks.agentic import (
    Examiner,
    answer_mentions,
    called_tool,
    judge,
    reasoning_mentions,
    simulate,
)
from kaggle_benchmarks.agentic.demo import (
    FOOTBALL_JUDGE,
    TRAVEL,
    lazy_agent,
    thorough_agent,
    travel_tools,
)

pn.extension()


# %% [markdown]
# The rubric-as-analyzers: a great agent discovers the event (a tool) AND
# surfaces it (answer / judge).


# %%
def analyzers_for_travel():
    return [
        called_tool("get_events"),
        called_tool("web_search"),
        reasoning_mentions("clásico", "football", "match"),
        answer_mentions("clásico", "real madrid", "match"),
        judge(
            "Did the agent flag the local event as a price/crowd risk?", FOOTBALL_JUDGE
        ),
    ]


# %% [markdown]
# ## Grade both agents

# %%
examiner = Examiner(author_models=["mock-pro"])
reports = []
for make_agent in (thorough_agent, lazy_agent):
    agent = make_agent()
    traj = simulate(TRAVEL, agent, travel_tools(TRAVEL))
    reports.append(examiner.grade(traj, TRAVEL, agent.name, analyzers_for_travel()))

# %% [markdown]
# ## Leaderboard

# %%
print(f"{'agent':16} {'score':>6}  {'passed':>6}  error_classes")
for r in sorted(reports, key=lambda r: r.score, reverse=True):
    errs = ", ".join(r.error_classes) or "-"
    print(f"{r.agent:16} {r.score:6.0%}  {str(r.passed):>6}  {errs}")

# %% [markdown]
# ## Per-agent report (renders each `AssertionResult`)

# %%
pn.Column(*[pn.panel(r) for r in reports])

# %%
