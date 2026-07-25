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
# # Rendering gallery (no agents, no LLMs)
#
# Build the `kaggle_benchmarks.agentic` objects **by hand** (or load them from
# JSON) and just render them. Nothing is executed — this is purely to see the
# `__panel__` visualizations. Run the cells in Jupyter/VS Code.

# %%
import pathlib
import tempfile

from kaggle_benchmarks import actors
from kaggle_benchmarks.agentic import Persona, Report, Scenario, Suite, Trajectory
from kaggle_benchmarks.assertions import AssertionResult
from kaggle_benchmarks.llm_messages import LLMMessage
from kaggle_benchmarks.messages import Message
from kaggle_benchmarks.tools.base import ToolInvocation, ToolInvocationResult

# %% [markdown]
# ## 1. A Trajectory from compact tuples
#
# `Trajectory.from_steps` is the quickest way to hand-author a trajectory.

# %%
Trajectory.from_steps(
    [
        ("user", "Which weekend in October should I visit Rome?"),
        ("reason", "I'll check the weather and any big local events."),
        ("call", "get_events", {"month": "2025-10"}),
        ("result", "get_events", [{"name": "Rome Marathon", "date": "2025-10-19"}]),
        (
            "say",
            "Pick Oct 11 — calmer. Oct 19 is the Rome Marathon (crowds + higher prices).",
        ),
    ],
    agent="demo-agent",
)

# %% [markdown]
# ## 2. The same Trajectory built from the library's own types
#
# Steps are just `Message` / `LLMMessage` / `ToolInvocation` /
# `ToolInvocationResult`. The persona is an `Actor`, so it speaks for itself.

# %%
traveler = Persona(
    profile="first-time traveler",
    goal="best October weekend in Rome",
    name="Traveler",
    avatar="🧳",
)
agent = actors.Actor(name="demo-agent", role="assistant", avatar="🤖")

traj = Trajectory(scenario_id="rome-001")
traj.add(
    Message(content="Which weekend in October should I visit Rome?", sender=traveler)
)
traj.add(
    LLMMessage(
        content="", sender=agent, reasoning_traces="Check weather + local events."
    )
)
traj.add(ToolInvocation(name="get_events", arguments={"month": "2025-10"}))
traj.add(
    ToolInvocationResult(
        name="get_events",
        arguments={"month": "2025-10"},
        output=[{"name": "Rome Marathon", "date": "2025-10-19"}],
    )
)
traj.add(
    LLMMessage(
        content="Pick Oct 11 — calmer. Oct 19 is the Rome Marathon.", sender=agent
    )
)
traj.answer = "Pick Oct 11 — calmer. Oct 19 is the Rome Marathon."
traj

# %% [markdown]
# ## 3. A Scenario (the persona is an `Actor`)

# %%
scenario = Scenario(
    id="rome-001",
    persona=traveler,
    shared_context={"budget_usd": 1200, "month": "2025-10"},
    hidden_nuances=["The Rome Marathon on Oct 19 spikes hotel prices and crowds."],
    expected_behaviors=["checks local events", "flags the marathon weekend"],
    rubric={"must": ["surface the marathon"], "nice": ["explain the trade-off"]},
    tags=["planning", "hidden_constraint"],
    provenance={"author_model": "hand-written"},
)
scenario

# %% [markdown]
# ## 4. A Suite — built, then saved and loaded from JSON
#
# `Suite.save` / `Suite.load` round-trip through JSON, so you can hand-edit the
# file and reload to iterate on tasks (the persona serializes as a small dict).

# %%
suite = Suite(
    scenarios=[
        scenario,
        Scenario(
            id="lisbon-001",
            persona=Persona(
                profile="budget backpacker",
                goal="a cheap weekend in Lisbon",
                name="Backpacker",
                avatar="🎒",
            ),
            hidden_nuances=[
                "A tech conference books out the hostels the second weekend."
            ],
            tags=["planning", "hidden_constraint"],
        ),
    ],
    metadata={"problem": "trip-date planning", "author_models": ["hand-written"]},
)
suite

# %%
path = pathlib.Path(tempfile.mkdtemp()) / "sample.suite.json"
suite.save(str(path))
print("saved to", path, "— edit it and re-run Suite.load to iterate by hand")
Suite.load(str(path))

# %% [markdown]
# ## 5. A Report (each check is an `AssertionResult`)

# %%
results = [
    AssertionResult(
        passed=True,
        expectation="tool get_events is called",
        details={"analyzer": "called_tool(get_events)", "error_class": None},
    ),
    AssertionResult(
        passed=False,
        expectation="answer surfaces the marathon",
        details={
            "analyzer": "answer_mentions(marathon)",
            "error_class": "missed_hidden_constraint",
        },
    ),
]
Report(
    scenario_id="rome-001",
    agent="demo-agent",
    score=0.5,
    passed=False,
    results=results,
    error_classes=["missed_hidden_constraint"],
)

# %%
