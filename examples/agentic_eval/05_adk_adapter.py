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
# # ADK adapter shape
#
# Design doc §9. Wrapping an external agent (Google ADK) as an eval Agent = mapping
# its *event stream* into a `Trajectory`. Real ADK is async and network-backed, so
# here we replay a **fake** ADK-shaped event stream (mirroring `dev/base_adk.py`:
# `event.author`, `event.content.parts[*]` with text / function_call /
# function_response, `event.is_final_response()`), then convert it with the
# `Trajectory.from_steps` alternative constructor.

# %%
import dataclasses

from kaggle_benchmarks.agentic import Trajectory

# %% [markdown]
# ## Fake ADK event objects (shape mirrors `google.adk` types)


# %%
@dataclasses.dataclass
class Part:
    text: str | None = None
    function_call: dict | None = None  # {"name":..., "args":{...}}
    function_response: dict | None = None  # {"name":..., "response":...}
    thought: bool = False


@dataclasses.dataclass
class Content:
    parts: list[Part]


@dataclasses.dataclass
class Event:
    author: str
    content: Content
    final: bool = False

    def is_final_response(self) -> bool:
        return self.final


def fake_adk_run(query: str):
    """Stand-in for ``runner.run_async(...)`` yielding ADK Events."""
    yield Event(
        "agent", Content([Part(text="I'll check events for that date.", thought=True)])
    )
    yield Event(
        "agent",
        Content(
            [Part(function_call={"name": "get_events", "args": {"month": "2025-10"}})]
        ),
    )
    yield Event(
        "tool",
        Content(
            [
                Part(
                    function_response={
                        "name": "get_events",
                        "response": [{"name": "El Clásico", "date": "2025-10-18"}],
                    }
                )
            ]
        ),
    )
    yield Event(
        "agent",
        Content(
            [
                Part(
                    text="Oct 18 has El Clásico — expect price/crowd spikes; flagging it."
                )
            ]
        ),
        final=True,
    )


# %% [markdown]
# ## The adapter: ADK Events → library `Trajectory`
#
# We translate the event stream into `from_steps` tuples, so the resulting
# trajectory is built from the library's own Message / LLMMessage /
# ToolInvocation types.


# %%
def adk_to_trajectory(events) -> Trajectory:
    steps: list[tuple] = []
    for event in events:
        for part in event.content.parts:
            if part.function_call:
                steps.append(
                    (
                        "call",
                        part.function_call["name"],
                        part.function_call.get("args", {}),
                    )
                )
            elif part.function_response:
                steps.append(
                    (
                        "result",
                        part.function_response["name"],
                        part.function_response["response"],
                    )
                )
            elif part.text and part.thought:
                steps.append(("reason", part.text))
            elif part.text and event.is_final_response():
                steps.append(("say", part.text))
            elif part.text:
                steps.append(("reason", part.text))
    return Trajectory.from_steps(steps, agent="adk-agent")


traj = adk_to_trajectory(fake_adk_run("Best October weekend for Barcelona?"))
print("tool calls:", [tc.name for tc in traj.tool_calls()])
traj

# %% [markdown]
# The real adapter drives `runner.run_async(...)` on an installed `google.adk`
# agent (optional extra). See design §9 / `dev/base_adk.py` for the real
# Runner/Session/Event wiring.

# %%
