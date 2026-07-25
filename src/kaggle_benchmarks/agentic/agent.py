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

"""The eval-facing Agent — a black box under test (design doc §3.1).

* ``Agent`` protocol: ``act(conversation) -> Response`` for the single-shot case.
* ``LLMAgent``: the real adapter — wraps an ``LLMChat`` and runs the tool loop
  (``tools.native.native_tool_agent``), returning a ``Trajectory``.
* ``ConstantAgent``: a dummy.
* ``PlannedAgent``: a *scripted plan* the simulation runner executes against
  emulated tools (mocked policy; see ``agentic.simulation``).

Data types are pydantic models. NOTE naming: the library's ``actors.Actor`` is a
*speaker identity*; this Agent is a *policy that produces answers* (design doc M0
flags settling this).
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, Union, runtime_checkable

import pydantic

from kaggle_benchmarks.agentic.trajectory import Trajectory
from kaggle_benchmarks.messages import Message


class Response(pydantic.BaseModel):
    """What an agent's ``act()`` returns: the answer, the trajectory, and
    metadata (usage, model, timings, …). Design decision §3.1.

    Intended to also support *progressive* construction for streaming — a live
    UI can watch it fill in — but that lives in a follow-up.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    answer: Any = None
    trajectory: Trajectory
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)


@runtime_checkable
class Agent(Protocol):
    name: str

    def act(self, conversation: list[Message]) -> Response: ...


# --- scripted actions for the mocked PlannedAgent ---
class Reason(pydantic.BaseModel):
    text: str


class Call(pydantic.BaseModel):
    name: str
    args: dict[str, Any] = pydantic.Field(default_factory=dict)


class Say(pydantic.BaseModel):
    text: str


Action = Union[Reason, Call, Say]


class PlannedAgent(pydantic.BaseModel):
    """A mocked agent expressed as a fixed plan (not reactive to results)."""

    name: str
    plan: list[Action] = pydantic.Field(default_factory=list)


class ConstantAgent(pydantic.BaseModel):
    """A dummy black box: always answers the same, empty trajectory otherwise."""

    name: str
    answer: str

    def act(self, conversation: list[Message]) -> Response:
        traj = Trajectory.from_steps([("say", self.answer)], agent=self.name)
        return Response(answer=self.answer, trajectory=traj)


class LLMAgent:
    """Real adapter: an ``LLMChat`` (+ optional tools) as an eval Agent.

    Runs the library's native tool loop and adopts the resulting chat as the
    trajectory. This is the production path (needs a configured model); the
    demos use ``PlannedAgent`` + ``simulate`` for a deterministic, offline run.

    Not a pydantic model — it holds a live ``LLMChat`` and behaviour, not data.
    """

    def __init__(
        self, llm, tools: list[Callable] | None = None, name: str | None = None
    ):
        self.llm = llm
        self.tools = tools or []
        self.name = name or getattr(llm, "name", "llm-agent")

    def act(self, conversation: list[Message]) -> Response:
        from kaggle_benchmarks import chats
        from kaggle_benchmarks.tools.native import native_tool_agent

        with chats.new(self.name) as chat:
            for msg in conversation:
                chats.send(msg)
            result = native_tool_agent(self.llm, self.tools)
        return Response(
            answer=result.content,
            trajectory=Trajectory.from_chat(chat, answer=result.content),
        )


def as_agent(
    obj: Any, *, tools: list[Callable] | None = None, name: str | None = None
) -> Agent:
    """Adapt an existing object into an eval Agent (design decision §3.1).

    - already an ``Agent`` (has ``act``) → returned unchanged;
    - anything else (e.g. an ``LLMChat``) → wrapped in ``LLMAgent``.

    Reuses the library's existing actors rather than forking a new identity type.
    """
    if isinstance(obj, Agent):
        return obj
    return LLMAgent(obj, tools=tools, name=name)


def act_in_current_chat(agent: Agent) -> Response:
    """Run a (stateless) agent against the *current* chat's history.

    Convenience for interactive/notebook use; ``act()`` itself stays stateless
    (design decision §3.1).
    """
    from kaggle_benchmarks import chats

    return agent.act(chats.get_current_chat().messages)
