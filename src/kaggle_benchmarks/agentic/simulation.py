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

"""Simulation runner (design doc §4.2).

A (mocked) user simulator + env-aware **emulated tools** (with result caching) +
the agent under test, recorded as a ``Trajectory`` of library types.

Real-library mapping: the user simulator is a ``ChatRoom`` ``Participant`` built
from the persona; tool emulators are callables over ``scenario.environment``
(caching via hishel/joblib); the tool loop is ``tools.native.native_tool_agent``.
The blocker for the fully-real version is tools inside ``ChatRoom``
(``Participant.reply(tools=...)`` currently raises ``NotImplementedError``).
"""

from __future__ import annotations

import json
from typing import Any, Callable

import pydantic

from kaggle_benchmarks import actors
from kaggle_benchmarks.agentic.agent import Call, PlannedAgent, Reason, Say
from kaggle_benchmarks.agentic.scenario import Persona, Scenario
from kaggle_benchmarks.agentic.trajectory import Trajectory
from kaggle_benchmarks.llm_messages import LLMMessage
from kaggle_benchmarks.messages import Message
from kaggle_benchmarks.tools.base import ToolInvocation, ToolInvocationResult


class EmulatedTool(pydantic.BaseModel):
    """A fake tool serving consistent results from a scenario's environment.

    Results are cached by (name, args) so popular calls are stable and cheap.
    On failure it returns the ``Exception`` as its output (see
    ``analyzers.no_tool_errors``).
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    name: str
    fn: Callable[..., Any]  # fn(env, **args) -> result
    env: dict[str, Any] = pydantic.Field(default_factory=dict)
    cache: dict[str, Any] = pydantic.Field(default_factory=dict)

    def __call__(self, **args: Any) -> Any:
        key = json.dumps(args, sort_keys=True, default=str)
        if key not in self.cache:
            try:
                self.cache[key] = self.fn(self.env, **args)
            except Exception as e:  # surfaced as a tool error
                self.cache[key] = e
        return self.cache[key]


def emulate(
    name: str,
    fn: Callable[..., Any],
    env: dict[str, Any],
    cache: dict[str, Any] | None = None,
) -> EmulatedTool:
    return EmulatedTool(
        name=name, fn=fn, env=env, cache=cache if cache is not None else {}
    )


class UserSimulator(pydantic.BaseModel):
    """A fake user that knows its persona/goal. Real: a ChatRoom participant."""

    # Persona is an Actor (plain class), so allow it as an arbitrary type.
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    persona: Persona
    opening: str

    @classmethod
    def from_persona(
        cls, persona: Persona, opening: str | None = None
    ) -> "UserSimulator":
        return cls(persona=persona, opening=opening or persona.goal)


def simulate(
    scenario: Scenario,
    agent: PlannedAgent,
    tools: dict[str, EmulatedTool],
    user: UserSimulator | None = None,
    max_tool_calls: int = 20,
) -> Trajectory:
    """Run one scenario and return the agent's trajectory (library-typed steps).

    The agent is a mocked ``PlannedAgent``; the runner does the real work of
    driving the emulated tools, caching, and recording the trajectory.
    """
    user = user or UserSimulator.from_persona(scenario.persona)
    speaker = actors.Actor(name=agent.name, role="assistant", avatar="🤖")
    traj = Trajectory(scenario_id=scenario.id)

    # The persona is an Actor, so it speaks for itself (its own name + avatar).
    traj.add(Message(content=user.opening, sender=user.persona))

    calls = 0
    for action in agent.plan:
        if isinstance(action, Reason):
            traj.add(
                LLMMessage(content="", sender=speaker, reasoning_traces=action.text)
            )
        elif isinstance(action, Call):
            args = dict(action.args)
            traj.add(ToolInvocation(name=action.name, arguments=args))
            calls += 1
            tool = tools.get(action.name)
            if tool is None:
                out: Any = RuntimeError(f"no tool named {action.name!r}")
            elif calls > max_tool_calls:
                out = RuntimeError("tool budget exhausted")
            else:
                out = tool(**args)
            traj.add(ToolInvocationResult(name=action.name, arguments=args, output=out))
        elif isinstance(action, Say):
            traj.add(LLMMessage(content=action.text, sender=speaker))
            traj.answer = action.text

    return traj
