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

"""Trajectory — the record of what an agent did (design doc §3.2).

Built entirely from the library's own custom types: steps are ``Message`` /
``LLMMessage`` (utterances + reasoning + usage) and ``ToolInvocation`` /
``ToolInvocationResult`` (tool calls + results). Reasoning is an ``LLMMessage``
with empty content and ``reasoning_traces`` set.

``Trajectory`` is a pydantic model (``arbitrary_types_allowed`` since steps hold
the library dataclasses). Alternative constructors let you build one without a
live model: ``from_chat`` (adopt an existing ``Chat``) and ``from_steps``
(compact tuples, handy for tests/demos and for adapting a third-party harness).
"""

from __future__ import annotations

from typing import Any, Union

import pydantic

from kaggle_benchmarks import actors, chats
from kaggle_benchmarks.agentic._render import PanelRenderable
from kaggle_benchmarks.llm_messages import LLMMessage
from kaggle_benchmarks.messages import Message
from kaggle_benchmarks.tools.base import ToolInvocation, ToolInvocationResult
from kaggle_benchmarks.usage import Usage

Step = Union[Message, ToolInvocation, ToolInvocationResult]


def _is_reasoning(step: Any) -> bool:
    return (
        isinstance(step, Message)
        and bool(getattr(step, "reasoning_traces", None))
        and not str(step.content).strip()
    )


class Trajectory(pydantic.BaseModel, PanelRenderable):
    """Ordered steps + final answer. The unit of analysis."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    steps: list[Any] = pydantic.Field(default_factory=list)
    answer: Any = None
    scenario_id: str | None = None

    # --- construction ---
    def add(self, step: Step) -> Step:
        self.steps.append(step)
        return step

    @classmethod
    def from_chat(cls, chat: chats.Chat, answer: Any = None) -> "Trajectory":
        """Adopt an existing Chat transcript as a trajectory.

        Pulls out each message plus any ``tool_calls`` recorded on it (base
        ``Message`` keeps them in ``_meta``; ``LLMMessage`` as a field).
        """
        steps: list[Any] = []
        for msg in chat.history:
            if not isinstance(msg, Message):
                continue
            steps.append(msg)
            for call in msg.tool_calls or []:
                steps.append(call)
        return cls(steps=steps, answer=answer)

    @classmethod
    def from_steps(
        cls, steps: list[tuple[Any, ...]], *, agent: str = "agent"
    ) -> "Trajectory":
        """Build a trajectory from compact tuples (no model needed).

        Step forms::

            ("user", text)
            ("reason", text)
            ("call", name, args_dict)
            ("result", name, output)
            ("say", text)          # the final answer
        """
        speaker = actors.Actor(name=agent, role="assistant", avatar="🤖")
        traj = cls()
        for step in steps:
            kind = step[0]
            if kind == "user":
                traj.add(Message(content=step[1], sender=actors.user))
            elif kind == "reason":
                traj.add(
                    LLMMessage(content="", sender=speaker, reasoning_traces=step[1])
                )
            elif kind == "call":
                args = step[2] if len(step) > 2 else {}
                traj.add(ToolInvocation(name=step[1], arguments=args))
            elif kind == "result":
                out = step[2] if len(step) > 2 else None
                traj.add(ToolInvocationResult(name=step[1], arguments={}, output=out))
            elif kind in ("say", "assistant"):
                traj.add(LLMMessage(content=step[1], sender=speaker))
                traj.answer = step[1]
            else:
                raise ValueError(f"unknown step kind: {kind!r}")
        return traj

    # --- accessors used by analyzers (§3.3) ---
    def tool_calls(self) -> list[ToolInvocation]:
        return [s for s in self.steps if isinstance(s, ToolInvocation)]

    def tool_results(self) -> list[ToolInvocationResult]:
        return [s for s in self.steps if isinstance(s, ToolInvocationResult)]

    def called(self, name: str) -> bool:
        return any(tc.name == name for tc in self.tool_calls())

    def messages(self) -> list[Message]:
        return [s for s in self.steps if isinstance(s, Message)]

    def reasoning_text(self) -> str:
        return "\n".join(
            s.reasoning_traces
            for s in self.steps
            if _is_reasoning(s)  # type: ignore[misc]
        )

    def final_text(self) -> str:
        if self.answer is not None:
            return str(self.answer)
        for msg in reversed(self.messages()):
            if (
                msg.sender
                and msg.sender.role == "assistant"
                and str(msg.content).strip()
            ):
                return str(msg.content)
        return ""

    def full_text(self) -> str:
        parts: list[str] = []
        for s in self.steps:
            if _is_reasoning(s):
                parts.append(s.reasoning_traces)
            elif isinstance(s, Message):
                parts.append(str(s.content))
            elif isinstance(s, ToolInvocation):
                parts.append(f"{s.name}({s.arguments})")
            elif isinstance(s, ToolInvocationResult):
                parts.append(str(s.output))
        return "\n".join(p for p in parts if p)

    def usage(self) -> Usage:
        total = Usage()
        for s in self.steps:
            if isinstance(s, LLMMessage) and s.usage is not None:
                total = total + s.usage
        return total

    def to_chat(self, name: str = "trajectory") -> chats.Chat:
        """A Chat of just the message steps (for library run/eval interop)."""
        return chats.Chat(history=list(self.messages()), name=name)

    # --- visualization ---
    def render(self) -> str:
        lines: list[str] = []
        for s in self.steps:
            if _is_reasoning(s):
                lines.append(f"🧠 {s.reasoning_traces}")
            elif isinstance(s, ToolInvocation):
                lines.append(f"🔧 call {s.name}({s.arguments})")
            elif isinstance(s, ToolInvocationResult):
                lines.append(f"📤 {s.name} -> {s.output}")
            elif isinstance(s, Message):
                role = s.sender.role if s.sender else "?"
                lines.append(f"💬 [{role}] {s.content}")
        return "\n".join(lines)

    def __panel__(self):
        import panel as pn

        objs: list[Any] = [
            pn.pane.Markdown(f"### 🧭 Trajectory ({len(self.steps)} steps)")
        ]
        for s in self.steps:
            if _is_reasoning(s):
                objs.append(
                    pn.pane.Markdown(
                        f"🧠 *{s.reasoning_traces}*", styles={"color": "#666"}
                    )
                )
            elif isinstance(s, ToolInvocation):
                objs.append(pn.pane.Markdown(f"🔧 **{s.name}**(`{s.arguments}`)"))
            elif isinstance(s, ToolInvocationResult):
                objs.append(pn.pane.Markdown(f"📤 `{s.name}` → {s.output}"))
            elif isinstance(s, Message):
                objs.append(pn.panel(s))  # Message renders itself (rendering seam)
        u = self.usage()
        if u.input_tokens is not None or u.output_tokens is not None:
            objs.append(
                pn.pane.Markdown(
                    f"*tokens — in: {u.input_tokens}, out: {u.output_tokens}*",
                    styles={"color": "gray"},
                )
            )
        return pn.Column(*objs, sizing_mode="stretch_width")
