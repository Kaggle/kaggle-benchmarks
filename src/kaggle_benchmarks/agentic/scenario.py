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

"""Scenario + Suite — the generated, storable task data (design doc §4.1).

Part of the **experimental** ``kaggle_benchmarks.agentic`` package. Pydantic
models, so they validate and (de)serialize for free. ``Suite`` round-trips as
JSON today; the intended home is the proto (``BenchmarkTaskVersion`` + a scenario
message) via ``clients`` / ``kaggle``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterator, Literal

import pydantic

from kaggle_benchmarks import actors
from kaggle_benchmarks.agentic._render import PanelRenderable

Role = Literal["system", "user", "assistant", "developer", "tool"]


class Persona(actors.Actor):
    """The simulated user of a scenario — an ``Actor`` with a profile and goal.

    Being an ``Actor`` means the persona is a first-class speaker: it can send
    messages and shows up in a trajectory/chat with its own name + avatar, just
    like the system, the user, or a tool.
    """

    def __init__(
        self,
        profile: str = "",
        goal: str = "",
        *,
        name: str = "User",
        avatar: str = "🧑",
        role: Role = "user",
        id: str | None = None,
    ):
        super().__init__(name=name, role=role, avatar=avatar, id=id)
        self.profile = profile
        self.goal = goal

    def to_dict(self) -> dict[str, str]:
        return {**self.as_dict(), "profile": self.profile, "goal": self.goal}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Persona":
        return cls(
            profile=d.get("profile", ""),
            goal=d.get("goal", ""),
            name=d.get("name", "User"),
            avatar=d.get("avatar", "🧑"),
            role=d.get("role", "user"),
            id=d.get("id"),
        )

    def __repr__(self) -> str:
        return f"Persona(name={self.name!r}, goal={self.goal!r})"


class Scenario(pydantic.BaseModel, PanelRenderable):
    """One generated case.

    ``environment`` is the ground truth the emulated tools serve; ``hidden_nuances``
    is the twist the agent under test must discover.
    """

    # Persona is an Actor (a plain class), not a pydantic model, so allow it as
    # an arbitrary type and (de)serialize it explicitly below.
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    id: str
    persona: Persona
    shared_context: dict[str, Any] = pydantic.Field(default_factory=dict)
    hidden_nuances: list[str] = pydantic.Field(default_factory=list)
    environment: dict[str, Any] = pydantic.Field(default_factory=dict)
    expected_behaviors: list[str] = pydantic.Field(default_factory=list)
    rubric: dict[str, list[str]] = pydantic.Field(
        default_factory=dict
    )  # {"must":[],"nice":[]}
    tags: list[str] = pydantic.Field(default_factory=list)
    provenance: dict[str, Any] = pydantic.Field(default_factory=dict)

    @pydantic.field_validator("persona", mode="before")
    @classmethod
    def _coerce_persona(cls, value: Any) -> Persona:
        if isinstance(value, Persona):
            return value
        if isinstance(value, dict):
            return Persona.from_dict(value)
        raise TypeError(f"persona must be a Persona or dict, got {type(value)!r}")

    @pydantic.field_serializer("persona")
    def _dump_persona(self, persona: Persona) -> dict[str, str]:
        return persona.to_dict()

    def signature(self) -> str:
        """Text fingerprint used for de-duplication (see ``agentic.fairness``)."""
        return " ".join(
            [self.persona.profile, self.persona.goal, *self.hidden_nuances]
        ).lower()

    def __panel__(self):
        import panel as pn

        must = "\n".join(f"- {m}" for m in self.rubric.get("must", [])) or "- (none)"
        nice = "\n".join(f"- {m}" for m in self.rubric.get("nice", [])) or "- (none)"
        hidden = "\n".join(f"- {h}" for h in self.hidden_nuances) or "- (none)"
        header = pn.pane.Markdown(
            f"### 🧪 Scenario `{self.id}`\n"
            f"**Persona:** {self.persona.profile}\n\n"
            f"**Goal:** {self.persona.goal}\n\n"
            f"**Tags:** {', '.join(self.tags) or '—'}"
        )
        rubric = pn.pane.Markdown(f"**Must:**\n{must}\n\n**Nice:**\n{nice}")
        # Hidden info is collapsed — it's for the examiner, not the agent.
        secret = pn.Card(
            pn.pane.Markdown(hidden),
            title="🔒 Hidden nuances (examiner-only)",
            collapsed=True,
            styles={"background": "#fff8e1"},
        )
        return pn.Column(header, rubric, secret, sizing_mode="stretch_width")


class Suite(pydantic.BaseModel, PanelRenderable):
    scenarios: list[Scenario] = pydantic.Field(default_factory=list)
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)

    def version(self) -> str:
        """Content hash — stamp into every Run so edits invalidate stale results."""
        blob = json.dumps([s.signature() for s in self.scenarios], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def __len__(self) -> int:
        return len(self.scenarios)

    def __iter__(self) -> Iterator[Scenario]:  # type: ignore[override]
        return iter(self.scenarios)

    def __getitem__(self, i: int) -> Scenario:
        return self.scenarios[i]

    def save(self, path: str) -> None:
        data = self.model_dump()
        data["metadata"] = {**self.metadata, "version": self.version()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Suite":
        with open(path) as f:
            return cls.model_validate(json.load(f))

    def __panel__(self):
        import panel as pn

        rows = [
            pn.pane.Markdown(
                f"- `{s.id}` — {s.persona.goal}  \n  *tags: {', '.join(s.tags)}*"
            )
            for s in self.scenarios
        ]
        meta = ", ".join(f"{k}={v}" for k, v in self.metadata.items())
        return pn.Column(
            pn.pane.Markdown(
                f"### 📚 Suite ({len(self)} scenarios) · `{self.version()}`"
            ),
            pn.pane.Markdown(meta or "_no metadata_"),
            *rows,
            sizing_mode="stretch_width",
        )
