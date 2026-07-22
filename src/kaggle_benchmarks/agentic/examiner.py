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

"""The Examiner: authors scenarios + grades trajectories (design doc §4.1/§4.3),
plus the fairness layer.

Authoring rotates a panel of author models and de-duplicates (see
``agentic.fairness``); grading runs the analyzers and aggregates a ``Report``.
Model calls are injected (``author_one`` / the analyzers' judge), so this is
deterministic and testable without a live model.
"""

from __future__ import annotations

from typing import Callable

import pydantic

from kaggle_benchmarks.agentic import fairness
from kaggle_benchmarks.agentic._render import PanelRenderable
from kaggle_benchmarks.agentic.analyzers import Analyzer, error_class_of, run_analyzers
from kaggle_benchmarks.agentic.scenario import Scenario, Suite
from kaggle_benchmarks.agentic.trajectory import Trajectory
from kaggle_benchmarks.assertions import AssertionResult


class Report(pydantic.BaseModel, PanelRenderable):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    scenario_id: str
    agent: str
    score: float
    passed: bool
    results: list[AssertionResult] = pydantic.Field(default_factory=list)
    error_classes: list[str] = pydantic.Field(default_factory=list)

    def __panel__(self):
        import panel as pn

        mark = "✅" if self.passed else "❌"
        errs = ", ".join(self.error_classes) or "—"
        header = pn.pane.Markdown(
            f"### {mark} `{self.agent}` on `{self.scenario_id}` — {self.score:.0%}\n"
            f"**Error classes:** {errs}"
        )
        # Each AssertionResult renders itself (assertions.AssertionResult.__panel__).
        return pn.Column(
            header, *[pn.panel(r) for r in self.results], sizing_mode="stretch_width"
        )


class Examiner:
    def __init__(self, author_models: list[str], judge_name: str = "mock-judge"):
        self.author_models = author_models
        self.judge_name = judge_name

    def author(
        self,
        problem: str,
        author_one: Callable[[str, str, int], Scenario],
        n: int,
        *,
        seed: int = 0,
        dedup_threshold: float = 0.9,
    ) -> Suite:
        """Generate ``n`` scenarios, rotating the author model, then de-dup."""
        raw: list[Scenario] = []
        for i in range(n):
            model = fairness.pick(self.author_models, i)
            sc = author_one(problem, model, i)
            sc.provenance = {"author_model": model, "seed": seed, "index": i}
            raw.append(sc)

        kept, dropped = fairness.dedup(raw, threshold=dedup_threshold)
        return Suite(
            scenarios=kept,
            metadata={
                "problem": problem,
                "author_models": self.author_models,
                "seed": seed,
                "dedup_threshold": dedup_threshold,
                "generated": len(raw),
                "kept": len(kept),
                "dropped_as_duplicate": len(dropped),
                "judge": self.judge_name,
            },
        )

    def grade(
        self,
        trajectory: Trajectory,
        scenario: Scenario,
        agent: str,
        analyzers: list[Analyzer],
    ) -> Report:
        results = run_analyzers(trajectory, analyzers)
        passed_n = sum(1 for r in results if r.passed)
        error_classes = sorted(
            {
                ec
                for r in results
                if not r.passed and (ec := error_class_of(r)) is not None
            }
        )
        return Report(
            scenario_id=scenario.id,
            agent=agent,
            score=passed_n / len(results) if results else 1.0,
            passed=passed_n == len(results),
            results=results,
            error_classes=error_classes,
        )
