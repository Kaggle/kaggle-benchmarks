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

"""Trajectory analyzers (design doc §3.3).

Each analyzer is ``Trajectory -> AssertionResult`` — reusing the library's own
``AssertionResult`` (which already renders itself via ``__panel__``). The error
class for the taxonomy rides in ``AssertionResult.details["error_class"]``.
"""

from __future__ import annotations

from typing import Callable, Protocol

from kaggle_benchmarks.agentic.trajectory import Trajectory
from kaggle_benchmarks.assertions import AssertionResult

Analyzer = Callable[[Trajectory], AssertionResult]


class ErrorClass:
    """Straw-man failure taxonomy (design doc §3.3) — edit freely."""

    MISSED_HIDDEN_CONSTRAINT = "missed_hidden_constraint"
    TOOL_UNDERUSE = "tool_underuse"
    TOOL_OVERUSE = "tool_overuse"
    WRONG_TOOL = "wrong_tool"
    HALLUCINATION = "hallucinated_fact"
    IGNORED_GOAL = "ignored_user_goal"
    UNSAFE_ACTION = "unsafe_action"
    GAVE_UP_EARLY = "gave_up_early"
    FORMAT_VIOLATION = "format_violation"


class _Prompter(Protocol):
    def prompt(self, text: str, **kwargs) -> str: ...


def _result(
    name: str, passed: bool, expectation: str, error_class: str, detail: str
) -> AssertionResult:
    return AssertionResult(
        passed=passed,
        expectation=expectation,
        details={
            "analyzer": name,
            "error_class": None if passed else error_class,
            "detail": detail,
        },
    )


def error_class_of(result: AssertionResult) -> str | None:
    return (result.details or {}).get("error_class")


# --- structural ---
def called_tool(name: str, *, error_class: str = ErrorClass.TOOL_UNDERUSE) -> Analyzer:
    def check(traj: Trajectory) -> AssertionResult:
        ok = traj.called(name)
        return _result(
            f"called_tool({name})",
            ok,
            f"tool {name!r} is called",
            error_class,
            "" if ok else f"{name!r} never called",
        )

    return check


def no_tool_errors() -> Analyzer:
    def check(traj: Trajectory) -> AssertionResult:
        # Emulated tools signal failure by returning an Exception as the output
        # (ToolInvocationResult has no error flag).
        bad = [r.name for r in traj.tool_results() if isinstance(r.output, Exception)]
        return _result(
            "no_tool_errors",
            not bad,
            "no tool returned an error",
            ErrorClass.WRONG_TOOL,
            "" if not bad else f"errors: {bad}",
        )

    return check


def max_steps(n: int) -> Analyzer:
    def check(traj: Trajectory) -> AssertionResult:
        ok = len(traj.steps) <= n
        return _result(
            f"max_steps({n})",
            ok,
            f"trajectory has <= {n} steps",
            ErrorClass.GAVE_UP_EARLY,
            f"{len(traj.steps)} steps",
        )

    return check


# --- reasoning / answer inspection ---
def reasoning_mentions(*terms: str) -> Analyzer:
    def check(traj: Trajectory) -> AssertionResult:
        text = traj.reasoning_text().lower()
        hit = [t for t in terms if t.lower() in text]
        return _result(
            f"reasoning_mentions({'|'.join(terms)})",
            bool(hit),
            f"reasoning mentions one of {terms}",
            ErrorClass.MISSED_HIDDEN_CONSTRAINT,
            f"matched {hit}" if hit else "no match",
        )

    return check


def answer_mentions(
    *terms: str, error_class: str = ErrorClass.MISSED_HIDDEN_CONSTRAINT
) -> Analyzer:
    def check(traj: Trajectory) -> AssertionResult:
        text = traj.final_text().lower()
        hit = [t for t in terms if t.lower() in text]
        return _result(
            f"answer_mentions({'|'.join(terms)})",
            bool(hit),
            f"answer surfaces one of {terms}",
            error_class,
            f"matched {hit}" if hit else "not surfaced",
        )

    return check


# --- judge-based (LLM). Pass any object with .prompt(str) -> str (an LLMChat,
#     or a mock). Rotate a judge panel for fairness (design doc §10). ---
def judge(
    question: str,
    judge_llm: _Prompter,
    *,
    error_class: str = ErrorClass.MISSED_HIDDEN_CONSTRAINT,
) -> Analyzer:
    def check(traj: Trajectory) -> AssertionResult:
        prompt = (
            f"{question}\n\nAgent trajectory:\n{traj.full_text()}\n\n"
            "Answer with YES or NO, then a brief reason."
        )
        verdict = str(judge_llm.prompt(prompt))
        passed = "yes" in verdict.strip().lower()[:5]
        return _result(
            f"judge({question[:40]})",
            passed,
            question,
            error_class,
            verdict.strip()[:120],
        )

    return check


def run_analyzers(traj: Trajectory, analyzers: list[Analyzer]) -> list[AssertionResult]:
    return [a(traj) for a in analyzers]
