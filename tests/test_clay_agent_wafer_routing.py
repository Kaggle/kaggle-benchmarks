# Copyright 2026 Issac Davis
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

from __future__ import annotations

import ast
import contextlib
import runpy
import sys
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / "agent_tasks" / "clay_agent_wafer_routing.py"


class _FakeTask:
    def __init__(self, func, metadata: dict[str, Any]) -> None:
        self.func = func
        self.metadata = metadata
        self.last_result: Any = None

    def run(self, llm):
        self.last_result = self.func(llm)
        return self


class _FakeChats:
    @staticmethod
    @contextlib.contextmanager
    def new(_name: str):
        yield


class _FakeAssertions:
    @staticmethod
    def assert_true(value: bool, expectation: str | None = None) -> None:
        if not value:
            raise AssertionError(expectation or "expected true")


class _FakeLLM:
    def __init__(self, expected_by_display: dict[str, int], mode: str) -> None:
        self.expected_by_display = expected_by_display
        self.mode = mode

    def prompt(self, prompt: str, *, tools: list, **_kwargs: Any) -> str:
        display, expected = next(
            (display, value)
            for display, value in self.expected_by_display.items()
            if f"Equation: {display}" in prompt
        )
        del display
        if self.mode == "correct":
            tools[0](expected)
        elif self.mode == "wrong":
            tools[0](expected + 1)
        elif self.mode == "multiple":
            tools[0](expected)
            tools[0](expected)
        elif self.mode != "missing":
            raise ValueError(f"unknown fake mode: {self.mode}")
        return "done"


def _literal_cases() -> tuple[dict[str, Any], ...]:
    tree = ast.parse(TASK_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "CASES":
                return ast.literal_eval(node.value)
    raise AssertionError("CASES literal not found")


def _execute(monkeypatch: pytest.MonkeyPatch, mode: str):
    cases = _literal_cases()
    expected = {str(case["display"]): int(case["expected"]) for case in cases}
    fake_module = types.ModuleType("kaggle_benchmarks")
    fake_module.chats = _FakeChats()
    fake_module.assertions = _FakeAssertions()
    fake_module.llm = _FakeLLM(expected, mode)

    def task(**metadata):
        def decorate(func):
            return _FakeTask(func, metadata)

        return decorate

    fake_module.task = task
    monkeypatch.setitem(sys.modules, "kaggle_benchmarks", fake_module)
    return runpy.run_path(str(TASK_PATH))


def test_case_contract_is_exact_and_balanced() -> None:
    cases = _literal_cases()
    assert len(cases) == 16
    assert len({case["id"] for case in cases}) == len(cases)
    family_counts: dict[str, int] = {}
    for case in cases:
        assert set(case) == {"id", "family", "a", "b", "c", "display", "expected"}
        assert type(case["a"]) is int and case["a"] != 0
        assert type(case["b"]) is int and type(case["c"]) is int
        assert (case["c"] - case["b"]) % case["a"] == 0
        assert case["expected"] == (case["c"] - case["b"]) // case["a"]
        assert -16 <= case["expected"] <= 16
        family = str(case["family"])
        family_counts[family] = family_counts.get(family, 0) + 1
    assert set(family_counts.values()) == {2, 4}
    assert sum(family_counts.values()) == 16


def test_file_has_matching_task_and_top_level_run() -> None:
    tree = ast.parse(TASK_PATH.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module == "__future__"
        for node in tree.body
    )
    task_names = []
    has_top_level_run = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and isinstance(
                    decorator.func, ast.Attribute
                ):
                    if decorator.func.attr == "task":
                        for keyword in decorator.keywords:
                            if keyword.arg == "name":
                                task_names.append(keyword.value)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            function = node.value.func
            has_top_level_run |= (
                isinstance(function, ast.Attribute) and function.attr == "run"
            )
    assert len(task_names) == 1
    assert isinstance(task_names[0], ast.Name) and task_names[0].id == "TASK_NAME"
    assert has_top_level_run


@pytest.mark.parametrize(
    ("mode", "expected_score"),
    [("correct", 1.0), ("wrong", 0.0), ("missing", 0.0), ("multiple", 0.0)],
)
def test_external_socket_grading(
    monkeypatch: pytest.MonkeyPatch, mode: str, expected_score: float
) -> None:
    namespace = _execute(monkeypatch, mode)
    task = namespace["clay_agent_wafer_routing"]
    assert task.last_result == expected_score


def test_sacred_egg_timer_is_injected_one_use_and_hash_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _execute(monkeypatch, "correct")
    timer_type = namespace["SacredEggTimer"]
    timer = timer_type.lay(
        "test",
        budget_s=2,
        period_s=1,
        now_monotonic_ns=10,
        now_utc_ns=2_000_000_000,
    )
    receipt = timer.hatch(
        now_monotonic_ns=1_000_000_010,
        now_utc_ns=3_000_000_000,
    )
    assert receipt["within_budget"] is True
    assert receipt["laid_window"] == 2
    assert receipt["hatched_window"] == 3
    assert len(receipt["receipt_sha256"]) == 64
    with pytest.raises(RuntimeError, match="already hatched"):
        timer.hatch(now_monotonic_ns=1_000_000_011, now_utc_ns=3_000_000_001)

    expired = timer_type.lay(
        "expired",
        budget_s=1,
        now_monotonic_ns=0,
        now_utc_ns=0,
    ).hatch(now_monotonic_ns=1_000_000_001, now_utc_ns=1_000_000_001)
    assert expired["within_budget"] is False
    assert expired["status"] == "EXPIRED"
