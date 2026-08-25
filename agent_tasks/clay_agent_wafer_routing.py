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

# %%
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import kaggle_benchmarks as kbench

# %%
TASK_NAME = "clay-agent-wafer-routing"
SOCKET_MIN = -16
SOCKET_MAX = 16

# Local design authority: C:\dev\loom\egg_lock.py. Only its injected-clock,
# time-window, and one-use hatch semantics are reused here. This public timer
# has no secret and makes no authentication, encryption, or PQC claim.
SACRED_EGG_LINEAGE_SHA256 = (
    "ef30e0c74b84795388b3d205212de92fd291bd516375ad18c65b3efff58f29fc"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass
class SacredEggTimer:
    """One-use, injected-clock deadline receipt inspired by Sacred Egg timing.

    This is deliberately a public benchmark timer, not a security primitive.
    Monotonic time decides expiry; UTC supplies an auditable coarse window.
    Once hatched, the timer cannot be reused.
    """

    label: str
    budget_ns: int
    laid_monotonic_ns: int
    laid_utc_ns: int
    period_ns: int
    spent: bool = False

    @classmethod
    def lay(
        cls,
        label: str,
        budget_s: int,
        *,
        period_s: int = 30,
        now_monotonic_ns: int | None = None,
        now_utc_ns: int | None = None,
    ) -> "SacredEggTimer":
        if not label:
            raise ValueError("timer label must be non-empty")
        if type(budget_s) is not int or budget_s <= 0:
            raise ValueError("timer budget_s must be a positive integer")
        if type(period_s) is not int or period_s <= 0:
            raise ValueError("timer period_s must be a positive integer")
        return cls(
            label=label,
            budget_ns=budget_s * 1_000_000_000,
            laid_monotonic_ns=(
                time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
            ),
            laid_utc_ns=time.time_ns() if now_utc_ns is None else now_utc_ns,
            period_ns=period_s * 1_000_000_000,
        )

    def hatch(
        self,
        *,
        now_monotonic_ns: int | None = None,
        now_utc_ns: int | None = None,
    ) -> dict[str, Any]:
        if self.spent:
            raise RuntimeError("Sacred Egg timer already hatched")
        self.spent = True
        end_monotonic_ns = (
            time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        )
        end_utc_ns = time.time_ns() if now_utc_ns is None else now_utc_ns
        elapsed_ns = end_monotonic_ns - self.laid_monotonic_ns
        within_budget = 0 <= elapsed_ns <= self.budget_ns
        payload: dict[str, Any] = {
            "schema": "clay_sacred_egg_timer_receipt_v1",
            "timer_kind": "public_one_shot_deadline_receipt",
            "authentication_claim": False,
            "lineage_sha256": SACRED_EGG_LINEAGE_SHA256,
            "label": self.label,
            "laid_window": self.laid_utc_ns // self.period_ns,
            "hatched_window": end_utc_ns // self.period_ns,
            "budget_ns": self.budget_ns,
            "elapsed_ns": elapsed_ns,
            "within_budget": within_budget,
            "status": "HATCHED" if within_budget else "EXPIRED",
        }
        payload["receipt_sha256"] = hashlib.sha256(
            _canonical_json(payload).encode("ascii")
        ).hexdigest()
        return payload


CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "normal-01",
        "family": "normal",
        "a": 3,
        "b": 2,
        "c": 14,
        "display": "3*x + 2 = 14",
        "expected": 4,
    },
    {
        "id": "normal-02",
        "family": "normal",
        "a": -4,
        "b": 3,
        "c": 19,
        "display": "-4*x + 3 = 19",
        "expected": -4,
    },
    {
        "id": "reversed-01",
        "family": "reversed",
        "a": 2,
        "b": 4,
        "c": 10,
        "display": "10 = 2*x + 4",
        "expected": 3,
    },
    {
        "id": "reversed-02",
        "family": "reversed",
        "a": -3,
        "b": -2,
        "c": 7,
        "display": "7 = -3*x - 2",
        "expected": -3,
    },
    {
        "id": "constant-first-01",
        "family": "constant_first",
        "a": 2,
        "b": 5,
        "c": -7,
        "display": "5 + 2*x = -7",
        "expected": -6,
    },
    {
        "id": "constant-first-02",
        "family": "constant_first",
        "a": -2,
        "b": -8,
        "c": 4,
        "display": "-8 - 2*x = 4",
        "expected": -6,
    },
    {
        "id": "reversed-03",
        "family": "reversed",
        "a": 3,
        "b": 6,
        "c": 12,
        "display": "12 = 6 + 3*x",
        "expected": 2,
    },
    {
        "id": "reversed-04",
        "family": "reversed",
        "a": 2,
        "b": -5,
        "c": -15,
        "display": "-15 = -5 + 2*x",
        "expected": -5,
    },
    {
        "id": "zero-bias-01",
        "family": "zero_bias",
        "a": 7,
        "b": 0,
        "c": -21,
        "display": "7*x = -21",
        "expected": -3,
    },
    {
        "id": "zero-bias-02",
        "family": "zero_bias",
        "a": -9,
        "b": 0,
        "c": 45,
        "display": "-9*x = 45",
        "expected": -5,
    },
    {
        "id": "zero-answer-01",
        "family": "zero_answer",
        "a": 4,
        "b": 9,
        "c": 9,
        "display": "4*x + 9 = 9",
        "expected": 0,
    },
    {
        "id": "zero-answer-02",
        "family": "zero_answer",
        "a": 5,
        "b": -6,
        "c": -6,
        "display": "5*x - 6 = -6",
        "expected": 0,
    },
    {
        "id": "boundary-01",
        "family": "boundary",
        "a": 1,
        "b": 11,
        "c": -5,
        "display": "11 + x = -5",
        "expected": -16,
    },
    {
        "id": "boundary-02",
        "family": "boundary",
        "a": 1,
        "b": -13,
        "c": 3,
        "display": "-13 + x = 3",
        "expected": 16,
    },
    {
        "id": "mixed-sign-01",
        "family": "mixed_sign",
        "a": -2,
        "b": 7,
        "c": -9,
        "display": "-2*x + 7 = -9",
        "expected": 8,
    },
    {
        "id": "mixed-sign-02",
        "family": "mixed_sign",
        "a": 8,
        "b": -5,
        "c": -77,
        "display": "8*x - 5 = -77",
        "expected": -9,
    },
)


def _prompt(display: str) -> str:
    return (
        "You are controlling a typed algebra action socket. Solve the integer equation "
        "for x, then call submit_integer exactly once with the solution. The socket accepts "
        f"only integers from {SOCKET_MIN} through {SOCKET_MAX}. Do not use a second call to "
        "correct yourself; the first and only submitted action is final. The tool will only "
        "acknowledge recording and will not reveal correctness.\n\n"
        f"Equation: {display}"
    )


# %%
@kbench.task(
    name=TASK_NAME,
    description=(
        "Measures exact typed-tool routing on sixteen visible linear equations while "
        "Sacred Egg one-use timers record a bounded common harness."
    ),
    version=1,
)
def clay_agent_wafer_routing(llm) -> float:
    task_timer = SacredEggTimer.lay(TASK_NAME, budget_s=1800)
    correct = 0
    family_totals: dict[str, int] = {}
    family_correct: dict[str, int] = {}
    case_receipts: list[dict[str, Any]] = []

    for index, case in enumerate(CASES):
        submissions: list[Any] = []
        case_timer = SacredEggTimer.lay(case["id"], budget_s=180)

        def submit_integer(value: int) -> str:
            """Record one integer action. This never reveals whether the value is correct."""
            submissions.append(value)
            if type(value) is not int:
                return "REJECTED: value must be an integer"
            if not SOCKET_MIN <= value <= SOCKET_MAX:
                return "REJECTED: value is outside the socket range"
            return "RECORDED"

        error_type: str | None = None
        with kbench.chats.new(f"case-{case['id']}"):
            try:
                llm.prompt(
                    _prompt(case["display"]),
                    tools=[submit_integer],
                    temperature=0,
                    seed=index,
                )
            except Exception as exc:  # preserve the remaining independent cases
                error_type = type(exc).__name__

        receipt = case_timer.hatch()
        chosen = submissions[0] if len(submissions) == 1 else None
        is_correct = (
            error_type is None
            and len(submissions) == 1
            and type(chosen) is int
            and chosen == case["expected"]
        )
        correct += int(is_correct)
        family = str(case["family"])
        family_totals[family] = family_totals.get(family, 0) + 1
        family_correct[family] = family_correct.get(family, 0) + int(is_correct)
        case_receipts.append(
            {
                "case_id": case["id"],
                "family": family,
                "submission_count": len(submissions),
                "submitted_value": chosen if type(chosen) is int else None,
                "correct": is_correct,
                "error_type": error_type,
                "timer": receipt,
            }
        )

    task_receipt = task_timer.hatch()
    within_budget = task_receipt["within_budget"] and all(
        item["timer"]["within_budget"] for item in case_receipts
    )
    kbench.assertions.assert_true(
        within_budget,
        expectation="Every Sacred Egg timer must hatch before its monotonic deadline.",
    )

    family_accuracy = {
        family: family_correct[family] / total
        for family, total in sorted(family_totals.items())
    }
    audit = {
        "schema": "clay_agent_wafer_routing_audit_v1",
        "task": TASK_NAME,
        "correct": correct,
        "total": len(CASES),
        "accuracy": correct / len(CASES),
        "family_accuracy": family_accuracy,
        "task_timer": task_receipt,
        "case_receipts": case_receipts,
    }
    audit["audit_sha256"] = hashlib.sha256(
        _canonical_json(audit).encode("ascii")
    ).hexdigest()
    print("CLAY_AGENT_WAFER_AUDIT=" + _canonical_json(audit))
    return correct / len(CASES)


# %%
clay_agent_wafer_routing.run(kbench.llm)
