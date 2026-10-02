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

"""LLM-as-a-judge benchmark task, and its golden tests.

``assess_with_judge`` has a *subject* model answer "What is Kaggle?" and then a
*second* (judge) model grade that answer against criteria via
:func:`kaggle_benchmarks.assertions.assess_response_with_judge`.

The task is followed by its tests. Because it takes two models, the scripted
tests script both sides through ``fake(...)`` and run with no API key, while the
live test pairs a subject model from ``REFERENCE_MODELS`` with a real
:func:`judge_model` and skips when no provider is configured. Tests asserting a
*failure* are scripted only — a real model may legitimately answer correctly.
"""

import pytest
from models import REFERENCE_MODELS, fake, judge_model

import kaggle_benchmarks as kbench

# The two criteria the judge scores, mirrored from the task body so the scripted
# report can echo them back verbatim.
_CRITERIA = [
    "The answer must mention data science or machine learning.",
    "The answer should mention competitions.",
]


def _judge_report(*passed: bool) -> dict:
    """Builds a scripted judge ``AssessReport`` payload (one result per criterion)."""
    return {
        "results": [
            {
                "criterion": criterion,
                "passed": ok,
                "reason": "Satisfied." if ok else "Not satisfied.",
                "confidence": 100,
            }
            for criterion, ok in zip(_CRITERIA, passed)
        ]
    }


@kbench.task(name="assess_with_judge")
def assess_with_judge(llm, judge_llm) -> None:
    """A task where the LLM answers a question, and a Judge LLM evaluates the answer."""
    response: str = llm.prompt("What is Kaggle?")
    kbench.assertions.assert_in("platform", response.lower())

    assessment = kbench.assertions.assess_response_with_judge(
        response_text=response,
        judge_llm=judge_llm,
        criteria=_CRITERIA,
    )

    for result in assessment.results:
        kbench.assertions.assert_true(
            result.passed,
            expectation=f"Judge Criterion '{result.criterion}' should pass: {result.reason}",
        )


def _scripted_run(subject_answer: str, judge_report: dict):
    """Runs the task with a scripted subject model and a scripted judge model."""
    return assess_with_judge.run(fake([subject_answer]), fake([judge_report]))


def test_assess_with_judge_scripted():
    assert _scripted_run(
        "Kaggle is a platform for data science and machine learning competitions.",
        _judge_report(True, True),
    ).passed


def test_assess_with_judge_judge_fails():
    assert not _scripted_run(
        "Kaggle is a platform for data science and machine learning competitions.",
        _judge_report(False, True),
    ).passed


def test_assess_with_judge_missing_platform_fails():
    # The subject omits the required word "platform". assert_in records the
    # failure without raising, so the judge still runs and passes.
    assert not _scripted_run(
        "Kaggle is a site for data science and machine learning competitions.",
        _judge_report(True, True),
    ).passed


@pytest.mark.parametrize("llm", REFERENCE_MODELS)
def test_assess_with_judge(llm):
    assert assess_with_judge.run(llm, judge_model()).passed
