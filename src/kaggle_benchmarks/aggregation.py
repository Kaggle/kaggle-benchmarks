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

"""Reducing a set of runs to one number.

A task body computes the overall figure for the whole evaluation, but nothing
computes the public and private halves separately.

Result types with no obvious reduction are deliberately absent from the table
below. Which key of a dict is the score, and how confidence intervals
combine, are questions only the task author can answer.
"""

import logging
from typing import TYPE_CHECKING, Any, Callable

from kaggle_benchmarks import results

if TYPE_CHECKING:
    from kaggle_benchmarks import runs

logger = logging.getLogger(__name__)

Aggregator = Callable[["runs.Runs[Any]"], Any]


def _passed(run: "runs.Run[Any]") -> bool:
    """Whether this row counts as a pass.

    `Run.passed` returns True for any cached run (see the TODO on that
    property), which would score a cached evaluation 1.0. For those we use
    the value written to disk instead.
    """
    if run.cached and isinstance(run.result, bool):
        return run.result
    return run.passed


def _pass_rate(runs_: "runs.Runs[Any]") -> float:
    """The fraction of rows that passed.

    A crashed row counts as a failure rather than being dropped, so a
    benchmark cannot be improved by breaking it.
    """
    return sum(1 for run in runs_ if _passed(run)) / len(runs_)


def _mean(runs_: "runs.Runs[Any]") -> float:
    """The mean of the numeric results.

    A crashed row produced no number, so it is dropped and logged. Unlike
    the pass rate above, which counts one as a failure: there is no value
    meaning "failed" on a scale we do not know.
    """
    values = [
        run.result
        for run in runs_
        if isinstance(run.result, (int, float)) and not isinstance(run.result, bool)
    ]
    if skipped := len(runs_) - len(values):
        logger.warning(
            f"{skipped} of {len(runs_)} runs produced no number and were "
            "left out of the mean."
        )
    if not values:
        raise ValueError("No run produced a numeric result to average.")
    return sum(values) / len(values)


def _pass_count_rate(runs_: "runs.Runs[Any]") -> float:
    """Total passes over total attempts.

    Pooled rather than averaging per-row rates, so a row with a hundred
    attempts weighs a hundred times one with a single attempt.
    """
    pairs = [
        run.result
        for run in runs_
        if isinstance(run.result, tuple)
        and len(run.result) == 2
        and all(isinstance(part, (int, float)) for part in run.result)
    ]
    if skipped := len(runs_) - len(pairs):
        logger.warning(
            f"{skipped} of {len(runs_)} runs recorded no pass count and were "
            "left out of the rate."
        )
    attempted = sum(total for _, total in pairs)
    if not attempted:
        raise ValueError("No run recorded an attempt to count passes over.")
    return sum(passes for passes, _ in pairs) / attempted


DEFAULTS: dict[type, Aggregator] = {
    results.PassFail: _pass_rate,
    results.Boolean: _pass_rate,
    results.Numerical: _mean,
    results.Score: _mean,
    results.PassCount: _pass_count_rate,
}

# Completes the sentence "Task 'X' returns ..." in the error below.
_DESCRIPTIONS: dict[type, str] = {
    results.Dictionary: "a dict",
    results.MetricWithCI: "a metric with a confidence interval",
}


def has_default(result_type: type) -> bool:
    return result_type in DEFAULTS


def describe_missing_default(result_type: type, task_name: str) -> str:
    """The error shown when a task's results cannot be reduced on our own."""
    described = _DESCRIPTIONS.get(
        result_type, f"a result of type {result_type.__name__}"
    )
    return (
        f"Task {task_name!r} returns {described}, so kaggle-benchmarks cannot "
        "score the public/private split on its own. Pass aggregate= to "
        "evaluate(), for example:\n\n"
        "    aggregate=lambda runs: sum(r.result['score'] for r in runs) / len(runs)"
    )


def default_for(result_type: type, task_name: str = "this task") -> Aggregator:
    """The default aggregator for a result type.

    Raises with an explanation when the result type has no sensible default.
    """
    if not has_default(result_type):
        raise ValueError(describe_missing_default(result_type, task_name))
    return DEFAULTS[result_type]
