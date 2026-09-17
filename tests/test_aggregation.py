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

"""Reducing a set of runs to one number."""

import pytest

from kaggle_benchmarks import results, runs, tasks, utils
from kaggle_benchmarks.runs import Split


def scored(result_type, values, **kwargs):
    """A Runs of finished runs of the given type, holding the given results."""

    @tasks.task(name=f"T{id(values)}", store_task=False, store_run=False)
    def t():
        pass

    task = tasks.Task(func=t.func, name=t.name, result_type=result_type)
    return runs.Runs(
        [
            runs.Run(task=task, result=value, status=utils.Status.SUCCESS, **kwargs)
            for value in values
        ]
    )


def test_filtering_splits_the_runs_in_two():
    public = runs.Run(task=None, split=Split.PUBLIC)
    private = runs.Run(task=None, split=Split.PRIVATE)
    unsplit = runs.Run(task=None)
    everything = runs.Runs([public, private, unsplit])

    assert list(everything.public) == [public]
    assert list(everything.private) == [private]
    assert list(everything.filter(None)) == [unsplit]


@pytest.mark.parametrize(
    "result_type, values, expected",
    [
        (results.Score, [1.0, 2.0, 6.0], 3.0),
        (results.Numerical, [1, 2], 1.5),
        (results.Boolean, [True, True, False, False], 0.5),
        (results.PassFail, [None, None, results.FAILED], 2 / 3),
        # Pooled, so a row with many attempts outweighs a row with one.
        (results.PassCount, [(1, 1), (1, 100)], 2 / 101),
    ],
)
def test_the_default_for_each_result_type(result_type, values, expected):
    assert scored(result_type, values).score() == pytest.approx(expected)


def test_a_type_with_no_default_says_what_to_pass():
    with pytest.raises(ValueError, match="Pass aggregate="):
        scored(results.Dictionary, [{"score": 1.0}]).score()


def test_an_explicit_aggregate_wins():
    numbers = scored(results.Score, [1.0, 9.0])

    assert numbers.score(aggregate=lambda rs: max(r.result for r in rs)) == 9.0


def test_a_crash_counts_against_a_pass_rate():
    """Otherwise a benchmark could be improved by breaking it."""
    passing = scored(results.PassFail, [None, None])
    passing[0].status = utils.Status.FAILED

    assert passing.score() == 0.5


def test_a_crash_is_dropped_from_a_mean(caplog):
    """There is no number that means "failed" on a scale we do not know."""
    numbers = scored(results.Score, [2.0, 4.0, results.FAILED])

    assert numbers.score() == 3.0
    assert "left out of the mean" in caplog.text


def test_a_cached_pass_fail_run_uses_its_stored_value():
    """`Run.passed` returns True for anything cached, which would score 1.0."""
    cached = scored(results.PassFail, [True, False], cached=True)

    assert cached.score() == 0.5


def test_scoring_nothing_refuses():
    with pytest.raises(ValueError, match="empty"):
        runs.Runs().score()
