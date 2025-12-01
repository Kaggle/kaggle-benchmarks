# Copyright 2025 Kaggle Inc.
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

import random
import time

import pandas as pd

from kaggle_benchmarks import orchestration
from kaggle_benchmarks.tasks import task


def test_combinations():
    result = list(orchestration.sample_combinations({"a": [1, 2, 3], "b": ["a", "b"]}))
    assert len(result) == 6
    assert {(1, "a"), (2, "a"), (3, "a"), (1, "b"), (2, "b"), (3, "b")} == {
        (r["a"], r["b"]) for r in result
    }


def test_run():
    def trial(a, b, c):
        return f"{a}{b}{c}"

    result = list(
        orchestration.evaluate_function(
            func=trial,
            grid={"a": [1, 2, 3], "b": ["a", "b"], "c": [1]},
        )
    )
    assert len(result) == 6

    result = list(
        orchestration.evaluate_function(
            func=trial,
            grid={"a": [1, 2, 3], "b": ["a", "b"]},
            evaluation_data=pd.DataFrame({"c": [2, 3, 4]}),
        )
    )
    assert len(result) == 18


def test_prepare_tasks():
    def trial(a, b, c):
        return f"{a}{b}{c}"

    result = list(
        orchestration.prepare_tasks(
            func=trial,
            grid={"a": [1, 2, 3], "b": ["a", "b"], "c": [1]},
            evaluation_data=pd.DataFrame({"c": [2, 3, 4]}),
            id_format="b={b}/{id}",
        )
    )
    assert len(result) == 18
    assert {r.id for r in result} == {f"b={b}/{i}" for b in "ab" for i in range(3)}


def test_run_tasks():
    def square(x):
        time.sleep(random.random() / 10)
        return x**2

    results = orchestration.run_tasks(
        [
            orchestration.Task(id=str(i), params={"x": i}, func=square)
            for i in range(10)
        ],
        n_jobs=4,
    )

    assert len(results) == 10
    assert results == {str(i): square(i) for i in range(10)}


@task()
def inner_task(x: int) -> int:
    return x * 10


@task()
def parent_task_parallel():
    """An orchestrator task that evaluates inner_task in parallel."""
    test_data = pd.DataFrame({"x": [1, 2, 3, 4]})

    inner_task.evaluate(evaluation_data=test_data, n_jobs=4)

    return None


def test_parallel_subruns_are_nested_correctly():
    """
    Tests that sub-runs executed in parallel (n_jobs > 1) are correctly
    nested under their parent run and not created as top-level runs.
    """
    parent_run = parent_task_parallel.run()

    assert parent_run.passed

    # Verify that the parent run object has captured the four parallel runs
    # as its sub-runs.
    assert len(parent_run.subruns) == 4

    # Verify the results of the sub-runs
    subrun_results = [run.result for run in parent_run.subruns]
    assert sorted(subrun_results) == [10, 20, 30, 40]
