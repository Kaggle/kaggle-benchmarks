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

import operator

import pandas as pd
import panel as pn
import pytest

from kaggle_benchmarks import chats, config, runs, tasks, utils


@tasks.task()
def task_that_errors():
    raise ValueError("This is an intentional error")


@tasks.task()
def bench(llm, message: str = "hi"):
    assert llm.prompt(message).lower() in {"quack", "honk"}


def test_task(duck):
    assert bench(duck) is None
    run = bench.run(duck, message="hey")

    assert isinstance(run, runs.Run)
    assert run.passed
    assert len(run.chat.messages) == 2
    assert run.chat.messages[0].text == "hey"
    assert run.chat.messages[1].text == "quack"
    assert run.chat.messages[1].sender == duck


def test_grid(duck):
    result = bench.evaluate(
        grid={"llm": [duck]},
        evaluation_data=pd.DataFrame({"message": ["meow", "howl", "moo"]}),
    )

    assert isinstance(result, runs.Runs)
    assert len(result.runs) == 3


def test_evaluate_thread_safety(duck):
    num_runs = 1000

    evaluation_data = pd.DataFrame({"message": ["hi"] * num_runs})

    results = bench.evaluate(grid={"llm": [duck]}, evaluation_data=evaluation_data)

    assert len(results.runs) == num_runs

    assert all(run.passed for run in results.runs)

    # Verify the integrity of the shared run counter.
    # We can't directly check the global _run_counters,
    # but we can check if the generated run IDs are unique.
    run_ids = {run.id for run in results.runs}
    assert len(run_ids) == num_runs


def test_evaluate_multi_job_safety(duck):
    num_runs = 1000

    evaluation_data = pd.DataFrame({"message": ["hi"] * num_runs})

    results = bench.evaluate(
        grid={"llm": [duck]}, evaluation_data=evaluation_data, n_jobs=4
    )

    assert len(results.runs) == num_runs

    run_ids = {run.id for run in results.runs}
    assert len(run_ids) == num_runs


class Failer:
    """A helper class that fails for a specified number of calls."""

    def __init__(self, fail_times: int):
        self.attempts = 0
        self.fail_times = fail_times

    def __call__(self, *args, **kwargs):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise ValueError(f"Intentional failure on attempt {self.attempts}")
        return


def test_evaluate_with_retries_succeeds_on_retry(monkeypatch):
    monkeypatch.setattr(config, "continue_with_exceptions", True)
    failer = Failer(fail_times=1)

    # With 1 retry, it should fail once and then succeed on the 2nd attempt.
    @tasks.task()
    def fallible_task():
        return failer()

    # Stop after 2 attempts if the task is still failing.
    runs = fallible_task.evaluate(
        stop_condition=lambda runs: len(runs.runs) == 1,
        max_attempts=5,
    )

    assert len(runs.runs) == 1
    run = runs.runs[0]
    assert run.passed
    assert failer.attempts == 2


def test_evaluate_with_retries_fails_after_retries(monkeypatch):
    monkeypatch.setattr(config, "continue_with_exceptions", True)
    failer = Failer(fail_times=10)

    # With 2 retries, it will attempt 3 times (1 initial + 2 retries) and fail each time.
    @tasks.task()
    def fallible_task():
        return failer()

    runs = fallible_task.evaluate(
        stop_condition=lambda runs: len(runs.runs) == 1,
        max_attempts=2,
    )

    assert len(runs.runs) == 0
    assert failer.attempts == 2


@pytest.mark.parametrize("continue_with_exceptions", [True, False])
def test_standalone_run_on_exception(monkeypatch, continue_with_exceptions):
    monkeypatch.setattr(config, "continue_with_exceptions", continue_with_exceptions)
    if continue_with_exceptions:
        run = task_that_errors.run()
        assert not run.passed
        assert run.status == utils.Status.FAILED
        assert (
            run.error_message
            and 'ValueError("This is an intentional error")' in run.error_message
        )
    else:
        with pytest.raises(ValueError, match="This is an intentional error"):
            task_that_errors.run()


@pytest.mark.parametrize("continue_with_exceptions", [True, False])
def test_run_with_subrun_on_exception(monkeypatch, continue_with_exceptions):
    monkeypatch.setattr(config, "continue_with_exceptions", continue_with_exceptions)

    @tasks.task()
    def wrapper_task():
        task_that_errors.run()

    if continue_with_exceptions:
        run = wrapper_task.run()
        assert not run.passed
        assert run.status == utils.Status.FAILED
        assert (
            run.error_message
            and 'ValueError("This is an intentional error")' in run.error_message
        )
    else:
        with pytest.raises(ValueError, match="This is an intentional error"):
            wrapper_task.run()


@pytest.mark.parametrize("continue_with_exceptions", [True, False])
def test_run_evaluating_subrun_on_exception(
    duck, monkeypatch, continue_with_exceptions
):
    monkeypatch.setattr(config, "continue_with_exceptions", continue_with_exceptions)

    @tasks.task()
    def parameterized_task_that_errors(llm, x):
        raise ValueError("This is an intentional error")

    df = pd.DataFrame({"x": [1, 2, 3]})

    @tasks.task()
    def wrapper_task(llm):
        parameterized_task_that_errors.evaluate(
            stop_condition=lambda runs: len(runs) == df.shape[0],
            max_attempts=1,
            retry_delay=2,
            llm=[llm],
            evaluation_data=df,
            n_jobs=1,
        )

    if continue_with_exceptions:
        run = wrapper_task.run(llm=duck)
        assert not run.passed
        assert run.status == utils.Status.FAILED
        assert (
            run.error_message
            and 'ValueError("This is an intentional error")' in run.error_message
        )
    else:
        with pytest.raises(ValueError, match="This is an intentional error"):
            wrapper_task.run(llm=duck)


@pytest.mark.parametrize("continue_with_exceptions", [True, False])
def test_evaluate_on_exception(monkeypatch, continue_with_exceptions):
    monkeypatch.setattr(config, "continue_with_exceptions", continue_with_exceptions)

    failer = Failer(fail_times=2)

    @tasks.task()
    def fallible_task():
        failer()

    if continue_with_exceptions:
        # It will eventually succeed with enough attempts.
        runs = fallible_task.evaluate(
            stop_condition=lambda runs: len(runs.runs) == 1,
            max_attempts=3,
        )
        assert len(runs.runs) == 1
        assert failer.attempts == 3
        assert runs.runs[0].passed
        assert runs.runs[0].status == utils.Status.SUCCESS
        assert runs.runs[0].error_message is None
    else:
        with pytest.raises(ValueError, match="Intentional failure on attempt 1"):
            runs = fallible_task.evaluate(
                stop_condition=lambda runs: len(runs.runs) == 1,
                max_attempts=3,
            )


def test_evaluate_with_retries_parallel(monkeypatch):
    monkeypatch.setattr(config, "continue_with_exceptions", True)

    @tasks.task()
    def fallible_task(failer):
        return failer()

    evaluation_data = pd.DataFrame(
        {"failer": [Failer(fail_times=i) for i in range(4)]},
        index=[f"run{i}" for i in range(4)],
    )

    runs = fallible_task.evaluate(
        evaluation_data=evaluation_data,
        n_jobs=2,
        stop_condition=lambda runs: len(runs.runs) == 4,
        max_attempts=5,
    )

    assert len(runs) == 4
    assert all(run.passed for run in runs)


def test_subruns(duck):
    @tasks.task()
    def outer(llm):
        bench.run(llm, "hi")
        bench.run(llm, "moo")

    run = outer.run(duck)
    assert len(run.subruns) == 2


def test_assertions():
    @tasks.task()
    def failed():
        assert False

    run = failed.run()
    assert not run.result
    assert run.chat
    assert len(run.chat.messages) > 0

    @tasks.task()
    def nested_fail():
        with chats.new():
            assert False, "no reason"

    run = nested_fail.run()
    assert not run.result
    assert run.chat
    assert len(run.chat.messages) > 0


def test_bind_dataframe():
    @tasks.task()
    def task(op, x, y) -> bool:
        return bool(op(x, y) % 3)

    bound = task.bind_dataframe(pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))
    run = bound.run(op=operator.add)
    assert run.result == (2, 3)


@pytest.mark.parametrize("mode", ["tabs", "columns"])
def test_pivot(duck, goose, mode):
    index = ["a", "b", "c"]
    runs = bench.evaluate(
        grid={"llm": [duck, goose]},
        evaluation_data=pd.DataFrame({"message": ["meow", "howl", "moo"]}, index=index),
    )

    pivot = runs.pivot(by="llm", mode=mode)
    assert isinstance(pivot, pn.viewable.Viewable)


def test_group():
    @tasks.task()
    def a(x, y):
        return x == y

    @tasks.task()
    def group(x):
        return a.evaluate(x=[x], y=["a", "b", "c"]).as_dataframe().result.mean()

    result = group.evaluate(x=["b", "c", "d"])
    pane = result.group_by(by="x")
    assert isinstance(pane, pn.widgets.Tabulator)


def test_params():
    @tasks.task()
    def x(a, b, c=3, **kw) -> int:
        return a + b + c + sum(kw.values())

    run = x.run(1, b=2, d=4, e=5)
    assert run.params == {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}


def test_partial():
    @tasks.task(name="math")
    def math(func, x, y) -> float:
        return func(x, y)

    partial = math.partial({"func": lambda x, y: x + y})
    assert isinstance(partial, tasks.Task)
    assert partial.name == "math"
    assert partial.run(x=1, y=2).result == 3

    partial = math.partial({"func": lambda x, y: x + y}, name="addition")
    assert isinstance(partial, tasks.Task)
    assert partial.name == "addition"
    assert partial.run(x=1, y=2).result == 3


def test_task_length_limits_enforced(monkeypatch):
    monkeypatch.setattr(config, "task_name_max_length", 5)
    monkeypatch.setattr(config, "task_description_max_length", 5)

    @tasks.task(name="ok", description="short")
    def within_limit():
        pass

    with pytest.raises(ValueError, match=r"Task name is 6 characters"):
        tasks.task(name="x" * 6)(lambda: None)

    with pytest.raises(ValueError, match=r"Task description is 6 characters"):
        tasks.task(name="ok", description="y" * 6)(lambda: None)


def test_task_length_limits_disabled_when_none(monkeypatch):
    monkeypatch.setattr(config, "task_name_max_length", None)
    monkeypatch.setattr(config, "task_description_max_length", None)

    @tasks.task(name="x" * 1000, description="y" * 1000)
    def unbounded():
        pass

    assert len(unbounded.name) == 1000


def test_nested_task_evaluate_with_retries_coerces_to_one(duck, caplog):
    @tasks.task()
    def single_task(llm, x):
        return llm.prompt(str(x))

    @tasks.task()
    def multi_task(llm, df):
        return single_task.evaluate(llm=[llm], evaluation_data=df, max_attempts=2)

    df = pd.DataFrame({"x": [1, 2, 3]})
    with caplog.at_level("WARNING", logger="kaggle_benchmarks.tasks"):
        run = multi_task.run(duck, df)

    assert any(
        "`max_attempts` must be 1 for nested task evaluations" in record.message
        for record in caplog.records
    )
    nested_runs = run.result
    assert isinstance(nested_runs, runs.Runs)
    assert len(nested_runs) == df.shape[0]
