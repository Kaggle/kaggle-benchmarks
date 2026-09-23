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

import pytest

from kaggle_benchmarks import assertions, results, runs, tasks, transient


def test_sentinels_print_as_names():
    """They reach DataFrames and logs, where an object address is useless."""
    assert repr(results.PENDING) == "PENDING"
    assert repr(results.UNMEASURED) == "UNMEASURED"
    assert f"{results.FAILED}" == "FAILED"


def test_unmeasured_is_its_own_sentinel():
    assert results.UNMEASURED is not results.FAILED
    assert not results.UNMEASURED


def test_raising_unmeasured_does_not_propagate():
    @tasks.task(name="Unmeasurable")
    def my_task():
        raise tasks.Unmeasured("provider returned 503")

    run = my_task.run()

    assert run.result is results.UNMEASURED
    assert not run.measured
    assert run.error_message == "provider returned 503"


def test_an_unmeasured_run_does_not_read_as_a_failure():
    """❌ would claim the model was asked and got it wrong."""

    @tasks.task(name="Unmeasurable")
    def unmeasurable():
        raise tasks.Unmeasured("503")

    @tasks.task(name="Wrong")
    def wrong():
        assertions.assert_true(False)

    assert unmeasurable.run().format_result() == "⚠️"
    assert wrong.run().format_result() != "⚠️"


def test_siblings_still_run_after_an_unmeasured_task():
    """The reason Unmeasured is swallowed rather than raised."""
    ran = []

    @tasks.task(name="First")
    def first():
        ran.append("first")

    @tasks.task(name="Skipped")
    def skipped():
        raise tasks.Unmeasured("503")

    @tasks.task(name="Third")
    def third():
        ran.append("third")

    @tasks.benchmark(name="Suite")
    def suite() -> float:
        subruns = [first.run(), skipped.run(), third.run()]
        return subruns[0].passed

    suite.run()
    assert ran == ["first", "third"]


def test_a_real_exception_still_propagates():
    @tasks.task(name="Buggy")
    def buggy():
        raise TypeError("a genuine bug")

    with pytest.raises(TypeError, match="a genuine bug"):
        buggy.run()


def _run(result, passed=True):
    """A Run standing in for one that already finished."""

    @tasks.task(name="stub")
    def stub():
        pass

    run = runs.Run(task=stub, result=result)
    if not passed:
        run.assertion_results.append(
            assertions.AssertionResult(passed=False, expectation="nope")
        )
    return run


def test_score_leaves_unmeasured_out_of_the_denominator():
    group = runs.Runs([_run(None), _run(None, passed=False), _run(results.UNMEASURED)])

    score = group.score()

    assert score.value == 0.5  # 1 of the 2 that were measured
    assert (score.measured, score.total) == (2, 3)
    assert not score.complete


def test_score_reports_when_nothing_could_be_measured():
    group = runs.Runs([_run(results.UNMEASURED), _run(results.UNMEASURED)])

    score = group.score()

    assert score.value is None
    assert str(score) == "no measurements (0/2)"


def test_score_carries_its_denominator_into_its_repr():
    """1.0 over one task must not read like 1.0 over twenty."""
    thin = runs.Runs([_run(None), _run(results.UNMEASURED)]).score()
    full = runs.Runs([_run(None), _run(None)]).score()

    assert thin.value == full.value == 1.0
    assert str(thin) != str(full)
    assert "1/2" in str(thin)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("503 UNAVAILABLE: model overloaded"),
        RuntimeError("Error code: 429 - rate limit exceeded"),
        TimeoutError("took too long"),
        RuntimeError("upstream connection reset"),
    ],
)
def test_transient_errors_are_recognised(error):
    assert transient.is_transient(error)


@pytest.mark.parametrize(
    "error",
    [
        ValueError("a genuine bug"),
        RuntimeError("Error code: 400 - your request was malformed"),
        RuntimeError("Error code: 401 - expired token"),
        KeyError("missing"),
    ],
)
def test_ordinary_errors_are_not_mistaken_for_transient(error):
    assert not transient.is_transient(error)


def test_status_code_attribute_is_preferred_over_the_message():
    class ApiError(Exception):
        status_code = 503

    assert transient.is_transient(ApiError("something went wrong"))


def test_unmeasured_block_converts_only_transient_errors():
    with pytest.raises(tasks.Unmeasured, match="503"):
        with transient.unmeasured():
            raise RuntimeError("503 UNAVAILABLE")

    with pytest.raises(ValueError, match="genuine"):
        with transient.unmeasured():
            raise ValueError("genuine bug")


def test_unmeasured_block_is_transparent_when_nothing_raises():
    with transient.unmeasured():
        value = 1 + 1
    assert value == 2


def test_an_unmeasured_run_goes_out_as_errored_with_no_result():
    """The wire has only COMPLETED and ERRORED, and no verdict is not a pass."""
    from kaggle_benchmarks.kaggle import benchmark_types_pb2 as proto_types
    from kaggle_benchmarks.kaggle import serialization

    @tasks.task(name="Unmeasurable")
    def unmeasurable():
        raise tasks.Unmeasured("503")

    run = unmeasurable.run()

    assert serialization._run_state(run) == (
        proto_types.BenchmarkTaskRunState.BENCHMARK_TASK_RUN_STATE_ERRORED
    )
    # Would have raised NotImplementedError before UNMEASURED was handled.
    assert serialization._prepare_results_data(run) == []


def test_the_block_reads_as_a_task_would_use_it():
    @tasks.task(name="Flaky")
    def flaky():
        with transient.unmeasured():
            raise RuntimeError("503 UNAVAILABLE")

    run = flaky.run()

    assert run.result is results.UNMEASURED
    assert "503" in run.error_message
