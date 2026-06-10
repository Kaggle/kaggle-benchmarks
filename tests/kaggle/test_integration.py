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

import tempfile

import pytest

from kaggle_benchmarks import (
    ExecutionMode,
    assertions,
    config,
    kaggle,
    runs,
    task,
    utils,
)


@pytest.fixture
def client(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        kaggle_client = kaggle.KaggleClient(directory=temp_dir)
        config.execution_mode = ExecutionMode.RUN
        # Reset global run counters to avoid flaky filenames: the counter
        # is keyed by id(task), and CPython can reuse addresses of GC'd
        # Task objects from earlier tests, inflating the counter.
        runs._run_counters.clear()

        monkeypatch.setattr("kaggle_benchmarks.client", kaggle_client)
        yield kaggle_client

        config.execution_mode = ExecutionMode.TESTING


@task()
def inner() -> bool:
    return True


@task()
def outer() -> bool:
    return inner.run().result and inner.run().result


def test_saves_subruns(client):
    outer.run()
    # Both the inner and outer run should be saved.
    assert (client.directory / "Outer-run_id_Run_1.run.json").is_file()
    assert (client.directory / "Inner-run_id_Run_2.run.json").is_file()
    assert (client.directory / "Inner-run_id_Run_2.run.json").is_file()


def test_run_assert(client):
    @task()
    def failed():
        assertions.assert_false(True)

    @task()
    def passed():
        assertions.assert_true(True)

    failed.run()
    with open(client.directory / "Failed-run_id_Run_1.run.json") as fp:
        assert "BENCHMARK_TASK_RUN_ASSERTION_STATUS_FAILED" in fp.read()

    passed.run()
    with open(client.directory / "Passed-run_id_Run_1.run.json") as fp:
        assert "BENCHMARK_TASK_RUN_ASSERTION_STATUS_PASSED" in fp.read()


def test_run_return(client):
    @task()
    def failed() -> bool:
        return False

    @task()
    def passed() -> bool:
        return True

    failed.run()
    with open(client.directory / "Failed-run_id_Run_1.run.json") as fp:
        assert '"booleanResult": false' in fp.read()

    passed.run()
    with open(client.directory / "Passed-run_id_Run_1.run.json") as fp:
        assert '"booleanResult": true' in fp.read()


def test_load_cached_run_float(client, duck):
    call_count = 0

    @task()
    def my_task(llm, x) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    # First run, should execute and create a cache file.
    run1 = my_task.run(duck, x=5, _id="cached_run")
    assert not run1.cached
    assert run1.result == 10
    assert call_count == 1
    assert (
        client.directory / "My_Task-run_param_id_cached_run_Duck.run.json"
    ).is_file()

    # Second run, with caching enabled. Should load from cache.
    client.use_cache = True
    run2 = my_task.run(duck, x=5, _id="cached_run")
    assert run2.cached
    assert run2.result == 10
    assert call_count == 1  # Should not have been called again.


def test_load_cached_run_dict(client, duck):
    call_count = 0

    @task()
    def my_dict_task(llm, x) -> dict:
        nonlocal call_count
        call_count += 1
        return {"score": x * 2}

    # First run, should execute and create a cache file.
    run1 = my_dict_task.run(duck, x=5, _id="cached_run_dict")
    assert not run1.cached
    assert run1.result == {"score": 10}
    assert call_count == 1
    assert (
        client.directory / "My_Dict_Task-run_param_id_cached_run_dict_Duck.run.json"
    ).is_file()

    # Second run, with caching enabled. Should load from cache.
    with client.enable_cache():
        run2 = my_dict_task.run(duck, x=5, _id="cached_run_dict")
        assert run2.cached
        assert run2.result == {"score": 10}
        assert call_count == 1


def test_load_cached_run_float_zero(client, duck):
    """proto3 JSON drops default-valued fields, so a float result of 0.0
    round-trips to `numericResult: {}` on disk. load_run_result must treat
    a missing `value` as the default 0.0 instead of raising ValueError.

    Regression test for warning seen in real eval runs:
        "Reading cached run failed: Could not determine result from cached
         run file: ... No known result key found in 'results' field."
    """
    call_count = 0

    @task()
    def zero_task(llm) -> float:
        nonlocal call_count
        call_count += 1
        return 0.0

    run1 = zero_task.run(duck, _id="zero")
    assert not run1.cached
    assert run1.result == 0.0
    assert call_count == 1

    client.use_cache = True
    run2 = zero_task.run(duck, _id="zero")
    assert run2.cached
    assert run2.result == 0.0
    assert call_count == 1  # cache hit, not re-executed


def test_load_cached_run_metric_with_zero_value_nonzero_ci(client, duck):
    """A (0.0, ci) tuple has its `value` dropped by proto3 JSON, leaving
    `numericResult: {"confidenceInterval": ci}`. The load path must still
    reconstruct a tuple, not fall through to ValueError.
    """
    call_count = 0

    @task()
    def zero_value_ci(llm) -> tuple[float, float]:
        nonlocal call_count
        call_count += 1
        return (0.0, 0.05)

    run1 = zero_value_ci.run(duck, _id="zero_v_ci")
    assert not run1.cached
    assert run1.result == pytest.approx((0.0, 0.05))
    assert call_count == 1

    client.use_cache = True
    run2 = zero_value_ci.run(duck, _id="zero_v_ci")
    assert run2.cached
    assert run2.result == pytest.approx((0.0, 0.05))
    assert call_count == 1


def test_load_cached_run_metric_with_ci(client, duck):
    call_count = 0

    @task()
    def my_pass_count_task(llm) -> tuple[float, float]:
        nonlocal call_count
        call_count += 1
        return (0.7, 0.5)

    # First run, should execute and create a cache file.
    run1 = my_pass_count_task.run(duck, _id="cached_run_mc")
    assert not run1.cached
    assert run1.result == pytest.approx((0.7, 0.5))
    assert call_count == 1
    assert (
        client.directory / "My_Pass_Count_Task-run_param_id_cached_run_mc_Duck.run.json"
    ).is_file()

    # Second run, with caching enabled. Should load from cache.
    client.use_cache = True
    run2 = my_pass_count_task.run(duck, _id="cached_run_mc")
    assert run2.cached
    assert run2.result == pytest.approx((0.7, 0.5))
    assert call_count == 1


def test_load_cached_run_pass_count(client, duck):
    call_count = 0

    @task()
    def my_pass_count_task(llm, x: int) -> tuple[int, int]:
        nonlocal call_count
        call_count += 1
        return (x, 10)

    # First run, should execute and create a cache file.
    run1 = my_pass_count_task.run(duck, x=7, _id="cached_run_pc")
    assert not run1.cached
    assert run1.result == (7, 10)
    assert call_count == 1
    assert (
        client.directory / "My_Pass_Count_Task-run_param_id_cached_run_pc_Duck.run.json"
    ).is_file()

    # Second run, with caching enabled. Should load from cache.
    client.use_cache = True
    run2 = my_pass_count_task.run(duck, x=7, _id="cached_run_pc")
    assert run2.cached
    assert run2.result == (7, 10)
    assert call_count == 1


def test_load_failed_cached_run_reruns(client, monkeypatch, duck):
    monkeypatch.setattr(config, "continue_with_exceptions", True)
    call_count = 0

    @task()
    def fallible_task(llm, x: int) -> bool:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("Intentional failure on first run")
        return True

    # First run: This should fail and create a failed run file.
    run1 = fallible_task.run(duck, x=10, _id="failed_run")
    assert not run1.passed
    assert run1.status == utils.Status.FAILED
    assert call_count == 1
    assert (
        client.directory / "Fallible_Task-run_param_id_failed_run_Duck.run.json"
    ).is_file()

    # Second run: With caching enabled, it should detect the failed state
    # in the cached file and re-run the task instead of loading from it.
    client.use_cache = True
    run2 = fallible_task.run(duck, x=10, _id="failed_run")
    assert not run2.cached  # Should not be loaded from cache.
    assert run2.passed  # Should succeed on the second attempt.
    assert call_count == 2  # Should have been called again.


def test_evaluate_retries_only_failed_samples_with_cache(client, monkeypatch, duck):
    """End-to-end demonstration of the production retry pattern.

    With `enable_cache()` + `on_failure="continue"` + `max_attempts > 1`,
    the retry attempt skips successful samples (cache hit on COMPLETED file)
    and re-runs only the failed ones (no skip on ERRORED file). Results
    from all attempts are merged into one Runs.
    """
    monkeypatch.setattr(config, "continue_with_exceptions", True)
    import pandas as pd

    call_counts: dict[str, int] = {}

    @task()
    def per_sample_task(llm, sample_id: str) -> bool:
        call_counts[sample_id] = call_counts.get(sample_id, 0) + 1
        # Sample "b" fails on its first call, succeeds on retry.
        # Samples "a" and "c" always succeed.
        if sample_id == "b" and call_counts[sample_id] == 1:
            raise ValueError(f"transient failure for {sample_id}")
        return True

    df = pd.DataFrame({"sample_id": ["a", "b", "c"]})

    with client.enable_cache():
        results = per_sample_task.evaluate(
            llm=[duck],
            evaluation_data=df,
            on_failure="continue",
            max_attempts=2,
        )

    # All three samples present in the merged result (positions 0, 1, 2),
    # all successful after the retry.
    assert len(results) == 3
    assert len(results.completed_runs) == 3
    assert len(results.errored_runs) == 0

    # The whole point: "a" and "c" ran exactly once (cache hit on retry),
    # "b" ran twice (failed first, retried successfully).
    assert call_counts["a"] == 1
    assert call_counts["c"] == 1
    assert call_counts["b"] == 2


def test_retry_merge_preserves_position_under_parallelism(client, monkeypatch, duck):
    """Position-based retry merge stays correct when n_jobs > 1.

    The merge in `Task._evaluate()` keys results by enumerate-index of each
    attempt's `Runs`. That only works if joblib returns results in input
    order even with parallelism. Heterogeneous failure pattern + n_jobs=4
    confirms each retried success lands at the original evaluation_data
    row's position.
    """
    import threading

    import pandas as pd

    monkeypatch.setattr(config, "continue_with_exceptions", True)

    lock = threading.Lock()
    call_counts: dict[str, int] = {}

    @task()
    def per_sample_task(llm, sample_id: str) -> bool:
        with lock:
            call_counts[sample_id] = call_counts.get(sample_id, 0) + 1
            n = call_counts[sample_id]
        # Samples at non-adjacent positions fail on attempt 1 to ensure
        # the retry attempt has a different parallelism pattern (fewer
        # samples → different worker scheduling) from attempt 1.
        if sample_id in {"s1", "s4", "s7"} and n == 1:
            raise ValueError(f"transient for {sample_id}")
        return True

    sample_ids = [f"s{i}" for i in range(10)]
    df = pd.DataFrame({"sample_id": sample_ids})

    with client.enable_cache():
        results = per_sample_task.evaluate(
            llm=[duck],
            evaluation_data=df,
            n_jobs=4,
            on_failure="continue",
            max_attempts=2,
        )

    assert len(results) == 10
    assert len(results.completed_runs) == 10

    # The critical assertion: each merged result is at the position
    # matching its evaluation_data row.
    for position, expected_id in enumerate(sample_ids):
        run = results[position]
        assert run.params["sample_id"] == expected_id, (
            f"Position {position}: expected sample_id={expected_id!r}, "
            f"got {run.params['sample_id']!r} (merge corrupted ordering)"
        )
        # And the user sees the retry's success, not a stale FAILED status.
        assert run.status == utils.Status.SUCCESS

    # Sanity: only the three flaky samples were retried.
    assert call_counts["s1"] == 2
    assert call_counts["s4"] == 2
    assert call_counts["s7"] == 2
    for i in (0, 2, 3, 5, 6, 8, 9):
        assert call_counts[f"s{i}"] == 1


def test_recommended_pattern_end_to_end_with_parallelism(client, monkeypatch, duck):
    """End-to-end exercise of the cookbook's recommended production pattern.

    Combines `enable_cache()` + `on_failure="continue"` + `max_attempts>1`
    + `n_jobs>1`, and aggregates via `completed_runs.as_dataframe()` per
    the documented usage. Sized to look like a miniature 'large eval'.
    """
    import threading

    import pandas as pd

    monkeypatch.setattr(config, "continue_with_exceptions", True)

    lock = threading.Lock()
    attempts: dict[int, int] = {}

    @task()
    def score_sample(llm, sample_id: int) -> bool:
        with lock:
            attempts[sample_id] = attempts.get(sample_id, 0) + 1
            n = attempts[sample_id]
        # Every 5th sample fails on attempt 1 (transient).
        if sample_id % 5 == 0 and n == 1:
            raise ValueError("transient")
        return sample_id % 2 == 0

    df = pd.DataFrame({"sample_id": list(range(20))})

    with client.enable_cache():
        results = score_sample.evaluate(
            llm=[duck],
            evaluation_data=df,
            n_jobs=4,
            on_failure="continue",
            max_attempts=3,
            retry_delay=0,
        )

    assert len(results) == 20
    assert len(results.completed_runs) == 20
    assert len(results.errored_runs) == 0

    # Aggregate using the documented pattern.
    completed_df = results.completed_runs.as_dataframe()
    accuracy = completed_df["result"].mean()
    assert accuracy == pytest.approx(0.5)  # half of 0..19 are even

    # Cache + selective retry: failed samples ran exactly twice; the rest once.
    for sample_id in range(20):
        expected = 2 if sample_id % 5 == 0 else 1
        assert attempts[sample_id] == expected, (
            f"sample_id={sample_id}: expected {expected} calls, got {attempts[sample_id]}"
        )


def test_cache_id_uses_model_over_name(client):
    """cache_id should prefer the `model` attribute over `name` when present."""
    from tests.mocks import MockedChat

    llm = MockedChat.from_contents(["quack"], name="My Custom Name", cycle=True)
    llm.model = "vendor/actual-model-slug"

    call_count = 0

    @task()
    def model_task(llm) -> bool:
        nonlocal call_count
        call_count += 1
        return True

    # First run: file should be named with the model slug, not the display name.
    run1 = model_task.run(llm, _id="test123")
    assert not run1.cached
    assert call_count == 1

    expected_file = (
        client.directory
        / "Model_Task-run_param_id_test123_vendor_actual-model-slug.run.json"
    )
    assert expected_file.is_file(), (
        f"Expected cache file using model slug, got: {list(client.directory.iterdir())}"
    )

    # Verify the name-based file does NOT exist.
    wrong_file = (
        client.directory / "Model_Task-run_param_id_test123_My_Custom_Name.run.json"
    )
    assert not wrong_file.is_file()

    # Second run with cache: should load from the model-slug-based file.
    client.use_cache = True
    run2 = model_task.run(llm, _id="test123")
    assert run2.cached
    assert run2.result is True
    assert call_count == 1  # Should not have been called again.
