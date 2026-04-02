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
    task,
    utils,
)


@pytest.fixture
def client(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        kaggle_client = kaggle.KaggleClient(directory=temp_dir)
        config.execution_mode = ExecutionMode.RUN

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
