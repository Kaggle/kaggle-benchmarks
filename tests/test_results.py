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

import logging

import pytest

from kaggle_benchmarks import assertions, results, tasks


def test_pass():
    @tasks.task()
    def my_task():
        assert True
        assertions.assert_true(True)

    assert my_task.result_type is results.PassFail

    run = my_task.run()
    assert run.passed
    assert run.result is None
    assert len(run.assertion_results) == 1


@pytest.mark.parametrize("use_buildin", [True, False])
def test_fail(use_buildin):
    @tasks.task()
    def my_task():
        if use_buildin:
            assert False
        else:
            assertions.assert_true(False)

    assert my_task.result_type is results.PassFail

    run = my_task.run()
    assert not run.passed
    # todo: the result value should be the same
    assert not use_buildin or isinstance(run.result, results.Unknown)
    assert len(run.assertion_results) == 1


def test_warnings(caplog):
    @tasks.task()
    def task(x) -> float:
        return x

    with caplog.at_level(logging.WARNING, logger=tasks.logger.name):
        task.run(1.2)

    assert not caplog.text

    with caplog.at_level(logging.WARNING, logger=tasks.logger.name):
        task.run("a")

    assert "wrong" in caplog.text.lower()
    assert "float" in caplog.text


def test_dictionary_result_type():
    @tasks.task()
    def my_dict_task() -> dict:
        return {"status": "ok", "value": 123}

    assert my_dict_task.result_type is results.Dictionary

    run = my_dict_task.run()
    assert run.passed
    assert isinstance(run.result, dict)
    assert run.result["status"] == "ok"
