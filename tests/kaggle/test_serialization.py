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

import datetime
import functools
import json
import logging
import tempfile
from pathlib import Path
from unittest import mock

from google.protobuf import json_format

from kaggle_benchmarks import tasks
from kaggle_benchmarks.kaggle import serialization

numeric_run_file1 = "testing-numeric-Run_1.run.json"
numeric_run_file2 = "testing-numeric-Run_2.run.json"
boolean_run_file1 = "testing-boolean-Run_1.run.json"
boolean_run_file2 = "testing-boolean-Run_2.run.json"


def test_source_code():
    def f(a):
        return a

    assert "return a" in serialization._get_source_code(f)


def test_source_code_partial():
    def f(a):
        return a

    assert "return a" in serialization._get_source_code(functools.partial(f, a=1))


def test_source_code_lambda(caplog):
    with caplog.at_level(logging.WARNING, logger=serialization.logger.name):
        # In some environments (like a REPL), getsource can't find the source
        # for a lambda. We just want to ensure it doesn't crash and logs a warning.
        source = serialization._get_source_code(lambda x: x)
        # inspect.getsource on a lambda can either work or fail depending on environment.
        # If it fails, we check for the warning.
        if not source:
            assert "Could not get source code for <lambda>" in caplog.text
        else:
            assert "lambda x: x" in source


def test_get_source_code_error(caplog):
    def my_func():
        pass

    with mock.patch("inspect.getsource", side_effect=OSError("file not found")):
        with caplog.at_level(logging.WARNING, logger=serialization.logger.name):
            source = serialization._get_source_code(my_func)
            assert source == ""
            assert "Could not get source code for my_func" in caplog.text


def test_generate_task_filename():
    assert serialization.generate_task_filename("My Task") == "My_Task.task.json"
    assert serialization.generate_task_filename("a/b") == "a_b.task.json"


def test_generate_run_filename():
    assert (
        serialization.generate_run_filename("My Task", "Run 1")
        == "My_Task-Run_1.run.json"
    )
    assert serialization.generate_run_filename("a/b", "c#d") == "a_b-cd.run.json"


def test_format_timestamp():
    # Test with timezone-aware datetime in UTC
    utc_dt = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    assert serialization._format_timestamp(utc_dt) == "2023-01-01T12:00:00Z"

    # Test with None
    assert serialization._format_timestamp(None) is None

    # Test with naive datetime
    naive_dt = datetime.datetime(2023, 1, 1, 12, 0, 0)
    assert serialization._format_timestamp(naive_dt) == "2023-01-01T12:00:00"

    # Test with datetime that already has Z (from isoformat)
    dt_with_z = datetime.datetime.fromisoformat("2023-01-01T12:00:00Z")
    assert serialization._format_timestamp(dt_with_z) == "2023-01-01T12:00:00Z"


def _assert_time_strs_equal(a: str, b: str):
    """Asserts that two time strings are equal up to the second, ignoring microseconds and timezones."""
    at = datetime.datetime.fromisoformat(a)
    bt = datetime.datetime.fromisoformat(b)
    a_truncated = at.strftime("%Y-%m-%dT%H:%M:%S")
    b_truncated = bt.strftime("%Y-%m-%dT%H:%M:%S")
    assert a_truncated == b_truncated


def test_merge_numeric_runfiles():
    @tasks.task()
    def square(x) -> float:
        return x**2

    def calculate_mean_score(run_results: list[dict]) -> float:
        print(run_results[0])
        scores = [result["numericResult"]["value"] for result in run_results]
        mean_score = sum(scores) / len(scores)
        return mean_score

    def calculate_mean_score_ci(run_results: list[dict]) -> tuple:
        print(run_results[0])
        scores = [result["numericResult"]["value"] for result in run_results]
        mean_score = sum(scores) / len(scores)
        return mean_score, 1.5  # Return a default CI value of 1.5

    with tempfile.TemporaryDirectory() as tmp_dir:
        run1 = square.run(3)
        message1 = serialization.dump_run(run1)
        run2 = square.run(4)
        message2 = serialization.dump_run(run2)

        # dump_run leaves the model_version blank, so we fill it in for the sake of this test
        if message1.model_version.slug == "":
            message1.model_version.slug = message2.model_version.slug = (
                "anthropic/claude-3-5-sonnet-v2"
            )

        with open(f"{tmp_dir}/{numeric_run_file1}", "w") as f:
            f.write(json_format.MessageToJson(message1))
        with open(f"{tmp_dir}/{numeric_run_file2}", "w") as f:
            f.write(json_format.MessageToJson(message2))

        runfiles = [f"{tmp_dir}/{numeric_run_file1}", f"{tmp_dir}/{numeric_run_file2}"]

        target_start_time = message1.start_time.ToJsonString()
        value1 = message1.results[0].numeric_result.value

        target_end_time = message2.end_time.ToJsonString()
        value2 = message2.results[0].numeric_result.value

        outputFileName = serialization.merge_results_from_runfiles(
            runfiles, calculate_mean_score, output_directory=tmp_dir
        )

        run_data = None
        with open(outputFileName, "r") as f:
            run_data = json.load(f)
        if run_data is None:
            raise FileNotFoundError(f"Error loading file {outputFileName}")

        # Compare timestamps by parsing them to avoid precision issues
        _assert_time_strs_equal(run_data["startTime"], target_start_time)
        _assert_time_strs_equal(run_data["endTime"], target_end_time)
        assert run_data["results"][0]["numericResult"]["value"] == (value1 + value2) / 2

        # Assert that runfiles have not been deleted
        assert Path(f"{tmp_dir}/{numeric_run_file1}").exists()
        assert Path(f"{tmp_dir}/{numeric_run_file2}").exists()

        # Same test as above, but with confidence interval (ci)
        outputFileName = serialization.merge_results_from_runfiles(
            runfiles,
            calculate_mean_score_ci,
            output_run_id="Run aggregated with ci",
            output_directory=tmp_dir,
            delete_run_files=True,
        )

        run_data = None
        with open(outputFileName, "r") as f:
            run_data = json.load(f)
        if run_data is None:
            raise FileNotFoundError(f"Error loading file {outputFileName}")

        _assert_time_strs_equal(run_data["startTime"], target_start_time)
        _assert_time_strs_equal(run_data["endTime"], target_end_time)
        assert run_data["results"][0]["numericResult"]["value"] == (value1 + value2) / 2
        assert run_data["results"][0]["numericResult"]["confidenceInterval"] == 1.5

        # Assert that runfiles have been deleted
        assert not Path(f"{tmp_dir}/{numeric_run_file1}").exists()
        assert not Path(f"{tmp_dir}/{numeric_run_file2}").exists()


def test_merge_boolean_runfiles():
    @tasks.task()
    def iseven(x) -> bool:
        return x % 2 == 0

    def calculate_all_true(run_results: list[dict]) -> float:
        print(run_results[0])
        scores = [result["booleanResult"] for result in run_results]
        all_true = all(score for score in scores)
        return all_true

    with tempfile.TemporaryDirectory() as tmp_dir:
        run1 = iseven.run(3)
        message1 = serialization.dump_run(run1)
        run2 = iseven.run(4)
        message2 = serialization.dump_run(run2)

        # dump_run leaves the model_version blank, so we fill it in for the sake of this test
        if message1.model_version.slug == "":
            message1.model_version.slug = message2.model_version.slug = (
                "anthropic/claude-3-5-sonnet-v2"
            )

        with open(f"{tmp_dir}/{boolean_run_file1}", "w") as f:
            f.write(json_format.MessageToJson(message1))
        with open(f"{tmp_dir}/{boolean_run_file2}", "w") as f:
            f.write(json_format.MessageToJson(message2))

        runfiles = [f"{tmp_dir}/{boolean_run_file1}", f"{tmp_dir}/{boolean_run_file2}"]

        target_start_time = message1.start_time.ToJsonString()
        value1 = message1.results[0].boolean_result

        target_end_time = message2.end_time.ToJsonString()
        value2 = message2.results[0].boolean_result

        outputFileName = serialization.merge_results_from_runfiles(
            runfiles,
            calculate_all_true,
            output_directory=tmp_dir,
            delete_run_files=True,
        )
        run_data = None
        with open(outputFileName, "r") as f:
            run_data = json.load(f)

        if run_data is None:
            raise FileNotFoundError(f"Error loading file {outputFileName}")

        _assert_time_strs_equal(run_data["startTime"], target_start_time)
        _assert_time_strs_equal(run_data["endTime"], target_end_time)
        assert run_data["results"][0]["booleanResult"] == (value1 and value2)

        # Assert that runfiles have been deleted
        assert not Path(f"{tmp_dir}/{boolean_run_file1}").exists()
        assert not Path(f"{tmp_dir}/{boolean_run_file2}").exists()


# --- Tests for _find_subtask_names ---


@tasks.task()
def subtask_for_test():
    pass


@tasks.task(name="custom_name_task")
def task_with_custom_name():
    pass


@tasks.task()
def main_task_with_subtasks():
    """A task that calls other tasks."""
    subtask_for_test.run()
    task_with_custom_name.run()


def test_find_subtask_names():
    """Tests that _find_subtask_names correctly identifies subtasks."""
    names = serialization._find_subtask_names(main_task_with_subtasks)
    # Names are capitalized by default, and then sorted.
    assert names == ["Subtask For Test", "custom_name_task"]


@tasks.task()
def task_with_no_subtasks():
    """A task that does not call other tasks."""
    assert 1 == 1


def test_find_subtask_names_with_no_subtasks():
    """Tests that _find_subtask_names returns an empty list for a task with no subtasks."""
    names = serialization._find_subtask_names(task_with_no_subtasks)
    assert names == []


@tasks.task()
def task_with_non_task_run_call():
    """A task that calls a .run() method on a non-Task object."""

    class FakeTask:
        def run(self):
            return "not a real task run"

    fake = FakeTask()
    fake.run()
    subtask_for_test.run()


def test_find_subtask_names_with_non_task_run():
    """Tests that _find_subtask_names ignores .run() calls on non-Task objects."""
    names = serialization._find_subtask_names(task_with_non_task_run_call)
    assert names == ["Subtask For Test"]
