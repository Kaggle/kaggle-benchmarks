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


def test_request_metrics_include_cost_fields():
    """Tests that request metrics include token cost fields from _meta."""
    from kaggle_benchmarks import actors, chats, messages

    chat = chats.Chat(name="test")
    chat.append(messages.Message(sender=actors.user, content="hello"))

    response = messages.Message(sender=actors.system, content="response")
    response.sender = actors.Actor(name="assistant", role="assistant")
    response._meta["input_tokens"] = 100
    response._meta["output_tokens"] = 50
    response._meta["input_tokens_cost_nanodollars"] = 1000
    response._meta["output_tokens_cost_nanodollars"] = 2000
    chat.append(response)

    conversations, _ = serialization._prepare_conversations_data(chat)
    metrics = conversations[0]["requests"][0]["metrics"]

    assert metrics["input_tokens"] == 100
    assert metrics["output_tokens"] == 50
    assert metrics["input_tokens_cost_nanodollars"] == 1000
    assert metrics["output_tokens_cost_nanodollars"] == 2000

    # Verify conversation-level aggregation includes cost fields
    conv_metrics = conversations[0]["metrics"]
    assert conv_metrics["input_tokens_cost_nanodollars"] == 1000
    assert conv_metrics["output_tokens_cost_nanodollars"] == 2000


def test_conversation_metrics_cost_none_when_missing():
    """Tests that conversation metrics keep cost as None when not provided."""
    from kaggle_benchmarks import actors, chats, messages

    chat = chats.Chat(name="test")
    chat.append(messages.Message(sender=actors.user, content="hello"))

    response = messages.Message(sender=actors.system, content="response")
    response.sender = actors.Actor(name="assistant", role="assistant")
    response._meta["input_tokens"] = 100
    response._meta["output_tokens"] = 50
    # No cost fields set
    chat.append(response)

    conversations, _ = serialization._prepare_conversations_data(chat)
    conv_metrics = conversations[0]["metrics"]

    assert conv_metrics["input_tokens"] == 100
    assert conv_metrics["output_tokens"] == 50
    assert conv_metrics["input_tokens_cost_nanodollars"] is None
    assert conv_metrics["output_tokens_cost_nanodollars"] is None


def test_int64_cost_fields_serialized_as_strings_in_json():
    """
    Documents that int64 fields (cost_nanodollars) are serialized as strings in JSON.

    Protocol Buffers serialize int64 values as strings to avoid precision loss in
    JavaScript (which uses 64-bit floats). This doesn't affect SDK dump/parse but
    FE consumers reading raw JSON should be aware of this.
    """
    from kaggle_benchmarks import actors, chats, messages
    from kaggle_benchmarks.kaggle import benchmark_types_pb2 as types

    chat = chats.Chat(name="test")
    chat.append(messages.Message(sender=actors.user, content="hello"))

    response = messages.Message(sender=actors.system, content="response")
    response.sender = actors.Actor(name="assistant", role="assistant")
    response._meta["input_tokens"] = 100
    response._meta["output_tokens"] = 50
    response._meta["input_tokens_cost_nanodollars"] = 123456789012345
    response._meta["output_tokens_cost_nanodollars"] = 987654321098765
    chat.append(response)

    conversations, _ = serialization._prepare_conversations_data(chat)

    # Create a Conversation proto and serialize to JSON
    conv_proto = json_format.ParseDict(
        conversations[0], types.Conversation(), ignore_unknown_fields=True
    )
    json_str = json_format.MessageToJson(conv_proto)
    raw_json = json.loads(json_str)

    # Verify int64 fields are serialized as strings in the raw JSON
    request_metrics = raw_json["requests"][0]["metrics"]
    assert request_metrics["inputTokensCostNanodollars"] == "123456789012345"
    assert request_metrics["outputTokensCostNanodollars"] == "987654321098765"
    assert isinstance(request_metrics["inputTokensCostNanodollars"], str)
    assert isinstance(request_metrics["outputTokensCostNanodollars"], str)

    # Conversation-level metrics also serialized as strings
    conv_metrics = raw_json["metrics"]
    assert conv_metrics["inputTokensCostNanodollars"] == "123456789012345"
    assert conv_metrics["outputTokensCostNanodollars"] == "987654321098765"


def test_int64_cost_fields_file_roundtrip():
    """
    Tests that int64 cost fields survive a file write/read cycle correctly.

    Writes a run.json file, reads it back as raw JSON (simulating FE read),
    and verifies the SDK can parse it back to proto with correct values.
    """
    from kaggle_benchmarks.kaggle import benchmark_types_pb2 as types

    # Create a BenchmarkTaskRun proto directly with large int64 cost values
    # Values > 2^53 would lose precision if stored as JS floats
    large_input_cost = 9007199254740993  # > 2^53
    large_output_cost = 9007199254740994  # > 2^53

    run_proto = types.BenchmarkTaskRun(
        py_run_id="test-cost-run",
        conversations=[
            types.Conversation(
                id="conv-1",
                requests=[
                    types.ModelRequest(
                        id="req-1",
                        metrics=types.ModelUsageMetrics(
                            input_tokens=100,
                            output_tokens=50,
                            input_tokens_cost_nanodollars=large_input_cost,
                            output_tokens_cost_nanodollars=large_output_cost,
                        ),
                    )
                ],
                metrics=types.ModelUsageMetrics(
                    input_tokens=100,
                    output_tokens=50,
                    input_tokens_cost_nanodollars=large_input_cost,
                    output_tokens_cost_nanodollars=large_output_cost,
                ),
            )
        ],
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "test_cost.run.json"

        # Write to file
        with open(filepath, "w") as f:
            f.write(json_format.MessageToJson(run_proto))

        # Read back as raw JSON (simulating FE read)
        with open(filepath, "r") as f:
            raw_json = json.load(f)

        # Verify int64 fields are strings in raw JSON
        conv = raw_json["conversations"][0]
        request_metrics = conv["requests"][0]["metrics"]
        assert isinstance(request_metrics["inputTokensCostNanodollars"], str)
        assert isinstance(request_metrics["outputTokensCostNanodollars"], str)
        assert request_metrics["inputTokensCostNanodollars"] == "9007199254740993"
        assert request_metrics["outputTokensCostNanodollars"] == "9007199254740994"

        # Conversation-level metrics also strings
        conv_metrics = conv["metrics"]
        assert isinstance(conv_metrics["inputTokensCostNanodollars"], str)
        assert conv_metrics["inputTokensCostNanodollars"] == "9007199254740993"

        # Parse back to proto (simulating SDK read)
        with open(filepath, "r") as f:
            json_str = f.read()
        parsed_proto = json_format.Parse(json_str, types.BenchmarkTaskRun())

        # Verify values are correctly restored as integers in proto
        parsed_metrics = parsed_proto.conversations[0].requests[0].metrics
        assert parsed_metrics.input_tokens_cost_nanodollars == 9007199254740993
        assert parsed_metrics.output_tokens_cost_nanodollars == 9007199254740994
