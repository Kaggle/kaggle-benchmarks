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
from google.protobuf import json_format
from pydantic import BaseModel

from kaggle_benchmarks import actors, assertions, chats, config, user
from kaggle_benchmarks.actors.llms import LLMResponse
from kaggle_benchmarks.kaggle import serialization
from kaggle_benchmarks.tasks import task


@task(name="test", version=2, description="Test")
def boolean(x) -> bool:
    return bool(x)


@task()
def square(x) -> float:
    return x**2


@task()
def combined():
    result1 = boolean.run(True).result
    assertions.assert_true(result1)
    result2 = square.run(2.0).result
    assertions.assert_equal(result2, 4.0)


@task()
def failing_task():
    """A task that is destined to fail."""
    assertions.assert_equal(1, 2, "one is not two")


@task()
def task_with_no_description():
    pass


@task()
def task_with_docstring():
    """This is a docstring."""
    pass


class MockScoringLLM(actors.LLMChat):
    def __init__(self, score_to_return: int, name="MockScoringLLM"):
        super().__init__(name=name)
        self.score_to_return = score_to_return

    def invoke(self, messages, **kwargs):
        return LLMResponse(content=f'{{"score": {self.score_to_return}}}')


class BeautyScore(BaseModel):
    score: int


@task()
def scoring_task(score_to_return: int, raises_exception: bool = False) -> int:
    """A task that returns a score and has an assertion."""
    llm = MockScoringLLM(score_to_return)
    eval_response = llm.prompt("Rate this", schema=BeautyScore)
    score = eval_response.score
    assertions.assert_true(score >= 4, f"Score {score} is less than 4.")

    if raises_exception:
        raise ValueError("Something went wrong")
    return score


class MockLLM(actors.LLMChat):
    def __init__(self, name="MockLLM"):
        super().__init__(name=name)

    def invoke(self, messages, **kwargs):
        last_user_message = next(
            (m.content for m in reversed(messages) if m.sender.role == "user"),
            "No user message.",
        )
        return LLMResponse(content=f"I heard you say: {last_user_message}")


@task()
def chatty_task():
    """A task with a conversation."""
    llm = MockLLM()
    user.send("Hello")
    llm.respond()
    user.send("How are you?")
    llm.respond()
    # Add an assertion to test assertion mapping to requests
    assertions.assert_true(True)


@task()
def task_with_subchat():
    """A task with a sub-chat."""
    llm = MockLLM()
    user.send("Outer hello")
    llm.respond()

    with chats.new("subchat"):
        user.send("Inner hello")
        llm.respond()
        assertions.assert_equal(1, 1)  # inner assertion

    user.send("Outer goodbye")
    llm.respond()
    assertions.assert_equal(2, 2)  # outer assertion
    pass


def test_task_serialization():
    message = serialization.dump_task(boolean)
    assert isinstance(message, serialization.types.BenchmarkTaskVersion)
    assert message.name == "test"

    result = json_format.MessageToDict(message)
    assert result["name"] == "test"
    assert result["versionNumber"] == 2
    assert result["description"] == "Test"


def test_subtasks_serialization():
    message = serialization.dump_task(combined)
    assert isinstance(message, serialization.types.BenchmarkTaskVersion)
    result = json_format.MessageToDict(message)
    assert result["subtaskFileNames"] == ["Square.task.json", "test.task.json"]


def test_boolean_run_serialization():
    run = boolean.run(x=True)

    message = serialization.dump_run(run)
    assert isinstance(message, serialization.types.BenchmarkTaskRun)
    assert message.results[0].boolean_result

    result = json_format.MessageToDict(message)
    assert result["results"][0]["booleanResult"]


def test_numeric_run_serialization():
    run = square.run(x=1.0)
    message = serialization.dump_run(run)
    assert isinstance(message, serialization.types.BenchmarkTaskRun)
    assert message.results[0].numeric_result.value == 1.0

    result = json_format.MessageToDict(message)
    assert result["results"][0]["numericResult"]["value"] == 1.0


@pytest.mark.parametrize(
    "task_obj, expected_description",
    [
        (task_with_docstring, "This is a docstring."),
        (task_with_no_description, None),
    ],
)
def test_task_description_serialization(task_obj, expected_description):
    message = serialization.dump_task(task_obj)
    result = json_format.MessageToDict(message)
    assert result.get("description") == expected_description


def test_failed_run_serialization():
    run = failing_task.run()
    message = serialization.dump_run(run)
    assert isinstance(message, serialization.types.BenchmarkTaskRun)
    # A failed assertion makes the run.passed False, which should be reflected
    # in boolean_result for PassFail result types.
    assert not message.results[0].boolean_result

    result = json_format.MessageToDict(message)
    assert not result["results"][0]["booleanResult"]
    assert "assertions" in result
    assert len(result["assertions"]) == 1
    assertion = result["assertions"][0]
    assert assertion["expectation"] == "one is not two"
    assert assertion["status"] == "BENCHMARK_TASK_RUN_ASSERTION_STATUS_FAILED"
    assert 'assertions.assert_equal(1, 2, "one is not two")' in assertion["definition"]


def test_combined_run_serialization():
    run = combined.run()
    message = serialization.dump_run(run)
    assert isinstance(message, serialization.types.BenchmarkTaskRun)
    result = json_format.MessageToDict(message)

    assert "subruns" in result
    assert len(result["subruns"]) == 2
    assert result["subruns"][0]["taskVersion"]["name"] == "test"
    assert result["subruns"][1]["taskVersion"]["name"] == "Square"

    # Also check assertions from the combined task
    assert "assertions" in result
    assert len(result["assertions"]) == 2
    assert (
        result["assertions"][0]["status"]
        == "BENCHMARK_TASK_RUN_ASSERTION_STATUS_PASSED"
    )
    assert "assertions.assert_true(result1)" in result["assertions"][0]["definition"]
    assert (
        result["assertions"][1]["status"]
        == "BENCHMARK_TASK_RUN_ASSERTION_STATUS_PASSED"
    )
    assert (
        "assertions.assert_equal(result2, 4.0)" in result["assertions"][1]["definition"]
    )


def test_chat_serialization():
    run = chatty_task.run()
    message = serialization.dump_run(run)
    result = json_format.MessageToDict(message)

    assert "conversations" in result
    assert len(result["conversations"]) == 1
    conversation = result["conversations"][0]
    assert "requests" in conversation
    assert len(conversation["requests"]) == 2  # two llm.respond() calls

    # Check first request
    request1 = conversation["requests"][0]
    assert len(request1["contents"]) == 2  # user, assistant
    assert request1["contents"][0]["role"] == "CONTENT_ROLE_USER"
    assert request1["contents"][0]["parts"][0]["text"] == "Hello"
    assert request1["contents"][0]["senderName"] == "User"
    assert request1["contents"][1]["role"] == "CONTENT_ROLE_ASSISTANT"
    assert "I heard you say: Hello" in request1["contents"][1]["parts"][0]["text"]
    assert request1["contents"][1]["senderName"] == "MockLLM"

    # Check second request
    request2 = conversation["requests"][1]
    assert len(request2["contents"]) == 2  # user, assistant
    assert request2["contents"][0]["role"] == "CONTENT_ROLE_USER"
    assert request2["contents"][0]["parts"][0]["text"] == "How are you?"
    assert request2["contents"][0]["senderName"] == "User"
    assert request2["contents"][1]["role"] == "CONTENT_ROLE_ASSISTANT"
    assert (
        "I heard you say: How are you?" in request2["contents"][1]["parts"][0]["text"]
    )
    assert request2["contents"][1]["senderName"] == "MockLLM"

    # Check assertion mapping
    assert "assertions" in result
    assert len(result["assertions"]) == 1
    assertion = result["assertions"][0]
    assert "conversationRequestIds" in assertion
    assert len(assertion["conversationRequestIds"]) == 1
    convo_ref = assertion["conversationRequestIds"][0]
    assert convo_ref["conversationId"] == run.chat.id
    # The assertion happens after the second request, so it should be linked to it.
    assert convo_ref["requestId"] == conversation["requests"][1]["id"]


def test_subchat_serialization():
    run = task_with_subchat.run()
    message = serialization.dump_run(run)
    result = json_format.MessageToDict(message)

    assert "conversations" in result
    # The subchat is nested inside the main chat's history, but serialization flattens it.
    assert len(result["conversations"]) == 2  # outer chat and subchat

    outer_chat_data = result["conversations"][0]
    subchat_data = result["conversations"][1]

    # Check outer chat
    assert outer_chat_data["id"] == run.chat.id
    assert len(outer_chat_data["requests"]) == 2  # two responses in outer chat
    assert (
        outer_chat_data["requests"][0]["contents"][0]["parts"][0]["text"]
        == "Outer hello"
    )
    assert (
        outer_chat_data["requests"][1]["contents"][0]["parts"][0]["text"]
        == "Outer goodbye"
    )

    # Check subchat
    # The subchat is the 3rd item in history (user, assistant, subchat)
    assert subchat_data["id"] == run.chat.history[2].id
    assert len(subchat_data["requests"]) == 1
    assert (
        subchat_data["requests"][0]["contents"][0]["parts"][0]["text"] == "Inner hello"
    )

    # Check assertion mapping
    assert "assertions" in result
    assert len(result["assertions"]) == 2

    inner_assertion = next(
        a for a in result["assertions"] if "assert_equal(1, 1)" in a["definition"]
    )
    outer_assertion = next(
        a for a in result["assertions"] if "assert_equal(2, 2)" in a["definition"]
    )

    # Inner assertion should be linked to subchat request
    assert len(inner_assertion["conversationRequestIds"]) == 1
    inner_convo_ref = inner_assertion["conversationRequestIds"][0]
    assert inner_convo_ref["conversationId"] == subchat_data["id"]
    assert inner_convo_ref["requestId"] == subchat_data["requests"][0]["id"]

    # Outer assertion should be linked to the second request of the outer chat
    assert len(outer_assertion["conversationRequestIds"]) == 1
    outer_convo_ref = outer_assertion["conversationRequestIds"][0]
    assert outer_convo_ref["conversationId"] == outer_chat_data["id"]
    assert outer_convo_ref["requestId"] == outer_chat_data["requests"][1]["id"]


def test_passing_scoring_task_serialization():
    run = scoring_task.run(score_to_return=5)
    message = serialization.dump_run(run)
    assert isinstance(message, serialization.types.BenchmarkTaskRun)

    # Check result
    assert message.results[0].numeric_result.value == 5.0

    # Check assertion
    assert len(message.assertions) == 1
    assert (
        message.assertions[0].status
        == serialization.types.BENCHMARK_TASK_RUN_ASSERTION_STATUS_PASSED
    )

    # Check JSON dict
    result = json_format.MessageToDict(message)
    assert result["results"][0]["numericResult"]["value"] == 5.0
    assert "assertions" in result
    assert len(result["assertions"]) == 1
    assert (
        result["assertions"][0]["status"]
        == "BENCHMARK_TASK_RUN_ASSERTION_STATUS_PASSED"
    )


def test_failing_scoring_task_serialization():
    run = scoring_task.run(score_to_return=3)
    message = serialization.dump_run(run)
    assert isinstance(message, serialization.types.BenchmarkTaskRun)

    # Check result
    assert message.results[0].numeric_result.value == 3.0

    # Check assertion
    assert len(message.assertions) == 1
    assert (
        message.assertions[0].status
        == serialization.types.BENCHMARK_TASK_RUN_ASSERTION_STATUS_FAILED
    )
    assert "Score 3 is less than 4" in message.assertions[0].expectation

    # Check JSON dict
    result = json_format.MessageToDict(message)
    assert result["results"][0]["numericResult"]["value"] == 3.0
    assert "assertions" in result
    assert len(result["assertions"]) == 1
    assert (
        result["assertions"][0]["status"]
        == "BENCHMARK_TASK_RUN_ASSERTION_STATUS_FAILED"
    )
    assert "Score 3 is less than 4" in result["assertions"][0]["expectation"]


def test_exception_run_serialization(monkeypatch):
    monkeypatch.setattr(config, "continue_with_exceptions", True)
    run = scoring_task.run(score_to_return=5, raises_exception=True)
    message = serialization.dump_run(run)
    assert isinstance(message, serialization.types.BenchmarkTaskRun)

    # Check result
    assert message.results[0].boolean_result is False

    # Check assertion
    assert len(message.assertions) == 1
    assert (
        message.assertions[0].status
        == serialization.types.BENCHMARK_TASK_RUN_ASSERTION_STATUS_PASSED
    )
    assert message.state == serialization.types.BENCHMARK_TASK_RUN_STATE_ERRORED
    assert message.error_message == "ValueError: Something went wrong"

    # Check JSON dict
    result = json_format.MessageToDict(message)
    assert result["results"][0]["booleanResult"] is False
    assert "assertions" in result
    assert len(result["assertions"]) == 1
    assert (
        result["assertions"][0]["status"]
        == "BENCHMARK_TASK_RUN_ASSERTION_STATUS_PASSED"
    )
    assert result["state"] == "BENCHMARK_TASK_RUN_STATE_ERRORED"
    assert result["errorMessage"] == "ValueError: Something went wrong"
