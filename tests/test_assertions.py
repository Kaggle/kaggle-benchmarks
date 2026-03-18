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

"""
pytest: disable_assert_rewrite
"""

import json
import re

import pytest

from kaggle_benchmarks import actors, assertions, tasks
from kaggle_benchmarks.actors.llms import LLMResponse


def assert_assertion_result_matches(
    actual_result: assertions.AssertionResult,
    expected_passed: bool,
    expected_expectation: str,
    expected_details_content: dict,
):
    """
    Asserts that an AssertionResult matches expected values,
    checking line_number for type and positivity rather than exact value.
    """
    assert actual_result.passed == expected_passed
    assert actual_result.expectation == expected_expectation

    assert actual_result.details is not None
    actual_line_number = actual_result.details.get("line_number")
    assert isinstance(actual_line_number, int), (
        f"Expected line_number to be an int, got {actual_line_number}"
    )
    assert actual_line_number > 0, (
        f"Expected line_number to be positive, got {actual_line_number}"
    )

    # Compare details, excluding line_number
    actual_details_content = {
        k: v for k, v in actual_result.details.items() if k != "line_number"
    }
    assert actual_details_content == expected_details_content


def test_assertions():
    def test():
        return assertions.assert_equal(1, 2, "1 should be 2")

    r = assertions.assert_in(1, [1, 2, 3])
    assert_assertion_result_matches(
        r,
        expected_passed=True,
        expected_expectation="Expected '1' in '[1, 2, 3]'",
        expected_details_content={
            "assertion_type": "assert_in",
            "source_code": "r = assertions.assert_in(1, [1, 2, 3])",
        },
    )

    r_test_fn = test()
    assert_assertion_result_matches(
        r_test_fn,
        expected_passed=False,
        expected_expectation="1 should be 2",
        expected_details_content={
            "assertion_type": "assert_equal",
            "source_code": 'return assertions.assert_equal(1, 2, "1 should be 2")',
        },
    )


def test_regex_assertions():
    # Test assert_contains_regex
    # Case 1: Simple match, should pass
    r_contains_pass = assertions.assert_contains_regex(r"\d+", "hello 123 world")
    assert_assertion_result_matches(
        r_contains_pass,
        expected_passed=True,
        expected_expectation="Expected pattern '\\d+' found in 'hello 123 world'",
        expected_details_content={
            "assertion_type": "assert_contains_regex",
            "source_code": 'r_contains_pass = assertions.assert_contains_regex(r"\\d+", "hello 123 world")',
        },
    )

    # Case 2: No match, should fail
    r_contains_fail = assertions.assert_contains_regex(r"\d+", "hello world")
    assert_assertion_result_matches(
        r_contains_fail,
        expected_passed=False,
        expected_expectation="Expected pattern '\\d+' found in 'hello world'",
        expected_details_content={
            "assertion_type": "assert_contains_regex",
            "source_code": 'r_contains_fail = assertions.assert_contains_regex(r"\\d+", "hello world")',
        },
    )

    # Case 3: Compiled pattern, should pass
    pattern = re.compile("world")
    r_contains_compiled = assertions.assert_contains_regex(pattern, "hello world")
    assert_assertion_result_matches(
        r_contains_compiled,
        expected_passed=True,
        expected_expectation="Expected pattern 're.compile('world')' found in 'hello world'",
        expected_details_content={
            "assertion_type": "assert_contains_regex",
            "source_code": 'r_contains_compiled = assertions.assert_contains_regex(pattern, "hello world")',
        },
    )

    # Test assert_not_contains_regex
    # Case 1: No match, should pass
    r_not_contains_pass = assertions.assert_not_contains_regex(r"\d+", "hello world")
    assert_assertion_result_matches(
        r_not_contains_pass,
        expected_passed=True,
        expected_expectation="Expected pattern '\\d+' not found in 'hello world'",
        expected_details_content={
            "assertion_type": "assert_not_contains_regex",
            "source_code": 'r_not_contains_pass = assertions.assert_not_contains_regex(r"\\d+", "hello world")',
        },
    )

    # Case 2: Match, should fail
    r_not_contains_fail = assertions.assert_not_contains_regex(
        r"\d+", "hello 123 world", expectation="custom message"
    )
    assert_assertion_result_matches(
        r_not_contains_fail,
        expected_passed=False,
        expected_expectation="custom message",
        expected_details_content={
            "assertion_type": "assert_not_contains_regex",
            "source_code": "r_not_contains_fail = assertions.assert_not_contains_regex(",
        },
    )

    # Case 3: Custom failure message
    r_not_contains_fail_custom = assertions.assert_not_contains_regex(
        r"\d+", "hello 123 world", expectation="Custom message"
    )
    assert_assertion_result_matches(
        r_not_contains_fail_custom,
        expected_passed=False,
        expected_expectation="Custom message",
        expected_details_content={
            "assertion_type": "assert_not_contains_regex",
            "source_code": "r_not_contains_fail_custom = assertions.assert_not_contains_regex(",
        },
    )


class Duck(actors.LLMChat):
    def invoke(self, messages, system, **kwargs):
        return LLMResponse(content="quack")


@pytest.fixture
def duck():
    yield Duck()


@assertions.assertion_handler()
def assert_duck_always_quacks(response, expectation) -> assertions.AssertionResult:
    passed = response == "quack"

    return assertions.AssertionResult(
        passed=passed,
        expectation=expectation,
    )


@tasks.task()
def a_task(llm, message: str = "hi"):
    response = llm.prompt(message).lower()
    assertions.assert_equal("honk", response)
    assertions.assert_in(response, ["quack", "honk"])
    assert_duck_always_quacks(llm.prompt(message), "Duck should only quack")


def test_a_task(duck):
    run = a_task.run(duck, message="hey")

    assert len(run.assertion_results) == 3

    assert_assertion_result_matches(
        run.assertion_results[0],
        expected_passed=False,
        expected_expectation="Expected: 'honk', Got: 'quack'",
        expected_details_content={
            "assertion_type": "assert_equal",
            "source_code": 'assertions.assert_equal("honk", response)',
        },
    )

    assert_assertion_result_matches(
        run.assertion_results[1],
        expected_passed=True,
        expected_expectation="Expected 'quack' in '['quack', 'honk']'",
        expected_details_content={
            "assertion_type": "assert_in",
            "source_code": 'assertions.assert_in(response, ["quack", "honk"])',
        },
    )

    assert_assertion_result_matches(
        run.assertion_results[2],
        expected_passed=True,
        expected_expectation="Duck should only quack",
        expected_details_content={
            "assertion_type": "assert_duck_always_quacks",
            "source_code": 'assert_duck_always_quacks(llm.prompt(message), "Duck should only quack")',
        },
    )


@assertions.assertion_handler()
def assert_duck_also_honk(response) -> assertions.AssertionResult:
    return assertions.assert_equal("honk", response, "Duck should also honk")


@tasks.task()
def a_task_with_nested_assertion(llm, message: str = "hi"):
    response = llm.prompt(message).lower()
    assert_duck_also_honk(response)


def test_a_task_with_nested_assertion(duck):
    run = a_task_with_nested_assertion.run(duck, message="hey")
    assert len(run.assertion_results) == 2

    # Inner assertion (from assert_equal inside assert_duck_also_honk)
    assert_assertion_result_matches(
        run.assertion_results[0],
        expected_passed=False,
        expected_expectation="Duck should also honk",
        expected_details_content={
            "assertion_type": "assert_equal",
            "source_code": 'return assertions.assert_equal("honk", response, "Duck should also honk")',
        },
    )

    # Outer assertion (from assert_duck_also_honk itself)
    assert_assertion_result_matches(
        run.assertion_results[1],
        expected_passed=False,
        expected_expectation="Duck should also honk",
        expected_details_content={
            "assertion_type": "assert_duck_also_honk",
            "source_code": "assert_duck_also_honk(response)",
        },
    )

    assert len(run.chat.history) == 4

    # Check payload of the first (inner) assertion result
    inner_payload_dict = json.loads(run.chat.history[-2].payload)
    assert inner_payload_dict["passed"] is False
    assert inner_payload_dict["expectation"] == "Duck should also honk"
    inner_details = inner_payload_dict["details"]
    inner_line_num = inner_details.pop("line_number")
    assert isinstance(inner_line_num, int) and inner_line_num > 0
    expected_inner_details_content = {
        "assertion_type": "assert_equal",
        "source_code": 'return assertions.assert_equal("honk", response, "Duck should also honk")',
    }
    assert inner_details == expected_inner_details_content

    # Check payload of the second (outer) assertion result
    outer_payload_dict = json.loads(run.chat.history[-1].payload)
    assert outer_payload_dict["passed"] is False
    assert outer_payload_dict["expectation"] == "Duck should also honk"
    outer_details = outer_payload_dict["details"]
    outer_line_num = outer_details.pop("line_number")
    assert isinstance(outer_line_num, int) and outer_line_num > 0
    expected_outer_details_content = {
        "assertion_type": "assert_duck_also_honk",
        "source_code": "assert_duck_also_honk(response)",
    }
    assert outer_details == expected_outer_details_content


@assertions.assertion_handler(raises_assertion_error=True)
def assert_equal_with_exception(
    actual, expected, expectation=None
) -> assertions.AssertionResult:
    passed = actual == expected

    return assertions.AssertionResult(
        passed=passed,
        expectation=expectation or "Values should be equal.",
    )


def test_assertion_raises_exception():
    with pytest.raises(AssertionError):
        assert_equal_with_exception(1, 2)


def test_custom_expectation():
    # Test with a custom expectation on a failing assertion
    r_custom_fail = assertions.assert_equal(
        1, 2, expectation="Custom expectation for failure"
    )
    assert_assertion_result_matches(
        r_custom_fail,
        expected_passed=False,
        expected_expectation="Custom expectation for failure",
        expected_details_content={
            "assertion_type": "assert_equal",
            "source_code": "r_custom_fail = assertions.assert_equal(",
        },
    )

    # Test with a custom expectation on a passing assertion
    r_custom_pass = assertions.assert_true(
        True, expectation="Custom expectation for success"
    )
    assert_assertion_result_matches(
        r_custom_pass,
        expected_passed=True,
        expected_expectation="Custom expectation for success",
        expected_details_content={
            "assertion_type": "assert_true",
            "source_code": "r_custom_pass = assertions.assert_true(",
        },
    )


def test_assert_fail():
    r = assertions.assert_fail("This is a forced failure")
    assert_assertion_result_matches(
        r,
        expected_passed=False,
        expected_expectation="This is a forced failure",
        expected_details_content={
            "assertion_type": "assert_fail",
            "source_code": 'r = assertions.assert_fail("This is a forced failure")',
        },
    )


def test_assess_response_with_judge():
    class MockJudge(actors.LLMChat):
        def __init__(self, return_value):
            super().__init__(name="MockJudge")
            self.return_value = return_value

        def prompt(self, message, schema=None, **kwargs):
            return self.return_value

    # Judge returns dict result
    report_dict = {
        "results": [
            {
                "criterion": "some expectation",
                "passed": True,
                "reason": "LGTM",
                "confidence": 10,
            }
        ]
    }
    judge_pass = MockJudge(report_dict)

    report = assertions.assess_response_with_judge(
        criteria=["some expectation"],
        response_text="some response",
        judge_llm=judge_pass,
    )

    assert len(report.results) == 1
    assert report.results[0].passed
    assert report.results[0].criterion == "some expectation"
    assert report.results[0].reason == "LGTM"
    assert report.results[0].confidence == 10

    # Judge returns AssessReport object
    report_obj = assertions.AssessReport(
        results=[
            assertions.AssessResult(
                criterion="some expectation",
                passed=True,
                reason="LGTM object",
                confidence=9,
            )
        ]
    )
    judge_obj = MockJudge(report_obj)
    report = assertions.assess_response_with_judge(
        criteria=["some expectation"],
        response_text="some response",
        judge_llm=judge_obj,
    )
    assert len(report.results) == 1
    assert report.results[0].passed
    assert report.results[0].criterion == "some expectation"
    assert report.results[0].reason == "LGTM object"
    assert report.results[0].confidence == 9
