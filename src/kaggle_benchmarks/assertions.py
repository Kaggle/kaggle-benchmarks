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

"""Benchmark assertions."""

import dataclasses
import functools
import inspect
import re
import textwrap
import uuid
from typing import Any, Callable, Iterable, List

import panel as pn

from kaggle_benchmarks import chats


@dataclasses.dataclass
class AssertionResult:
    passed: bool
    # A message summarizing the expectation of the assertion.
    expectation: str
    details: dict[str, Any] | None = None
    id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)

    def __panel__(self) -> pn.viewable.Viewable:
        """Custom Panel representation for an AssertionResult."""
        status_icon = "✅" if self.passed else "❌"
        title_color = "green" if self.passed else "red"

        objects = [
            pn.pane.Markdown(
                f"### {status_icon} Assertion {'Passed' if self.passed else 'Failed'}",
                styles={"color": title_color},
            )
        ]

        if not self.passed:
            objects.append(
                pn.pane.Markdown(
                    f"Failed assertion with expectation: {self.expectation}"
                )
            )

        if self.details:
            source_code = self.details.get("source_code")
            line_number = self.details.get("line_number")
            if source_code:
                check_info = f"`{source_code}`"
                if line_number is not None:
                    check_info += f" (L{line_number})"
                objects.append(pn.pane.Markdown(f"**Check:** {check_info}"))

        return pn.Column(*objects, sizing_mode="stretch_width")


class GetAssertExpressionError(Exception):
    """Custom exception for errors during assertion expression retrieval."""

    pass


def _report_assert_result(result: AssertionResult):
    from kaggle_benchmarks import actors, contexts

    current_run = contexts.get_current().run
    if current_run:
        current_run.assertion_results.append(result)

    actors.assertion.send(result, is_visible_to_llm=False)


def _get_assert_expression(depth: int = 2) -> tuple[int, str | None]:
    """Tries to get the source code and its line number that called an assertion."""

    frame = inspect.currentframe()
    if frame is None:
        raise GetAssertExpressionError("Unable to retrieve current execution frame.")

    # Traverse up the call stack to the desired frame
    for i in range(depth):
        if frame.f_back:
            frame = frame.f_back
        else:
            raise GetAssertExpressionError(
                f"Call stack depth insufficient (requested {depth}, actual < {i + 1})."
            )

    frame_info = inspect.getframeinfo(frame, context=1)

    if frame_info.code_context and len(frame_info.code_context) > 0:
        source_line = frame_info.code_context[0].strip()
        return frame_info.lineno, source_line
    raise GetAssertExpressionError(
        f"Source code context not found for line {frame_info.lineno} in {frame_info.filename}."
    )


def assertion_handler(
    assertion_type_name: str | None = None, raises_assertion_error: bool = False
):
    """
    Decorator factory for creating custom assertion logic functions.

    The decorated function must be type-hinted to return an `AssertionResult`
    instance. The `AssertionResult` object will have its `details` attribute
    populated by this handler with the 'assertion_type' and 'expression'.

    Args:
        assertion_type_name: An optional name for this assertion type.
            If None (default), the name of the decorated function will be used.
        raises_assertion_error: If True, the decorator will raise an
            AssertionError if the underlying assertion logic indicates a failure
            (i.e., result.passed is False). This mimics standard 'assert'
            behavior. If False (default), the failure is recorded, but an
            AssertionError is not raised, allowing execution to continue.

    Returns:
        A decorator that wraps the assertion logic function.
    """

    def decorator(func):
        resolved_assertion_type = assertion_type_name or func.__name__

        # Enforce that the decorated function returns AssertionResult
        sig = inspect.signature(func)
        if sig.return_annotation is not AssertionResult:
            raise TypeError(
                f"Function '{func.__name__}' decorated with assertion_handler "
                f"must have a return type annotation of 'AssertionResult', "
                f"but found '{sig.return_annotation}'. "
            )

        @functools.wraps(func)  # Preserves name, docstring, etc.
        def wrapper(*args, **kwargs):
            # Correct depth:
            # Frame 0: _get_assert_expression
            # Frame 1: wrapper (this function)
            # Frame 2: The actual user-level call to the decorated assert function
            try:
                line_no, source_expr = _get_assert_expression(depth=2)
            except GetAssertExpressionError:
                line_no, source_expr = None, None

            # Call the original assertion logic
            original_result: AssertionResult = func(*args, **kwargs)

            # Create a new AssertionResult by copying the original_result.
            # This ensures that if original_result was already reported by a nested
            # assertion_handler, its details are not mutated by this outer handler.
            final_result = dataclasses.replace(original_result)

            final_result.details = {
                "assertion_type": resolved_assertion_type,
                "line_number": line_no,
                "source_code": source_expr,
            }
            _report_assert_result(final_result)

            if raises_assertion_error and not final_result.passed:
                raise AssertionError(
                    f"Failed assertion with expectation: {final_result.expectation}"
                )
            return final_result

        return wrapper

    return decorator


@assertion_handler()
def assert_equal(
    expected: Any,
    actual: Any,
    expectation: str | None = None,
) -> AssertionResult:
    """
    Asserts that two values are equal.

    Args:
        expected: The expected value.
        actual: The actual value.
        expectation: An optional message summarizing the assertion.
    """
    passed = actual == expected

    return AssertionResult(
        passed=passed,
        expectation=expectation or f"Expected: '{expected}', Got: '{actual}'",
    )


@assertion_handler()
def assert_true(expr: bool, expectation: str | None = None) -> AssertionResult:
    """
    Asserts that the given expression is True.

    Args:
        expr: The expression to evaluate.
        expectation: An optional message summarizing the assertion.

    """
    passed = expr

    return AssertionResult(
        passed=passed,
        expectation=expectation or f"Expected '{expr}' to be True",
    )


@assertion_handler()
def assert_false(expr: bool, expectation: str | None = None) -> AssertionResult:
    """
    Asserts that the given expression is False.

    Args:
        expr: The expression to evaluate.
        expectation: An optional message summarizing the assertion.

    """
    passed = not expr

    return AssertionResult(
        passed=passed,
        expectation=expectation or f"Expected '{expr}' to be False",
    )


@assertion_handler()
def assert_fail(expectation: str | None = None) -> AssertionResult:
    """
    Signals a test failure unconditionally, with optional message.

    Args:
        expectation: An optional message summarizing the assertion.

    """
    return AssertionResult(
        passed=False,
        expectation=expectation or "Assertion failed unconditionally.",
    )


@assertion_handler()
def assert_in(
    member: Any,
    container: Any,
    expectation: str | None = None,
) -> AssertionResult:
    """
    Asserts that a member is present inside the container.

    Args:
        member: The element to check for membership.
        container: The collection to check against.
        expectation: An optional message summarizing the assertion.

    """
    passed = member in container

    return AssertionResult(
        passed=passed,
        expectation=expectation or f"Expected '{member}' in '{container}'",
    )


@assertion_handler()
def assert_not_in(
    member: Any,
    container: Any,
    expectation: str | None = None,
) -> AssertionResult:
    """
    Asserts that a member is not present inside the container.

    Args:
        member: The element to check for non-membership.
        container: The collection to check against.
        expectation: An optional message summarizing the assertion.

    """
    passed = member not in container

    return AssertionResult(
        passed=passed,
        expectation=expectation or f"Expected '{member}' not in '{container}'",
    )


@assertion_handler()
def assert_empty(
    container: Any,
    expectation: str | None = None,
) -> AssertionResult:
    """
    Asserts that the given container is empty.

    Args:
        container: The container to check if empty.
        expectation: An optional message summarizing the assertion.

    """
    passed = not container

    return AssertionResult(
        passed=passed,
        expectation=expectation or f"Expected '{container}' to be empty",
    )


@assertion_handler()
def assert_not_empty(
    container: Any,
    expectation: str | None = None,
) -> AssertionResult:
    """
    Asserts that the given container is not empty.

    Args:
        container: The container to check if not empty.
        expectation: An optional message summarizing the assertion.

    """
    passed = bool(container)

    return AssertionResult(
        passed=passed,
        expectation=expectation or f"Expected '{container}' not to be empty",
    )


@assertion_handler()
def assert_contains_regex(
    pattern: str | re.Pattern[str],
    text: str,
    expectation: str | None = None,
) -> AssertionResult:
    """Asserts that the given regex pattern is found anywhere in the text.

    Args:
        pattern: The regex pattern to search for.
        text: The string to search within.
        expectation: An optional message summarizing the assertion.
    """
    passed = re.search(pattern, text) is not None

    return AssertionResult(
        passed=passed,
        expectation=expectation or f"Expected pattern '{pattern}' found in '{text}'",
    )


@assertion_handler()
def assert_not_contains_regex(
    pattern: str | re.Pattern[str],
    text: str,
    expectation: str | None = None,
) -> AssertionResult:
    """Asserts that the given regex pattern is not found anywhere in the text.

    Args:
        pattern: The regex pattern to search for.
        text: The string to search within.
        expectation: An optional message summarizing the assertion.
    """
    passed = re.search(pattern, text) is None

    return AssertionResult(
        passed=passed,
        expectation=expectation
        or f"Expected pattern '{pattern}' not found in '{text}'",
    )


@assertion_handler()
def assert_raises_no_exceptions(
    callable_obj: Callable[..., Any],
    expectation: str | None = None,
    *args: Any,
    **kwargs: Any,
) -> AssertionResult:
    """
    Asserts that `callable_obj(*args, **kwargs)` does not raise any exception.

    Args:
        callable_obj: The callable to execute.
        expectation: An optional message summarizing the assertion.
        *args: Positional arguments to pass to `callable_obj`.
        **kwargs: Keyword arguments to pass to `callable_obj`.

    Returns:
        An AssertionResult indicating the outcome.
    """
    raised_exception_instance: BaseException | None = None
    try:
        callable_obj(*args, **kwargs)
    except BaseException as e:
        raised_exception_instance = e

    passed = raised_exception_instance is None

    return AssertionResult(
        passed=passed,
        expectation=expectation or "Expected no exception to be raised",
    )


@dataclasses.dataclass
class AssessResult:
    """Holds the result of a single criterion check."""

    criterion: str
    passed: bool
    reason: str
    confidence: int  # 0-10


@dataclasses.dataclass
class AssessReport:
    """Holds the list of results and provides helper methods."""

    results: List[AssessResult]

    def __post_init__(self):
        """
        Automatically converts dictionary items to
        AssessResult objects if the LLM library returns raw dicts.
        """
        cleaned_results = []
        for item in self.results:
            if isinstance(item, dict):
                cleaned_results.append(AssessResult(**item))
            elif isinstance(item, AssessResult):
                cleaned_results.append(item)
            else:
                continue
        self.results = cleaned_results


def assess_response_with_judge(
    criteria: Iterable[str],
    response_text: str,
    judge_llm: Any,
    prompt: Callable[[Iterable[str], str], str] | None = None,
) -> AssessReport:
    """
    Assess if an LLM response satisfies a list of specific criteria using a Judge LLM.

    This function runs an assertion against each criterion, and returns the assessment report.

    Args:
        criteria: A list of requirements or facts the response must satisfy.
        response_text: The actual output generated by the LLM under test.
        judge_llm: The LLM interface object. Must support .prompt(text, schema=...).
        prompt: An optional callable that generates the prompt. It accepts (criteria, response_text)
            and returns a string. If None, a default prompt is constructed.

    Returns:
        AssessReport: An object containing a list of results for each criterion.

    Raises:
        RuntimeError: If the judge_llm fails to generate a valid response.
    """

    if prompt is None:
        formatted_criteria = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(criteria))

        prompt_text = textwrap.dedent(f"""
        You are an impartial and strict technical evaluator.
        Your job is to verify if a generated response complies strictly with a provided list of criteria.

        ### Input Data
        **Generated Response:**
        {response_text}

        **Criteria to Verify:**
        {formatted_criteria}

        ### Evaluation Protocol
        1. **Analyze Individually:** Assess the response against EACH criterion strictly in isolation.
        2. **Cite Evidence:** You must identify specific segments of the text that support your decision. If no evidence exists, the criterion fails.
        3. **Handle Ambiguity:** - If the text is silent or ambiguous regarding a criterion, `passed` must be `False`.
            - Do not make assumptions or infer missing information.
        4. **Scoring:** Assign a confidence score (0-10):
            - 10: Explicit, verbatim evidence exists.
            - 5-9: Strong implicit evidence exists.
            - 0-4: Evidence is weak, ambiguous, or non-existent.

        ### Output Format
        You must output a single valid JSON object containing a root key `"results"`. The value of `"results"` is a list of objects.

        **JSON Schema:**
        {{
            "results": [
                {{
                    "criterion": "string (the exact text of the criterion being checked)",
                    "passed": boolean,
                    "reason": "string (concise explanation referencing the text)",
                    "confidence": integer (0-10)
                }}
            ]
        }}

        ### Constraint
        * Return ONLY the JSON object. Do not include markdown formatting (like ```json) or conversational text.
        """)
    else:
        prompt_text = prompt(criteria, response_text)

    try:
        with chats.new(f"Response assessment with {judge_llm.name}"):
            assess_report = judge_llm.prompt(prompt_text, schema=AssessReport)

        if isinstance(assess_report, dict):
            assess_report = AssessReport(**assess_report)

        for result in assess_report.results:
            assert_true(result.passed, expectation=result.criterion)

    except Exception as e:
        assess_report = AssessReport(results=[])
        assert_fail(
            expectation=f"check_with_llm fails because judge_llm cannot respond: {e}",
        )

    return assess_report
