# Copyright 2026 Kaggle Inc.
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

"""Native tool-calling benchmark tasks, and their golden tests.

Each task is followed by its tests: a scripted one that replays canned responses
through ``fake(...)`` and runs with no API key, and a live one parametrized over
a model pool, which skips when no provider is configured. Tests asserting a
*failure* are scripted only — a real model may well get it right.

Each scripted list scripts the exact tool-call sequence the task expects: a
scripted response may be an ``LLMMessage`` carrying ``tool_calls`` (the harness
runs the tool and feeds the result back), and the *next* scripted response is the
model's follow-up. See ``tests/test_tool_loop.py`` for the underlying mechanics.
"""

import pytest
from models import STREAMING_TOOL_MODELS, TOOL_MODELS, TOOL_SCHEMA_MODELS, fake
from pydantic import BaseModel, Field

import kaggle_benchmarks as kbench
from kaggle_benchmarks.llm_messages import LLMMessage
from kaggle_benchmarks.tools.base import ToolInvocation


def _tool_call(name: str, arguments: dict, call_id: str = "c1") -> LLMMessage:
    """Scripts a single tool-call turn: an assistant message with no text that
    requests ``name`` with ``arguments`` (see ``tests/test_tool_loop.py``)."""
    return LLMMessage(
        sender=None,
        content="",
        tool_calls=[ToolInvocation(name=name, arguments=arguments, call_id=call_id)],
    )


def run_simple_calculator(a: float, b: float, operator: str) -> float:
    """Supported operators are: + - * and /"""
    if operator == "+":
        return a + b
    if operator == "-":
        return a - b
    if operator == "*":
        return a * b
    if operator == "/":
        return a / b
    raise ValueError(f"Unknown operator: {operator}")


def increment_counter() -> int:
    """Increments a counter and returns the value."""
    increment_counter.count += 1
    return increment_counter.count


def add_tool(a: float, b: float) -> float:
    """Adds two numbers."""
    add_tool.calls += 1
    return a + b


def multiply_tool(a: float, b: float) -> float:
    """Multiplies two numbers."""
    multiply_tool.calls += 1
    return a * b


def get_user_profile(user_id: str) -> dict:
    """Returns user profile information as a dictionary."""
    if user_id == "user_123":
        return {"name": "Alice", "role": "Admin", "skills": ["Python", "SQL"]}
    return {"name": "Unknown", "role": "User", "skills": []}


def flaky_tool() -> str:
    """This tool always fails with an error."""
    raise ValueError("Tool execution failed simulated error.")


def lookup_city_population(city: str) -> int:
    """Looks up the population of a city. Returns the population as an integer."""
    populations = {"Tokyo": 14_000_000, "Paris": 2_100_000, "London": 9_000_000}
    return populations.get(city, 0)


def format_population(population: int) -> str:
    """Formats a population number with thousands separators."""
    return f"{population:,}"


def get_city_data(city_name: str) -> dict:
    """Returns data about a city including its population."""
    data = {
        "Berlin": {"name": "Berlin", "population": 3_700_000},
        "Sydney": {"name": "Sydney", "population": 5_300_000},
    }
    return data.get(city_name, {"name": city_name, "population": 0})


class CityInfo(BaseModel):
    """Structured info about a city."""

    name: str = Field(description="The city name.")
    population: int = Field(description="The city's population.")


@kbench.task(name="simple_tool_use")
def simple_tool_use(llm):
    problem = "What is 50 plus 25?"
    expected_answer = 75.0

    final_answer = llm.prompt(problem, tools=[run_simple_calculator])
    kbench.assertions.assert_tool_was_invoked(run_simple_calculator)

    kbench.assertions.assert_true(
        str(int(expected_answer)) in final_answer,
        f"Expected '{expected_answer}' to be in the final answer, got '{final_answer}'.",
    )


def test_simple_tool_use_scripted():
    responses = [
        _tool_call("run_simple_calculator", {"a": 50, "b": 25, "operator": "+"}),
        "75.",
    ]
    assert simple_tool_use.run(fake(responses)).passed


def test_simple_tool_use_without_tool_call_fails():
    # Answers directly, never calling the tool.
    assert not simple_tool_use.run(fake(["75."])).passed


@pytest.mark.parametrize("llm", TOOL_MODELS)
def test_simple_tool_use(llm):
    assert simple_tool_use.run(llm).passed


@kbench.task(name="streaming_tool_use")
def streaming_tool_use(llm):
    """Same as simple_tool_use but with streaming enabled."""
    llm.stream_responses = True

    problem = "What is 50 plus 25?"
    expected_answer = 75.0

    final_answer = llm.prompt(problem, tools=[run_simple_calculator])
    kbench.assertions.assert_tool_was_invoked(run_simple_calculator)

    kbench.assertions.assert_true(
        str(int(expected_answer)) in final_answer,
        f"Expected '{expected_answer}' to be in the final answer, got '{final_answer}'.",
    )


def test_streaming_tool_use_scripted():
    responses = [
        _tool_call("run_simple_calculator", {"a": 50, "b": 25, "operator": "+"}),
        "75.",
    ]
    assert streaming_tool_use.run(fake(responses)).passed


@pytest.mark.parametrize("llm", STREAMING_TOOL_MODELS)
def test_streaming_tool_use(llm):
    assert streaming_tool_use.run(llm).passed


@kbench.task(name="stateful_tool_double_execution")
def stateful_tool_double_execution(llm):
    increment_counter.count = 0  # Reset for each test run

    llm.prompt("Call the increment_counter tool.", tools=[increment_counter])

    kbench.assertions.assert_equal(
        1, increment_counter.count, expectation="Tool should be executed exactly once."
    )


def test_stateful_tool_double_execution_scripted():
    responses = [_tool_call("increment_counter", {}), "Done."]
    assert stateful_tool_double_execution.run(fake(responses)).passed


def test_stateful_tool_double_execution_called_twice_fails():
    llm = fake(
        [
            _tool_call("increment_counter", {}, call_id="c1"),
            _tool_call("increment_counter", {}, call_id="c2"),
            "Done.",
        ]
    )
    assert not stateful_tool_double_execution.run(llm).passed


@pytest.mark.parametrize("llm", TOOL_MODELS)
def test_stateful_tool_double_execution(llm):
    assert stateful_tool_double_execution.run(llm).passed


@kbench.task(name="multiple_tool_selection")
def multiple_tool_selection(llm):
    add_tool.calls = 0
    multiply_tool.calls = 0

    llm.prompt(
        "What is 12 multiplied by 34? Use the multiply_tool.",
        tools=[add_tool, multiply_tool],
    )

    kbench.assertions.assert_equal(
        1, multiply_tool.calls, expectation="Multiply tool should be called once."
    )
    kbench.assertions.assert_equal(
        0, add_tool.calls, expectation="Add tool should not be called."
    )


def test_multiple_tool_selection_scripted():
    responses = [_tool_call("multiply_tool", {"a": 12, "b": 34}), "408."]
    assert multiple_tool_selection.run(fake(responses)).passed


def test_multiple_tool_selection_wrong_tool_fails():
    llm = fake([_tool_call("add_tool", {"a": 12, "b": 34}), "46."])
    assert not multiple_tool_selection.run(llm).passed


@pytest.mark.parametrize("llm", TOOL_MODELS)
def test_multiple_tool_selection(llm):
    assert multiple_tool_selection.run(llm).passed


@kbench.task(name="complex_tool_return")
def complex_tool_return(llm):
    response = llm.prompt(
        "Get the profile for user_123 and tell me what their role is.",
        tools=[get_user_profile],
    )

    kbench.assertions.assert_contains_regex(
        r"(?i)admin", response, expectation="Model should identify the role as Admin."
    )


def test_complex_tool_return_scripted():
    responses = [
        _tool_call("get_user_profile", {"user_id": "user_123"}),
        "Alice is an Admin.",
    ]
    assert complex_tool_return.run(fake(responses)).passed


@pytest.mark.parametrize("llm", TOOL_MODELS)
def test_complex_tool_return(llm):
    assert complex_tool_return.run(llm).passed


@kbench.task(name="tool_error_handling")
def tool_error_handling(llm):
    response = llm.prompt(
        "Call the flaky_tool and report what happens.", tools=[flaky_tool]
    )

    kbench.assertions.assert_contains_regex(
        r"(?i)error|failed|valueerror",
        response,
        expectation="Model should report the tool failure.",
    )


def test_tool_error_handling_scripted():
    responses = [_tool_call("flaky_tool", {}), "The tool failed with an error."]
    assert tool_error_handling.run(fake(responses)).passed


@pytest.mark.parametrize("llm", TOOL_MODELS)
def test_tool_error_handling(llm):
    assert tool_error_handling.run(llm).passed


@kbench.task(name="multi_step_tool_chain")
def multi_step_tool_chain(llm):
    """Tests that the LLM can chain tool calls: look up a value then format it."""
    response = llm.prompt(
        "What is the population of Tokyo? "
        "First look it up with lookup_city_population, "
        "then format the result with format_population.",
        tools=[lookup_city_population, format_population],
    )

    kbench.assertions.assert_tool_was_invoked(lookup_city_population)
    kbench.assertions.assert_contains_regex(
        r"14,000,000",
        response,
        expectation="Response should contain the formatted population of Tokyo.",
    )


def test_multi_step_tool_chain_scripted():
    responses = [
        _tool_call("lookup_city_population", {"city": "Tokyo"}, call_id="c1"),
        _tool_call("format_population", {"population": 14_000_000}, call_id="c2"),
        "14,000,000.",
    ]
    assert multi_step_tool_chain.run(fake(responses)).passed


@pytest.mark.slow
@pytest.mark.parametrize("llm", TOOL_MODELS)
def test_multi_step_tool_chain(llm):
    assert multi_step_tool_chain.run(llm).passed


@kbench.task(name="tool_with_schema_output")
def tool_with_schema_output(llm):
    """Tests that tools and schema= work together: tool provides data,
    LLM returns structured output."""
    result = llm.prompt(
        "Look up the city data for Berlin and return it as a CityInfo.",
        tools=[get_city_data],
        schema=CityInfo,
    )

    kbench.assertions.assert_tool_was_invoked(get_city_data)
    kbench.assertions.assert_true(
        isinstance(result, CityInfo),
        f"Expected CityInfo, got {type(result).__name__}",
    )
    kbench.assertions.assert_equal(
        "Berlin", result.name, expectation="City name should be Berlin."
    )
    kbench.assertions.assert_equal(
        3_700_000,
        result.population,
        expectation="Population should be 3,700,000.",
    )


def test_tool_with_schema_output_scripted():
    # Tool round, a text turn to end the loop, then the schema-formatting response.
    responses = [
        _tool_call("get_city_data", {"city_name": "Berlin"}),
        "Done.",
        {"name": "Berlin", "population": 3_700_000},
    ]
    assert tool_with_schema_output.run(fake(responses)).passed


@pytest.mark.parametrize("llm", TOOL_SCHEMA_MODELS)
def test_tool_with_schema_output(llm):
    assert tool_with_schema_output.run(llm).passed
