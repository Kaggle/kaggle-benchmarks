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

# %% [markdown]
# ---
# title: Calculator Tool
# ---
# %%
from kaggle_benchmarks import actors, assertions, llm, task

tool = actors.Actor(name="Tool", role="tool", avatar="🛠️")


def run_simple_calculator(a: float, b: float, operator: str) -> float:
    """Calculates the result of an arithmetic operation like +, -, *, or /."""
    if operator == "+":
        return a + b
    if operator == "-":
        return a - b
    if operator == "*":
        return a * b
    if operator == "/":
        return a / b
    raise ValueError(f"Unknown operator: {operator}")


@task("Calculator Tool Use")
def use_calculator(llm, problem: str, expected_answer: float) -> None:
    final_answer = llm.prompt(problem, tools=[run_simple_calculator])
    assertions.assert_tool_was_invoked(
        run_simple_calculator, "LLM was expected to call a tool, but it did not."
    )

    assertions.assert_true(
        str(expected_answer) in answer,
        f"Expected '{expected_answer}' to be in the final answer, but got '{answer}'.",
    )


# %%

problem = "What is 485 multiplied by 12?"
expected = 485 * 12

use_calculator.run(llm, problem=problem, expected_answer=expected)

# %%
