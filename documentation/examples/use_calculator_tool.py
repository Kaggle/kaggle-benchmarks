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
# title: Manual Calculator Tool Calling
# ---
# %%
from kaggle_benchmarks import actors, assertions, llm, messages, task

tool = actors.Actor(name="Tool", role="tool", avatar="🛠️")


def run_simple_calculator(a: float, b: float, operator: str) -> float:
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
def use_calculator(
    llm, problem: str, expected_answer: float, stream_mode: bool = False
) -> None:
    calculator_tool = {
        "type": "function",
        "function": {
            "name": "simple_calculator",
            "description": "Calculates the result of an arithmetic operation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first number."},
                    "b": {"type": "number", "description": "The second number."},
                    "operator": {
                        "type": "string",
                        "description": "The operator (+, -, *, /).",
                    },
                },
                "required": ["a", "b", "operator"],
            },
        },
    }
    llm.stream_responses = stream_mode

    actors.user.send(problem)

    tool_call_msg = llm.respond(tools=[calculator_tool])
    tool_calls = tool_call_msg.tool_calls
    assertions.assert_true(
        bool(tool_calls), "LLM was expected to call a tool, but it did not."
    )

    tool_call = tool_calls[0]
    function_args = tool_call.arguments
    # Removes 'signature' parameter in thinking mode.
    function_args.pop("signature", None)
    tool_result = ""
    try:
        tool_result = run_simple_calculator(**function_args)
    except Exception as e:
        tool_result = f"Error executing tool: {type(e).__name__} - {e}"

    tool.send(
        messages.Message(
            sender=tool,
            content=str(tool_result),
            _meta={"tool_call_id": tool_call.call_id},
        )
    )

    final_answer_msg = llm.respond()
    final_answer = final_answer_msg.content

    assertions.assert_true(
        str(expected_answer) in final_answer,
        f"Expected '{expected_answer}' to be in the final answer, but got '{final_answer}'.",
    )


problem = "What is 485 multiplied by 12?"
expected = 485 * 12

# %%
use_calculator.run(llm, problem=problem, expected_answer=expected, stream_mode=True)

# %%
use_calculator.run(llm, problem=problem, expected_answer=expected, stream_mode=False)

# %%
