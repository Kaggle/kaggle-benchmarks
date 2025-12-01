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
# title: Roman Numerals
# ---
# This example shows how to test an LLM with a sequence of related questions in a single task.
# We use Roman numerals to check its ability to convert numbers and do math over several conversational turns.

# This example requires the `roman` library for calculating correct Roman numeral values.
# ```bash
# pip install roman
# ```
# %%

import re

import roman

from kaggle_benchmarks import actors, assertions, llms, task

print(sum(len(roman.toRoman(i)) for i in range(1, 101)))

# %%


@task(name="Roman Numerals")
def roman_numerals(llm):
    """Evaluate the LLM's ability to work with Roman numerals."""

    response = llm.prompt("Hey, can you write 2025 in Roman numerals?")
    assertions.assert_in("MMXXV", response, "Expected `MMXXV`")

    response = llm.prompt("Now, what about MMLXIII? What's that in regular numbers?")
    assertions.assert_in("2063", response, "Should be `2063`")

    response = llm.prompt(
        # 1009 - 6 - 6 - 500 + 504
        "Okay, let's do some math with Roman numerals: MIX - VI - VI - D + DIV + X * X"
    )

    assertions.assert_true(
        "1001" in response or "MI" in response,
        "The correct answer should contain `1001` or `MI`",
    )

    response = llm.prompt(
        "What's the total count of individual symbols used for all integers from I to C inclusive?"
    )

    assertions.assert_in(
        "401", response, "Total number of Roman symbols for 1-100 should be 401"
    )


roman_numerals.evaluate(llm=llms.values())

# %%

# %% [markdown]
# An alternative approach, more suitable for testing multiple related inputs, is to define a single task that accepts parameters.
# This allows evaluating the LLM's ability across a range of examples.
# %%


@task()
def roman_eval(llm, expression: str) -> bool:
    arabic_expression = re.sub(
        "[IVXLCDM]+", lambda x: str(roman.fromRoman(x.group(0))), expression
    )
    expected = eval(arabic_expression)

    response = llm.prompt(
        f"Evaluate this expression in Roman numerals and convert it to Arabic: {expression}",
        schema={"explanation": str, "answer": int},
    )
    actors.system.send(f"{expression} = {arabic_expression} = {expected}")

    return response.answer == expected


roman_eval.evaluate(
    llm=llms.values(),
    expression=["VI * DI + VI * CI", "VI + VI + D"],
)

# %%
