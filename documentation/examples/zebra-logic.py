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
# title: Zebra Logic
# ---
"""
https://huggingface.co/blog/yuchenlin/zebra-logic
"""

# %%
import io
from typing import Any

import pandas as pd

from kaggle_benchmarks import assertions, llm, prompting, task, utils


@assertions.assertion_handler()
def assert_frame_equal(
    left: pd.DataFrame,
    right: pd.DataFrame,
    expectation: str | None = None,
    **kwargs: Any,
) -> assertions.AssertionResult:
    """
    Asserts that two pandas DataFrames are equal.

    This is a wrapper around `pd.testing.assert_frame_equal`.
    Any additional keyword arguments are passed directly to it.

    Args:
        left: The first DataFrame.
        right: The second DataFrame.
        expectation: An optional custom message to summarize assert expectation.
        **kwargs: Additional keyword arguments to pass to
                  `pd.testing.assert_frame_equal`.

    Returns:
        An AssertionResult indicating the outcome.
    """
    passed: bool

    try:
        pd.testing.assert_frame_equal(left, right, **kwargs)
        passed = True
    except AssertionError:
        passed = False

    return assertions.AssertionResult(
        passed=passed,
        expectation=expectation or "Data frames should be equal.",
    )


@prompting.handler(types=pd.DataFrame)
def pandas_dataframe(_):
    content = yield "Write you reasoning and wrap the answer as a raw csv file using ```. Make sure to include a header and use quotation marks if needed."
    content = utils.extract_code_block(content)
    return pd.read_csv(io.StringIO(content))


two_houses = """
There are 2 houses, numbered 1 to 2 from left to right.
Each house is occupied by a different person.
Each house has a unique attribute for each of the following characteristics:
- Each person has a unique name: **Arnold, Eric**
- People own unique car models: **ford f150, tesla model 3**
- The people keep unique animals: **cat, horse**

**Clues**:
1. Eric is directly left of the person who owns a Tesla Model 3.
2. The person who keeps horses is in the first house.

Output table should have 'Houses', 'Name', 'CarModel', 'Animal' columns.

"""

expected = pd.DataFrame(
    [
        ["Eric", "ford f150", "horse"],
        ["Arnold", "tesla model 3", "cat"],
    ],
    columns=["Name", "CarModel", "Animal"],
).sort_values("Name")


@task(name="Two Houses Logic Puzzle")
def puzzle(llm):
    """Benchmark to evaluate LLM's ability to solve a two-houses logic puzzle."""
    df = llm.prompt(
        two_houses,
        schema=pd.DataFrame,
    )

    assert_frame_equal(
        left=df.sort_values("Name").reset_index(drop=True).drop(columns=["Houses"]),
        right=expected.reset_index(drop=True),
        expectation="Data frames should be equal.",
    )


puzzle.run(llm)

# %%
