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
# ## Tool Calling Task for Gemini
#
# This example tests the model's ability to use tools. We'll provide a simple
# `multiply` function and ask a question that requires the model to use it.
# %%
from kaggle_benchmarks import actors, assertions, task
from kaggle_benchmarks.kaggle import load_model

llm = load_model(
    model_name="google/gemini-2.5-pro",
    api="genai",
)


def multiply(a: int, b: int) -> int:
    "Multiplies two integers."
    return 42


@task("Test Tool Calling")
def test_tool_calling(llm):
    """Evaluates the LLM's ability to use a provided tool."""

    actors.user.send(
        "What is the result of 1264 times 456? Use provided tools `multiply` to get your answer"
    )
    response = llm.respond(tools=[multiply])
    assertions.assert_in(
        "42",
        response.content,
        "LLM failed to use the tool to get the correct answer.",
    )


test_tool_calling.run(llm)

# %%
