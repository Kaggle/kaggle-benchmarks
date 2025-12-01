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
# title: Test case for task/run json dump.
# ---
# This module demonstrates an example with a single top-level task (`simple_task`).
# This top-level task invokes multiple subtasks (`subtask1`, `subtask2`).
# The result of the `simple_task` is calculated as the ratio of
# the number of passed subtasks to the total number of subtasks executed.
# %%
from kaggle_benchmarks import assertions, llm, task


@task("subtask1", description="greeting")
def subtask1(llm):
    response = llm.prompt("Hello!")
    assertions.assert_not_empty(response)


@task("subtask2", description="introduction")
def subtask2(llm, name: str):
    response = llm.prompt(f"My name is {name}. What is my name?")
    assertions.assert_in(name, response)


@task("root_task", description="top level task")
def simple_task(llm) -> float:
    subtask_runs = [
        subtask1.run(llm),
        subtask2.run(llm, "Alan Turing"),
        subtask2.run(llm, "Richard Feynman"),
    ]

    return sum(subtask.passed for subtask in subtask_runs) / len(subtask_runs)


# %%

run = simple_task.run(llm)

# %%
run

# %%
