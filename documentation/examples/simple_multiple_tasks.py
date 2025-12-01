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
# This module demonstrates a simple use case with multiple top-level tasks.
# Each task is defined independently and can be run together or separately.
# %%
import base64

import requests

from kaggle_benchmarks import assertions, chats, content_types, llm, task, tools, user


def image_url_to_base64(url: str) -> str:
    # Explicit User-Agent to avoid https://meta.wikimedia.org/wiki/User-Agent_policy
    headers = {"User-Agent": "test"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an exception for bad status codes
    return base64.b64encode(response.content).decode("utf-8")


@task("task1", description="greeting")
def task1(llm):
    response = llm.prompt("Hello!")
    assertions.assert_not_empty(response)


@task("task2", description="guess animal by image url")
def task2(llm, image_url, solution):
    user.send(content_types.images.from_url(image_url))
    response = llm.prompt("What is the name of this animal?")
    assertions.assert_in(solution.lower(), response.lower())


@task("task3", description="guess animal by image")
def task3(llm, image_url, solution):
    image_base64 = image_url_to_base64(image_url)
    user.send(content_types.images.from_base64(image_base64))
    response = llm.prompt("What is the name of this animal?")
    assertions.assert_in(solution.lower(), response.lower())


@task("task4", description="subchats and multi requests")
def task4(llm):
    response = llm.prompt("Hello!")
    assertions.assert_not_empty(response)
    with chats.new("subchat1"):
        response = llm.prompt("Hello in subchat1!")
        assertions.assert_not_empty(response)
    response = llm.prompt(
        "Write python code to print 'hello world!' Generate only the code but nothing else."
    )
    code = tools.python.extract_code(response)
    code_output = tools.python.script_runner.run_code(code)
    assertions.assert_equal("hello world!", code_output.stdout.strip())
    llm.prompt("bye!")


# %%

run1 = task1.run(llm)
run2 = task2.run(
    llm,
    "https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg",
    "cat",
)
run3 = task3.run(
    llm,
    "https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg",
    "cat",
)
run4 = task4.run(llm)

# %%
run1
# %%
run2
# %%
run3

# %%
run4

# %%
