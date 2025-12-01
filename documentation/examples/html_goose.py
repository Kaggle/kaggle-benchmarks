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
# title: Evaluating HTML with JS
# ---
# LLMs can generate complex code, including HTML, CSS, and JavaScript for web pages.
# To accurately evaluate such generations, it's often insufficient to just inspect the raw code.
# The Browser Tool allows you to render the generated web content in a headless browser environment,
# inspect the Document Object Model (DOM), check console logs, and even capture visual snapshots.
#
# This example demonstrates evaluating an LLM's ability to create a dynamic HTML page with specific JavaScript-driven behavior.

# %%
from kaggle_benchmarks import assertions, llm, task, tools

instructions = """
Create a responsive, animated HTML page featuring a goose that says something in a comic book style every second.
Every message should have `words-of-wisdom` css class and should be logged in console.
The animation should be implemented using vanilla JavaScript and emojis, and the page should adapt to different screen sizes and devices
"""


@task(name="Wise Goose in HTML")
def html_goose(llm, seed: int = 0):
    """Benchmark to evaluate the LLM's ability to generate a dynamic HTML page."""
    response = llm.prompt(instructions, seed=seed)
    with tools.web.Browser() as b:
        snapshot = b.take_snapshot(
            tools.web.extract_html(response),
            wait_before=5000,
            full_page=True,
        )

        num_logs = len(snapshot.logs)
        assertions.assert_true(
            3 < num_logs < 10,
            f"Expected between 4 and 9 console logs (inclusive), but got {num_logs}.",
        )

        assertions.assert_in(
            "words-of-wisdom",
            snapshot.html,
            "Expected 'words-of-wisdom' CSS class in the HTML.",
        )


html_goose.run(llm, seed=1234)
# %%
