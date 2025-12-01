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
# title: Bouncing Arrows
# ---
# %%
from kaggle_benchmarks import assertions, llm, task, tools

bouncing_rules = """
Simulate the movement of arrows (`>` and `<`) in a 1D string. Each arrow moves one step per turn according to these rules:

*   `>` moves right, `<` moves left.
*   If an arrow hits a boundary or another arrow, it bounces back (reverses direction).
*   Bouncing takes turns: If arrows are stuck facing each other (e.g., `>>><<<`), they all reverse direction in the next turn (e.g., `<<<>>>`).
*   If two arrows try to move into the same space, the right-moving (`>`) arrow moves forward, while the left-moving (`<`) arrow bounces back.
*   Spaces (`' '`) remain empty.
"""

examples = [
    (" >> ", "  >>"),
    (" >< ", " <> "),
    ("> <", " >< "),
    (">>><<<", "<<<>>>"),
]

# %%


@task()
def bouncing_arrows(llm):
    response = llm.prompt(f"""{bouncing_rules}

    Here are some examples: {"".join(x + " -> " + y for x, y in examples)}

    What is the next step for: " >> <>"
    """)

    assertions.assert_in(" >>><", response, "Wrong answer")


bouncing_arrows.run(llm)
# %%


@task(name="Bouncing simulation in Python")
def bouncing_arrows_code(llm):
    usage_examples = "\n".join(f"next_turn({x!r}) == {y!r}" for x, y in examples)
    r = llm.prompt(f"""
{bouncing_rules}

Task: Write a Python function `next_turn(s: str) -> str` that simulates one step of the arrow movement.

Examples:
{usage_examples}

""")
    code = tools.python.extract_code(r).strip()
    result = tools.python.script_runner.run_code(f"""
{code}

assert next_turn(" >>>") == " <<<"
assert next_turn("> >>") == " ><<"

""")
    assertions.assert_equal(result.exit_code, 0, "Failed on test cases")


bouncing_arrows_code.run(llm)
# %%


@task(name="Bouncing simulation in JavaScript")
def bouncing_arrows_web(llm):
    r = llm.prompt(f"""
{bouncing_rules}

Task: Write an HTML page that simulates the arrow movement, starting with the position "  >>> >> >   > >> >>>" and making a step every 100 milliseconds.  The `div` containing the arrows must have the id "arrows".
""")
    with tools.web.Browser() as b:
        html = tools.web.extract_html(r)
        snapshot = b.take_snapshot(html, wait_before=1000, height=400)
        contains_either_arrow_direction = (">>>" in snapshot.html) or (
            "<<<" in snapshot.html
        )
        assertions.assert_true(
            contains_either_arrow_direction,
            expectation="Expected snapshot HTML to contain '>>>' or '<<<'.",
        )
        assertions.assert_in('id="arrows"', snapshot.html)


bouncing_arrows_web.run(llm)
# %%
