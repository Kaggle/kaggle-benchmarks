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
# title: A collection of demo candidates
# ---


# %%
import dataclasses
import enum
import re
from typing import Sequence

import kaggle_benchmarks as kbench


# %%
@kbench.task("subtraction")
def test_subtraction(llm):
    # Explicitly set stream mode.
    llm.stream_responses = True
    response = llm.prompt("9.9 - 9.11 = ?")
    kbench.assertions.assert_in("0.79", response, expectation="Expect 9.9-9.11=0.79")


test_subtraction.run(kbench.llm)


# %%
@kbench.task("understanding the world")
def test_understanding_the_world(llm):
    response = llm.prompt(
        "Can a pole 10 meters long pass a door that is 2 meters tall and 1 meter wide? Give your answer and explanation.",
        schema={"answer": bool, "explanation": str},
    )
    kbench.assertions.assert_true(
        response.answer,
        expectation="A 10m pole can pass through a 2m x 1m door.",
    )
    kbench.assertions.assert_in(
        "3D",
        response.explanation,
        expectation="The explanation should mention the 3D nature of the problem.",
    )


test_understanding_the_world.run(kbench.llm)


# %%
@kbench.task("parsing_html")
def test_parsing_html(llm):
    response = llm.prompt("""Given the following table, what's the median daily sunlight hours of the cities?
                          <table>
                            <tbody>
                                <tr>
                                    <td>1</td>
                                    <td>City 1</td>
                                    <td>10</td>
                                </tr>
                                <tr>
                                    <td>2</td>
                                    <td>City 2</td>
                                    <td>8</td>
                                </tr>
                                <tr>
                                    <td>3</td>
                                    <td>City 3</td>
                                    <td>7</td>
                                </tr>
                            </tbody>
                          </table>
                          """)
    kbench.assertions.assert_in(
        "8",
        response,
        expectation="The model should find the median of 10, 8, and 7 is 8.",
    )


test_parsing_html.run(kbench.llm)


# %%
@kbench.task("hallucination")
def test_hallucination(llm):
    response = llm.prompt(
        "When Richard Feynman mentioned gravity-light-contraction theory in his Nobel speech in 1965, did he think it was important?",
        schema={
            "answer": bool,
            "explanation": str,
        },
    )
    kbench.assertions.assert_false(
        response.answer,
        expectation="The model should recognize that 'gravity-light-contraction theory' is fictitious.",
    )
    kbench.assertions.assert_contains_regex(
        r"(not|never|no|didn't)",
        response.explanation.lower(),
        expectation="The explanation should state that the theory was not mentioned.",
    )


test_hallucination.run(kbench.llm)


# %%
@dataclasses.dataclass(frozen=True)
class FewShotResponse:
    number: int
    reason: str


@kbench.task("few-shot")
def test_few_shot(llm):
    response = llm.prompt(
        """Here is a list of letters and their associated numbers:
        H: 1
        B: 5
        C: 6
        F: 9
        P: 15
        What is the number for S and what's your reason?
        """,
        schema=FewShotResponse,
    )
    kbench.assertions.assert_equal(
        16,
        response.number,
        expectation="The model should deduce the pattern and find the correct number for 'S'.",
    )


test_few_shot.run(kbench.llm)


# %%
@dataclasses.dataclass(frozen=True)
class CodeAnalysisResponse:
    has_bugs: bool
    fixed_code: str


@kbench.task("code-analysis")
def test_code_analysis(llm):
    code = """
fruits = [
  'apple',
  'orange'
  'banana',
  'peach',
  'avocado'
]

print(len(fruits))
    """
    kbench.system.send("You are an expert Python programmer.")
    response = llm.prompt(
        f"Check carefully the following code. Does it have any bugs? If so, show me the fixed code.\nCode:\n{code}",
        schema=CodeAnalysisResponse,
    )
    kbench.assertions.assert_true(
        response.has_bugs,
        expectation="The model should identify the missing comma bug in the Python code.",
    )
    fixed_code = kbench.tools.python.extract_code(response.fixed_code)
    fixed_code_output = kbench.tools.python.script_runner.run_code(fixed_code)
    kbench.assertions.assert_equal(
        "5",
        fixed_code_output.stdout.strip(),
        expectation="The fixed code should output 5, the correct length of the list.",
    )


test_code_analysis.run(kbench.llm)


# %%
class Usecase(enum.Enum):
    FRONTEND = enum.auto()
    BACKEND = enum.auto()
    FULLSTACK = enum.auto()


@dataclasses.dataclass(frozen=True)
class Language:
    name: str
    usecase: Usecase


@dataclasses.dataclass(frozen=True)
class Output:
    languages: Sequence[Language]


@kbench.task("structured_output")
def test_structured_output(llm):
    response = llm.prompt(
        "List the 5 most popular programming languages. For each language, tell me its name and its usecase (FRONTEND, BACKEND, or FULLSTACK).",
        schema=Output,
    )
    kbench.assertions.assert_equal(
        5,
        len(response.languages),
        expectation="The model should list exactly 5 languages.",
    )
    kbench.assertions.assert_in(
        "javascript",
        [language["name"].lower() for language in response.languages],
        expectation="Javascript should be included in the list of popular languages.",
    )
    kbench.assertions.assert_true(
        any(
            language["name"].lower() == "python"
            and language["usecase"] in ("BACKEND", "FULLSTACK")
            for language in response.languages
        ),
        expectation="Python should be listed with its usecase as either BACKEND or FULLSTACK.",
    )


test_structured_output.run(kbench.llm)


# %%
@dataclasses.dataclass(frozen=True)
class Puzzle24Response:
    is_feasible: bool
    solution: str


def validate_solution(solution: str, integers: Sequence[int]) -> None:
    kbench.assertions.assert_true(
        re.search(r"[^0-9+\-*/()\s]", solution) is None,
        expectation="Solution should NOT have unexpected operators.",
    )
    numbers_in_solution = [int(num_str) for num_str in re.findall(r"\d+", solution)]
    kbench.assertions.assert_equal(
        sorted(integers),
        sorted(numbers_in_solution),
        expectation="Solution should use the given integers.",
    )
    solution_result = (
        kbench.tools.python.IPythonREPL()
        .invoke(solution, is_visible_to_llm=False)
        .output
    )
    kbench.assertions.assert_equal(
        24,
        float(solution_result),
        expectation="The solution expression should evaluate to 24.",
    )


@kbench.task("puzzle 24")
def test_puzzle_24(llm):
    prompt = """
    You are a very good mathematician agent.

    Here is your task:
    Given four integers, combine them
    by using only +, -, *, / operators to get 24.

    Your response should have two parts:
    - `is_feasible`: whether a solution is feasible.
    - `solution`: the actual expression to get 24. It can be empty if the solution is not feasible.
    """

    test_data = (
        ((1, 2, 3, 4), True),
        ((1, 1, 1, 1), False),
        ((13, 1, 1, 1), True),
        ((9, 9, 2, 8), True),
    )

    kbench.system.send(prompt)
    for integers, is_feasible in test_data:
        four_integers = ", ".join(str(i) for i in integers)
        response = llm.prompt(
            f"Here is the four integers: {four_integers}\nYour answer:",
            schema=Puzzle24Response,
        )
        kbench.assertions.assert_equal(
            is_feasible,
            response.is_feasible,
            expectation="The model should correctly identify if a solution is feasible.",
        )
        if response.is_feasible:
            validate_solution(response.solution, integers)


test_puzzle_24.run(kbench.llm)


# %%
@kbench.task("Recognize animals from images")
def test_recognize_animals_from_images(llm):
    image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg"
    )
    kbench.user.send(kbench.content_types.images.from_url(image_url))

    def recognize_image():
        response = llm.prompt("What is the name of this animal?")
        kbench.assertions.assert_in(
            "cat",
            response.lower(),
            expectation="The model should identify the animal in the image as a cat.",
        )

    kbench.assertions.assert_raises_no_exceptions(
        recognize_image,
        expectation="The image recognition prompt should run without errors.",
    )


test_recognize_animals_from_images.run(kbench.llm)

# %%


@kbench.task(name="Wise Goose in HTML")
def test_html_coding(llm, seed: int = 0):
    """Benchmark to evaluate the LLM's ability to generate a dynamic HTML page."""
    instructions = """
    Create a responsive, animated HTML page featuring a goose that says something in a comic book style every second.
    Every message should have `words-of-wisdom` css class and should be logged in console.
    The animation should be implemented using vanilla JavaScript and emojis, and the page should adapt to different screen sizes and devices
    """
    response = llm.prompt(instructions, seed=seed)
    with kbench.tools.web.Browser() as b:
        snapshot = b.take_snapshot(
            kbench.tools.web.extract_html(response),
            wait_before=5000,
            full_page=True,
        )

        num_logs = len(snapshot.logs)
    kbench.assertions.assert_true(
        3 < num_logs < 10,
        expectation=f"Expected between 4 and 9 console logs (inclusive), but got {num_logs}.",
    )

    kbench.assertions.assert_in(
        "words-of-wisdom",
        snapshot.html,
        expectation="Expected 'words-of-wisdom' CSS class in the HTML.",
    )


test_html_coding.run(kbench.llm, seed=1234)
# %%
