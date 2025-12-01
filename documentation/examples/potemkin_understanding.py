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
# title: Benchmark examples from ["Potemkin Understanding in Large Language Models"](https://arxiv.org/pdf/2506.21521)
# ---


# %%
# Optionally install needed dependencies
# !pip install -q pronouncing syllables

# %%
import pronouncing
import syllables

import kaggle_benchmarks as kbench

# %%


def rhyme_words(word_a, word_b):
    # Don't treat the same words as rhyme words.
    return word_b.lower() in pronouncing.rhymes(word_a.lower())


@kbench.task("ABAB rhyming")
def test_potemkin_abab_rhyming(llm):
    # concept
    response = llm.prompt("What is an ABAB rhyming scheme?")
    kbench.assertions.assert_contains_regex(
        r"first.*(third.*rhyme|rhyme.*third)",
        response.lower(),
        expectation="The model should explain that the first and third lines rhyme in an ABAB scheme.",
    )
    kbench.assertions.assert_contains_regex(
        r"second.*(fourth.*rhyme|rhyme.*fourth)",
        response.lower(),
        expectation="The model should explain that the second and fourth lines rhyme in an ABAB scheme.",
    )

    # editing
    response = llm.prompt(
        """
        Fill in the blank so that the resultant poem uses the ABAB rhyming scheme:

        Wondrous winter calls out
        Shivering under the frost
        Lies a lonely cat, sitting ?
        Alone but hardly lost

        Provide only the single missing word in your answer.
        """
    ).strip()
    kbench.assertions.assert_true(
        rhyme_words("out", response),
        expectation="The model should provide a word that rhymes with 'out' to complete the poem.",
    )


test_potemkin_abab_rhyming.run(kbench.llm)


# %%


def is_valid_triangle(a, b, c):
    """Checks if three side lengths can form a triangle."""
    try:
        a, b, c = float(a), float(b), float(c)
        if not (a > 0 and b > 0 and c > 0):
            return False
        return a + b > c and a + c > b and b + c > a
    except (ValueError, TypeError):
        return False


@kbench.task("triangle inequality")
def test_potemkin_triangle_inequality(llm):
    # concept
    response = llm.prompt("What is the triangle inequality theorem?")
    kbench.assertions.assert_contains_regex(
        r"(?=.*sum)(?=.*two sides)(?=.*third side)",
        response.lower(),
        expectation="The model's explanation of the triangle inequality theorem should mention that the sum of two sides is greater than the third side.",
    )

    # editing
    response = llm.prompt(
        """
        Name one length that could make the following a triangle.

        First side: 7
        Second side: ?
        Third side: 2

        Provide only the single number in your answer.
        """,
        schema=float,
    )
    kbench.assertions.assert_true(
        is_valid_triangle(7, 2, response),
        expectation="The model should provide a valid length for the second side that satisfies the triangle inequality.",
    )


test_potemkin_triangle_inequality.run(kbench.llm)


# %%
@kbench.task("uncle nephew relationship")
def test_potemkin_uncle_nephew_relationship(llm):
    # concept
    response = llm.prompt("What is the relationship between an uncle and a nephew?")
    kbench.assertions.assert_in(
        "brother",
        response.lower(),
        expectation="The model should correctly identify that an uncle is the brother of a parent.",
    )

    # classification
    response = llm.prompt(
        """
        Is there a uncle-nephew relationship in the following family structure?

        George is Paul's brother. Paul's wife is Paulina. George's mother is Rebecca.
        Rebecca's daughter is Justine.

        Provide your answer only as True or False.
        """,
        schema=bool,
    )
    kbench.assertions.assert_false(
        response,
        expectation="The model should correctly determine that there is no uncle-nephew relationship in the provided family structure.",
    )


test_potemkin_uncle_nephew_relationship.run(kbench.llm)


# %%
@kbench.task("haiku")
def test_potemkin_haiku(llm):
    # concept
    response = llm.prompt("What is a haiku?").lower()
    kbench.assertions.assert_in(
        "syllables",
        response,
        expectation="The model's definition of a haiku should mention syllables.",
    )
    kbench.assertions.assert_contains_regex(
        r"5[\s\S]+7[\s\S]+5",
        response,
        expectation="The model's definition of a haiku should include the 5-7-5 syllable structure.",
    )

    # classification
    response = (
        llm.prompt(
            """
        In the following text, what could fill in the blank so that the resultant text is a true haiku?


        ? main
        It's better for you to wane
        Let it float away

        Provide only the missing word in your answer.
        """
        )
        .strip()
        .lower()
    )
    kbench.assertions.assert_true(
        syllables.estimate(response) == 4,
        expectation="The model should provide a word with 4 syllables to complete the haiku's first line.",
    )


test_potemkin_haiku.run(kbench.llm)

# %%
