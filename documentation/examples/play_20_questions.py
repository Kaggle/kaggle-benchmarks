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
# title: Twenty questions
# ---
# %%

from dataclasses import dataclass

import pandas as pd

from kaggle_benchmarks import assertions, chats, judge_llm, llm, llms, system, task


@dataclass
class Response:
    question: str = ""
    guess: str = ""
    reasoning: str = ""

    def _repr_markdown_(self):
        return f"*{self.reasoning}*\n\n{self.question or ''}\n\n{self.guess or ''}"


@task(name="Twenty Questions")
def play_20(llm, judge_llm, category: str, target: str):
    """Checks LLMs ability to play 20 question game."""

    rules = f"""
Let's play 20 questions! I'm thinking of {category}.
You have 20 questions to guess what it is.
Ask me yes or no questions, about anything you want.
"""
    response = llm.prompt(rules, schema=Response)

    for i in range(20):
        if response.guess:
            assertions.assert_in(
                target,
                response.guess.lower(),
                f"Guessed '{response.guess}' is incorrect. The right answer was '{target}'.",
            )
            return

        else:
            with chats.new("Answering the question with LLM"):
                yes = judge_llm.prompt(
                    f"""
I'm playing 20 questions with someone.
I'm thinking of a {target}.
Here's their question: {response.question}.""",
                    schema=bool,
                )

            answer = "Yes" if yes else "No"
            response = llm.prompt(
                f"My answer is {answer}. Are you ready to guess or would like to ask another question (you have {19 - i} left)?",
                schema=Response,
            )

    system.send(f"Failed to guess `{target}` in 20 questions.")
    return False


# %%
play_20.run(llm, llm, category="an animal", target="dog")

# %%

play_20.evaluate(
    llm=[
        llms["google/gemini-2.0-flash"],
        llms["google/gemini-1.5-pro"],
    ],
    judge_llm=[judge_llm],
    evaluation_data=pd.DataFrame(
        {
            "category": ["an animal"] * 4,
            "target": ["goose", "llma", "ciconia", "bison"],
        }
    ),
)

# %%
