# Copyright 2026 Kaggle Inc.
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
# title: Example of a game that requires tool use.
# ---

# %%
import random

import kaggle_benchmarks as kbench
from kaggle_benchmarks.kaggle import models

SECRET_NUMBER = random.randint(1, 10)


def guess_number(guess: int) -> str:
    """Make a guess in the number guessing game."""
    if guess < SECRET_NUMBER:
        return "Higher"
    elif guess > SECRET_NUMBER:
        return "Lower"
    else:
        return "Correct!"


@kbench.task(name="guess-the-number-game")
def play_game(llm):
    prompt = "I'm thinking of a number between 1 and 10. Can you guess it?"
    response = llm.prompt(prompt, schema=int, tools=[guess_number])

    for _ in range(4):
        if response == SECRET_NUMBER:
            break
        response = llm.prompt(response, schema=int, tools=[guess_number])

    kbench.assertions.assert_equal(
        SECRET_NUMBER,
        response,
        expectation=f"LLM should have guessed the secret number. The secret number was {SECRET_NUMBER}",
    )


# %%

llm_with_genai_api = models.load_model(
    model_name=kbench.llm.name,
    api="genai",
)

play_game.run(llm=llm_with_genai_api)

# %%

llm_with_openai_api = models.load_model(
    model_name=kbench.llm.name,
    api="openai",
)

play_game.run(llm_with_openai_api)

# %%
