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
# title: Monty Hall-like problems
# ---
# %%

from typing import Literal

from kaggle_benchmarks import llm, task


@task()
def monty_hall(llm):
    """
    Tests the LLM's understanding of the Monty Hall problem and the advantage of switching doors.
    """
    return llm.prompt(
        """
Suppose you're on a game show. You're given the choice of three doors:
Behind one door is a car; behind the others, goats.
You pick a door, say No. 1. The host, who knows what's behind the doors,
opens another door, say No. 3, revealing a goat.
He then says to you, "Do you want to switch to door No. 2?"
Is it to your advantage to switch your choice?
        """,
        schema=bool,
    )


monty_hall.run(llm)

# %%


@task()
def monty_hall_inverted(llm):
    """
    Tests the LLM's ability to recognize when the classic Monty Hall strategy doesn't apply
    due to a cost associated with switching.
    """
    answer = llm.prompt(
        """
Suppose you're on a game show. You're given the choice of three doors:
Behind one door is a goat; behind the others, cars.
You pick a door, say No. 1. The host, who knows what's behind the doors,
opens another door, say No. 3, revealing a goat.
He then says to you, "You can pay $1000 to switch to door No. 2?"
Is it to your advantage to switch your choice?
        """,
        schema=bool,
    )

    return not answer


monty_hall_inverted.run(llm)

# %%


@task()
def rephrased_monty_hall(llm):
    """
    Tests if the LLM can recognize the Monty Hall principle in a different context (escape room).
    """
    response = llm.prompt(
        """
You’re in an escape room with three locked doors. Only one leads to the exit.
You have a set of keys, and only one works.
You choose the first door. Before you try the key, the game master gives you a hint:
"The second door is definitely wrong."

Now you must decide:

A: Stick with your original choice (door one).
B: Switch to the third door.

What gives you the best chance of escaping?
        """,
        schema={"reasoning": str, "answer": Literal["A", "B"]},
    )

    return response.answer.lower() == "b"


rephrased_monty_hall.run(llm)

# %%


@task()
def blank_side(llm):
    """
    Tests the LLM's reasoning about conditional probability in a two-sided card puzzle.
    """
    answer = llm.prompt(
        """
You are given two cards: one with both sides blank and the other with one side blank
and the other side showing a number.
You randomly draw a card and place it on the table, revealing only the top side,
which is blank.
What is the probability that the bottom side of the card is also blank?
        """,
        schema={"reasoning": str, "probability": float},
    )

    return (answer.probability - 0.66) < 0.1


blank_side.run(llm)

# %%


@task()
def run_all(llm):
    benchmarks = [
        monty_hall,
        monty_hall_inverted,
        rephrased_monty_hall,
        blank_side,
    ]

    for i, b in enumerate(benchmarks):
        trial = b.run(llm)
        if not trial.result:
            return i

    return len(benchmarks)


# %%

run_all.run(llm)

# %%
