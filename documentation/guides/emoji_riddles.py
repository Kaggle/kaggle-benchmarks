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
# title: "LLMs: Emojis"
# ---
# # Introduction to Benchmarking with Emoji Riddles
#
# ---

# %%
import emoji

from kaggle_benchmarks import chats, llm, system, task

# %% [markdown]
# ## Basic LLM Interaction
#
# The `llm.prompt()` function sends a prompt to the LLM and returns its response. Let's try solving a simple emoji riddle:
# %%

response = llm.prompt("Solve emoji riddle: 🦁🪄🚪?")
print(response)

# %% [markdown]
# ## Verifying the Answer
#
# We can check if the LLM's response contains the correct answer:
# %%

answer = "The Lion, the Witch and the Wardrobe"
print(answer in response)

# %% [markdown]
# However, directly comparing strings can be unreliable.
# LLMs might add extra punctuation, use different capitalization, or rephrase the answer slightly.
# A more robust approach is to use a "fuzzy match":
# %%


def fuzzy_match(pattern: str, text: str) -> bool:
    """Checks if a pattern is present in a text, ignoring case and non-alphabetic characters."""
    pattern = "".join(filter(str.isalpha, pattern))
    text = "".join(filter(str.isalpha, text))
    return pattern.lower() in text.lower()


assert fuzzy_match(
    answer,
    "My answer is: the lion the witch and the wardrobe",
)

# %% [markdown]
# ## Multi-Turn Conversations with `chats.new()`
#
# For more complex interactions, use `chats.new()` to create a conversation context.
# This allows you to send multiple messages and automatically track the conversation's history.
# %%

with chats.new("Riddle Chat") as chat:
    r = llm.prompt("Riddle: 🦁🪄🚪?")
    if not fuzzy_match(answer, r):
        r = llm.prompt("Hint: It's a famous book.")

    # use system to report useful information
    system.send(f"Riddle solved: {'✅ Yes' if fuzzy_match(answer, r) else '❌ No'}")

chat

# %% [markdown]
# ## Defining task with `@task()`
#
# The `@task()` decorator simplifies defining and running conversation simulations.
# It automatically handles conversation setup, result capture, and assertion processing.
# It also allows you to add metadata like name, description, and tags.
# %%


@task(name="Emoji riddle")
def solve_riddle(llm, riddle: str, hint: str, answer: str):
    """Task for emoji riddle."""
    # no need to use chats.new as task.run provides its own chat
    r = llm.prompt(f"riddle: {riddle}?")
    if not fuzzy_match(answer, r):
        r = llm.prompt(f"Hint: {hint}")
    # asserts can be used as well
    assert fuzzy_match(answer, r), f"Correct answer is {answer}"


# %% [markdown]
# ## Running a task
#
# The `solve_riddle.run()` method executes the task, managing the conversation and assertions.
# %%

solve_riddle.run(llm, riddle="🌍🥛", answer="World Cup", hint="🏆")

# %% [markdown]
# ## Generating and Solving Riddles
#
# This example demonstrates how to make the LLM create a riddle and then solve it.
# %%


def only_emojis(text):
    text = emoji.emojize(text)
    return "".join(x["emoji"] for x in emoji.emoji_list(text))


@task(name="Riddle Generation and Solving")
def riddle_generation_and_solving(llm, movie_title: str):
    """Task to test LLM's ability to generate an emoji riddle and then solve it."""

    system.send("Please ONLY provide the emoji riddle itself, without any explanation.")
    riddle = llm.prompt(
        f"Create a short emoji riddle puzzle that represents the movie: '{movie_title}'. ",
    )

    # create a new chat to hide previous interactions
    with chats.new("Self-Solving Chat"):
        solving_response = llm.prompt(
            f"Can you solve this emoji riddle puzzle: {only_emojis(riddle)}?"
        )

        assert fuzzy_match(movie_title, solving_response), (
            f"The answer is {movie_title}"
        )


riddle_generation_and_solving.run(llm, movie_title="Matrix")

# %% [markdown]
# ## Task Groups
#
# Multiple taks can be combined into a single "batch" task by nesting them.
# This is useful for evaluating the LLM on a set of related tasks.
# %%

tasks = [
    {
        "riddle": "🧙‍♂️💍💍💍💍💍💍💍💍",
        "answer": "lord of the rings",
        "hint": "Epic fantasy saga",
    },
    {
        "riddle": "⭐🌙🌃🌌",
        "answer": "starry night",
        "hint": "👂🌻",
    },
    {
        "riddle": "🍵🚢🇺🇸",
        "answer": "boston tea party",
        "hint": "historical event",
    },
]


@task(name="Emoji riddle")
def riddle_task_group(llm, tasks, movies):
    """Evaluates LLM on several emoji riddles and riddle generation."""
    for t in tasks:
        assert solve_riddle.run(llm, **t).result

    for movie in movies:
        riddle_generation_and_solving.run(llm, movie_title=movie)


riddle_task_group.run(llm, tasks, movies=["Titanic", "Matrix"])
# %%
