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
# title: Examples used in QUICK_START.md
# ---

# %%
import kaggle_benchmarks as kbench


# %%
@kbench.task(name="simple_riddle")
def solve_riddle(llm, riddle: str, answer: str):
    """Asks a riddle and checks for a keyword in the answer."""
    response = llm.prompt(riddle)

    # Assert that the model's response contains the answer, ignoring case.
    kbench.assertions.assert_contains_regex(
        f"(?i){answer}", response, expectation="LLM should give the right answer."
    )


# Execute the task
solve_riddle.run(
    llm=kbench.llm,
    riddle="What gets wetter as it dries?",
    answer="Towel",
)

# %%
from dataclasses import dataclass  # noqa: E402


@dataclass
class Person:
    """A dataclass to hold information about a person."""

    name: str
    age: int
    occupation: str


@kbench.task(name="extract_person_details")
def extract_details(llm, bio: str):
    """Extracts structured information from a biography into a dataclass."""
    prompt = f"Extract the name, age, and occupation from this text:\n\n{bio}"

    # The model will return an instance of the Person dataclass
    person = llm.prompt(prompt, schema=Person)

    kbench.assertions.assert_equal(
        "Marie Curie", person.name, expectation="LLM should give the right name."
    )
    kbench.assertions.assert_equal(
        66, person.age, expectation="LLM should give the right age."
    )
    kbench.assertions.assert_in(
        "physicist",
        person.occupation.lower(),
        expectation="LLM should recognize physicist in the occupation.",
    )


# Execute the task
extract_details.run(
    llm=kbench.llm,
    bio="Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity. She was born in 1867 and died in 1934 at the age of 66. She was the first woman to win a Nobel Prize.",
)


# %%
@kbench.task(name="describe_image")
def describe_image(llm, image_url: str, question: str, answer: str):
    """Sends an image and a question to a vision-capable model."""
    # Use a model that supports image inputs
    image = kbench.content_types.images.from_base64(
        kbench.content_types.images.image_url_to_base64(image_url)
    )
    kbench.user.send(image)
    response = llm.prompt(question)
    kbench.assertions.assert_contains_regex(
        f"(?i){answer}", response, expectation="LLM should give the right answer."
    )


describe_image.run(
    # You can also specify a model directly.
    llm=kbench.llms["google/gemini-2.5-flash"],
    image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg",
    question="What is the breed of the dog?",
    answer="Cavalier King Charles Spaniel",
)

# %%
@kbench.task(name="describe_video")
def describe_video(llm, video_url: str, question: str, answer: str):
    """Sends a video and a question to a video-capable model."""
    video = kbench.content_types.videos.from_url(video_url)
    response = llm.prompt(question, video=video)
    kbench.assertions.assert_contains_regex(
        f"(?i){answer}", response, expectation="LLM should give the right answer."
    )


describe_video.run(
    llm=kbench.llms["google/gemini-2.5-flash"],
    video_url="https://www.youtube.com/watch?v=aqz-KE-bpKQ",
    question="What is this video about?",
    answer="bunny",
)

# %%
from dataclasses import dataclass  # noqa: E402


@dataclass
class PoemEvaluation:
    """A structured evaluation of a poem."""

    score: float


@kbench.task(name="judge_poem_written_by_llm")
def judge_poem(question: str) -> float:
    """
    Has Gemini write a poem, then uses two judge LLMs to score it
    and returns the average score.
    """
    # The model to write poem
    gemini = kbench.llms["google/gemini-2.5-flash"]

    # two models to act as the judge
    judge_llm = kbench.llms["google/gemini-2.5-pro"]
    judge_llm2 = kbench.llms["meta/llama-3.1-70b"]

    # Generate poem
    with kbench.chats.new("gemini_chat"):
        gemini_poem = gemini.prompt(question)

    # Create a prompt for the judge, asking for a structured response
    judge_prompt = f"""
    You are a literary critic. An AI model, Gemini, has written a short poem based on the request: "{question}"

    Gemini's Poem: "{gemini_poem}"

    Please provide a score between 0 and 10 for the quality of this poem, where 0 is terrible and 10 is excellent.
    """

    # Get scores from both judges
    with kbench.chats.new("judge_chat_1"):
        evaluation1 = judge_llm.prompt(judge_prompt, schema=PoemEvaluation)

    with kbench.chats.new("judge_chat_2"):
        evaluation2 = judge_llm2.prompt(judge_prompt, schema=PoemEvaluation)

    # Assert scores are in valid range
    kbench.assertions.assert_true(
        0 <= evaluation1.score <= 10,
        expectation="Judge 1 score must be between 0 and 10.",
    )
    kbench.assertions.assert_true(
        0 <= evaluation2.score <= 10,
        expectation="Judge 2 score must be between 0 and 10.",
    )

    # Calculate average score
    average_score = (evaluation1.score + evaluation2.score) / 2

    kbench.assertions.assert_true(
        0 <= average_score <= 10,
        expectation="Average score must be between 0 and 10.",
    )

    return average_score


judge_poem.run(question="Write a 2-line poem about robots.")


# %%
@kbench.task(name="solve_with_python")
def solve_with_python(llm):
    """Asks the LLM to write and run Python code to solve a problem."""

    prompt = "What is the 15th Fibonacci number? Write a Python script to calculate and print it. Only print the final number."

    # Get the response from the model
    response = llm.prompt(prompt)

    # Extract the Python code from the model's response
    code_to_run = kbench.tools.python.extract_code(response)

    # Run the extracted code
    execution_result = kbench.tools.python.script_runner.run_code(code_to_run)

    # Assert that the code ran successfully and produced the correct output
    kbench.assertions.assert_empty(
        execution_result.stderr.strip(),
        expectation="The generated code should run without errors.",
    )
    code_output = execution_result.stdout.strip()
    kbench.assertions.assert_equal(
        "610",
        code_output,
        expectation=f"The code should print the 15th Fibonacci number ({code_output}), which should be equal to 610.",
    )


solve_with_python.run(llm=kbench.llm)

# %%
# Install and use extra dependencies.
# %pip install -q python-Levenshtein
from Levenshtein import distance  # noqa: E402


@kbench.task(name="fuzzy_spelling_check")
def check_spelling(llm):
    response = llm.prompt("What is the capital of France?")

    # Allow for minor spelling mistakes
    is_close_enough = distance(response.lower().strip(), "paris") <= 1
    kbench.assertions.assert_true(
        is_close_enough, f"Expected something close to 'paris', but got '{response}'"
    )


check_spelling.run(llm=kbench.llm)

# %%
import pandas as pd  # noqa: E402

data = {
    "riddle": [
        "I have cities, but no houses. I have mountains, but no trees. I have water, but no fish. What am I?",
        "What has an eye, but cannot see?",
        "What has to be broken before you can use it?",
    ],
    "answer_keyword": ["map", "needle", "egg"],
}
riddle_df = pd.DataFrame(data)


@kbench.task(name="evaluate_riddles")
def solve_and_check_riddle(llm, riddle: str, answer_keyword: str) -> bool:
    """A task that returns True if the model gets the riddle right."""
    response = llm.prompt(riddle)
    is_correct = answer_keyword.lower() in response.lower()
    kbench.assertions.assert_true(is_correct)
    return is_correct


# Evaluate the task on the DataFrame
# n_jobs > 1 will run evaluations in parallel
runs = solve_and_check_riddle.evaluate(
    llm=[kbench.llm],
    evaluation_data=riddle_df,
    n_jobs=3,
)

runs
# %%
# Summarize in a table.
runs.as_dataframe()


# %%
