# Quick Start Guide

Welcome to the Kaggle Benchmarks Cookbook! This guide provides a
collection of “recipes” — practical examples and patterns to help you
get the most out of the library.

## Recipe: Defining a Simple Pass/Fail Task

The most fundamental task type is one that asserts a condition and
returns nothing. If all assertions pass, the task succeeds; otherwise,
it fails. This is perfect for simple Q&A checks.

``` python
import kaggle_benchmarks as kbench

@kbench.task(name="check_capital")
def check_capital(llm):
    # 1. Prompt the model
    response = llm.prompt("What is the capital of France?")

    # 2. Assert the correctness of the response
    # We use a regex for robust, case-insensitive matching.
    # For other supported assertions please see the Assertion Notebook linked below.
    kbench.assertions.assert_contains_regex(
        r"(?i)Paris", response, expectation="Model should answer Paris."
    )

check_capital.run(kbench.llm)
```

## Recipe: Using different types of assertions

Assertions are the core of the evaluation logic in kaggle-benchmarks. They allow you to validate the output of your Language Models (LLMs) against expected criteria. Each assertion returns an AssertionResult which tracks pass/fail status and provides detailed feedback.

- `assert_equal`: Checks if two values are exactly equal.
- `assert_true / assert_false`: Validates boolean conditions.
- `assert_in / assert_not_in`: Checks for membership in a container (string, list, dict, etc.). Great for keyword spotting or ensuring forbidden words are avoided.
- `assert_empty / assert_not_empty`: Checks if a container is empty or not. Useful for validating lists of errors or extracted entities.
- `assert_contains_regex / assert_not_contains_regex`: Uses regular expressions to search for patterns. Essential for validating structured formats like dates, emails, or specific code patterns.
- `assert_fail`: Use assert_fail to unconditionally signal a failure when a specific code path is reached, such as within exception handlers or the else block of complex logic checks.

Examples of assertions:
```python
import re
from kaggle_benchmarks import assertions, llm, task, utils

@task("code_validation_task")
def code_validation_task(llm):
    prompt = "Write a Python function called 'add' that takes two arguments and returns their sum. Please include a docstring."
    generated_code = llm.prompt(prompt)

    # 1. Check if a function is defined
    assertions.assert_in(
        "def ",
        generated_code,
        expectation="The generated code should contain a function definition."
    )

    # 2. Check for the correct function name
    assertions.assert_contains_regex(
        r"def add\s*\(",
        generated_code,
        expectation="The function should be named 'add'."
    )

    # 3. Check for a docstring
    assertions.assert_true(
        '"""' in generated_code or "'''" in generated_code,
        expectation="The function should include a docstring."
    )

    # 4. Ensure no dangerous keywords are present
    forbidden_words = ["exec", "eval"]
    for word in forbidden_words:
        assertions.assert_not_in(
            word,
            generated_code,
            expectation=f"The output should not contain the dangerous keyword: {word}"
        )

    # 5. Check if the function correctly adds two numbers
    try:
        # A safe way to test the function's logic without using exec
        assertions.assert_contains_regex(
            r"return.*?\+.*?",
            generated_code,
            expectation="The function should return the sum of its arguments."
        )
    except Exception as e:
        assertions.assert_fail(f"Could not validate the function's logic: {e}")


code_validation_task.run(llm)
```

## Recipe: Using a judge LLM

For complex, open-ended, or subjective tasks, simple assertions may not
suffice. You can leverage a “Judge” LLM to evaluate responses against a
list of criteria using `assess_response_with_judge`.

Note that unlike deterministic assertions, judge evaluations can be
subjective and may vary between runs.

``` python
import kaggle_benchmarks as kbench

@kbench.task(name="haiku_evaluation")
def haiku_evaluation(llm, topic):
    response = llm.prompt(f"Write a haiku about {topic}.")

    # Assess the response using a judge (can be the same LLM or a stronger one)
    assessment = kbench.assertions.assess_response_with_judge(
        criteria=[
            "The poem must have exactly 3 lines.",
            "The syllable structure must be 5-7-5.",
            f"The poem must be about {topic}.",
        ],
        response_text=response,
        judge_llm=llm
    )

    # Convert judge results into formal assertions
    for result in assessment.results:
        kbench.assertions.assert_true(
            result.passed,
            expectation=f"Criterion '{result.criterion}' failed: {result.reason}"
        )

haiku_evaluation.run(kbench.judge_llm, topic="AGI")
```

You can provide a custom prompt and output schema to tailor the judge’s
evaluation to your specific needs.

``` python
import dataclasses
import textwrap
from typing import Iterable

import kaggle_benchmarks as kbench

@dataclasses.dataclass
class StoryCritique:
    """A custom schema for receiving a story critique from the judge."""

    overall_rating: int  # A rating from 1 (poor) to 5 (excellent).
    feedback: str  # General feedback on the story.
    passed_checks: list[str]  # A list of criteria that the story successfully met.


def custom_story_prompt(criteria: Iterable[str], response_text: str) -> str:
    """A custom prompt function that instructs the judge to use the StoryCritique schema."""
    formatted_criteria = "\n".join(f"- {c}" for c in criteria)
    return textwrap.dedent(f"""
        You are a literary critic. Evaluate the following short story based on the provided criteria.

        **Short Story:**
        {response_text}

        **Evaluation Criteria:**
        {formatted_criteria}

        Provide your assessment in a JSON format with three fields:
        1. `overall_rating`: An integer from 1 to 5.
        2. `feedback`: A short paragraph of constructive feedback.
        3. `passed_checks`: A list of strings containing the criteria that were fully met.
    """)


@kbench.task()
def critique_short_story(llm):
    """This task demonstrates using a custom prompt and schema for the judge."""
    story = llm.prompt(
        "Write a one-paragraph short story about a robot who discovers music."
    )

    critique = kbench.assertions.assess_response_with_judge(
        criteria=[
            "The story is exactly one paragraph.",
            "The main character is a robot.",
            "The story is about the discovery of music.",
            "The story has a clear beginning, middle, and end.",
            "The story is inspired by a true story.",  # This should FAIL.
        ],
        response_text=story,
        judge_llm=kbench.judge_llm,
        prompt_fn=custom_story_prompt,
        output_schema=StoryCritique,
    )

    if critique is None:
        kbench.assertions.assert_fail(
            expectation="Judge LLM should respond with a critique."
        )
    else:
        # You can now add assertions based on your custom critique object.
        kbench.assertions.assert_true(
            critique.overall_rating >= 3,
            f"Story rating was {critique.overall_rating}, which is below the acceptable threshold of 3.",
        )
        kbench.assertions.assert_true(
            "The story is inspired by a true story." not in critique.passed_checks,
            "The critique incorrectly passed a failing criterion.",
        )


critique_short_story.run(kbench.judge_llm)
```

## Recipe: Returning a Numerical Score

To track granular metrics like accuracy or a specific score, your task
should return a value. You **must** add a return type annotation (e.g.,
`-> float`) so the leaderboard knows how to interpret the result.

``` python
@kbench.task(name="math_score")
def math_score(llm) -> float:
    # ... perform complex logic or multiple checks ...
    score = 0.85
    return score
```

The supported return types are:

- `None`: The default. The task succeeds if no assertions fail.
- `bool`: Returns True for success (Pass) and False for failure (Fail).
- `float`: Returns a continuous score (e.g., 0.0 to 1.0).
- `tuple[int, int]`: Returns a count of (passed, total) items.
- `tuple[float, float]`: Returns a (score, confidence_interval) pair.
- `dict`: Returns a dictionary of metrics. Dictionary metrics are NOT currently supported on leaderboard. But they can be used for detailed analysis as intermediate tasks, e.g. for dataset evaluation.

Here are some examples of different return types

```python
import kaggle_benchmarks as kbench

# Pass/fail task
@kbench.task(name="check_capital")
def check_capital(llm):
    response = llm.prompt("What is the capital of France?")
    # We use assert_contains_regex for a case-insensitive match.
    kbench.assertions.assert_contains_regex(
        r"(?i)Paris", response, expectation="Model should answer Paris."
    )

# Bool task
@kbench.task(name="solve_riddle")
def solve_riddle(llm) -> bool:
    riddle = "I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?"
    response = llm.prompt(f"Solve this riddle: {riddle}")
    return "echo" in response.lower()

# Float (score) task
@kbench.task(name="conciseness_score")
def conciseness_score(llm) -> float:
    text = "The quick brown fox jumps over the lazy dog." * 5
    # Task: Summarize in exactly 5 words
    response = llm.prompt(f"Summarize this text in exactly 5 words: {text}")

    word_count = len(response.split())
    # Score: 1.0 if perfect, penalize deviation
    error = abs(word_count - 5)
    return max(0.0, 1.0 - (error / 5.0))

# Tuple[int, int] (Count) Task
@kbench.task(name="math_quiz")
def math_quiz(llm) -> tuple[int, int]:
    questions = {"2+2": "4", "3*3": "9", "10/2": "5"}
    correct = 0
    for q, a in questions.items():
        if a in llm.prompt(f"What is {q}?"):
            correct += 1
    return correct, len(questions)

# Tuple[float, float] (Metric ± Confidence) Task
import statistics

@kbench.task(name="memorization_test")
def memorization_test(llm) -> tuple[float, float]:
    # Run 5 trials to test stability
    scores = []
    for _ in range(5):
        seq = "7-9-2-5-1"
        response = llm.prompt(f"Repeat this sequence: {seq}")
        # Score 1.0 for exact match, 0.0 otherwise
        scores.append(1.0 if seq in response else 0.0)

    # Return mean accuracy and standard deviation
    return statistics.mean(scores), statistics.stdev(scores)

# Dict (Detailed Metrics) Task
@kbench.task(name="detailed_analysis")
def detailed_analysis(llm) -> dict:
    response = llm.prompt("Write a haiku about coding.")
    return {
        "char_count": len(response),
        "word_count": len(response.split()),
        "has_keywords": "code" in response.lower()
    }

```

## Recipe: Enforcing Structured Output with Schemas

You can force the LLM to return data in a specific structure by
providing a `schema` to the `prompt` method. This is essential for tasks
that require structured output.

**Currently the only supported schema types are: int, bool, dict, dataclass, and pydantic BaseModel. We recommend using dataclass or pydantic BaseModel for customized structures.**

**DO NOT use `list`, `List`, or `list[...]` as a schema.** If you need to return a list of items (like a list of names or planets), you must wrap it in a `dataclass`, or `BaseModel`.
* **Incorrect:** `llm.prompt(..., schema=list[str])`
* **Correct:** `llm.prompt(..., schema=dict)` (and ask for a dictionary containing the list) or define a `dataclass`/`BaseModel` with a list field.

``` python
from dataclasses import dataclass

@dataclass
class MovieReview:
    sentiment: str
    score: int

@kbench.task(name="analyze_review")
def analyze_review(llm):
    # The model will return an instance of MovieReview, not a string.
    result = llm.prompt(
        "Analyze this review: 'Fantastic movie!'. Respond with a valid JSON.",
        schema=MovieReview
    )
    print(f"Score: {result.score}, Sentiment: {result.sentiment}")
```

Please note different models have different capabilities to support structured output. Try use alternatives if one doesn't work for your use case. Here are some examples of different structure output.
```python
# Native Python Types as Output Schema
@kbench.task(name="extract_year")
def extract_year(llm):
    """Extracts a year as an integer."""
    text = "The Apollo 11 mission landed on the Moon in 1969. Respond with only the integer number."
    # Force the model to return an integer
    year = llm.prompt(f"Extract the year from this text: '{text}'", schema=int)

    kbench.assertions.assert_equal(
        1969, year, expectation="Extracted year should be 1969."
    )

# Dictionary as Output Schema
@kbench.task(name="extract_person")
def extract_person(llm):
    """Extracts basic person details into a dictionary."""
    text = "Contact info: John Doe, age 42, works as a Software Engineer."

    # Define schema as a dictionary of types
    person_schema = {
        "name": str,
        "age": int,
        "occupation": str
    }

    person = llm.prompt(
        f"Extract person details from: '{text}'. Respond with a valid JSON.",
        schema=person_schema
    )

    kbench.assertions.assert_equal(
        "John Doe", person.name, expectation="Name should be John Doe."
    )

# Dataclasses as Output Schema
from dataclasses import dataclass

@dataclass
class RPGCharacter:
    name: str
    class_type: str
    level: int
    inventory: str

@kbench.task(name="generate_character")
def generate_character(llm):
    """Generates an RPG character using a dataclass schema."""
    character = llm.prompt(
        "Generate a level 5 wizard character for a fantasy game. Respond with a valid JSON.",
        schema=RPGCharacter
    )

    kbench.assertions.assert_true(
        len(character.name) > 0, expectation="Character should have a name."
    )

# Pydantic class as Output Schema
from pydantic import BaseModel, Field


class Planet(BaseModel):
    name: str
    mass_earth_masses: float = Field(description="Mass relative to Earth")
    has_life: bool = Field(description="Whether the planet is known to have life")
    moons: list[str] = Field(default_factory=list, description="List of major moons")

@kbench.task(name="planet_info")
def planet_info(llm):
    """Retrieves planet information using a Pydantic model."""
    planet = llm.prompt(
        "Provide information about the planet Jupiter. Respond with a valid JSON.",
        schema=Planet
    )

    kbench.assertions.assert_contains_regex(
        r"(?i)jupiter", planet.name, expectation="Planet name should be Jupiter."
    )
```

## Recipe: Evaluating Performance on a Dataset

Instead of a single run, you often want to evaluate a task over many
examples. Use the `.evaluate()` method to run your task against every
row in a pandas DataFrame.

``` python
import pandas as pd
import kaggle_benchmarks as kbench

# 1. Prepare your dataset
df = pd.DataFrame([
    {"question": "2+2", "answer": "4"},
    {"question": "Capital of UK", "answer": "London"},
])

# 2. Define a task that accepts row columns as arguments
@kbench.task(name="single_qa_task", store_task=False)
def single_qa_task(llm, question, answer) -> dict:
    """Evaluates the model on a single question-answer pair."""
    response = llm.prompt(question)
    return {
        "question": question,
        "gold_target": answer,
        "predicted_answer": response,
        "is_correct": answer.lower() in response.lower(),
    }


# 3. Define a task to evaluate the whole dataset
@kbench.task(name="multi_qa_task")
def multi_qa_task(llm, df) -> tuple[float, float]:
    """Runs the evaluation on the entire dataset and returns accuracy and std."""

    # Optionally write code in `enable_cache` context to cache results
    with kbench.client.enable_cache():
        # .evaluate() runs the task in parallel (up to n_jobs)
        # It automatically iterates over the rows of `evaluation_data`.
        runs = single_qa_task.evaluate(
            stop_condition=lambda runs: len(runs) == df.shape[0],
            max_attempts=1,
            retry_delay=15,
            llm=[llm],
            evaluation_data=df,
            n_jobs=2,
            timeout=120,
            remove_run_files=True,  # Optionally remove sub runs files to save space.
        )

    # Convert the results to a DataFrame for easy analysis.
    eval_df = runs.as_dataframe()

    # Calculate aggregate metrics.
    # We extract the 'is_correct' field from the result dictionary of each run.
    accuracy = float(eval_df.result.str.get("is_correct").mean())
    std = float(eval_df.result.str.get("is_correct").std())

    print(f"Accuracy: {accuracy:.2f}, Std: {std:.2f}")
    return accuracy, std

# 4. Run evaluation
run = multi_qa_task.run(kbench.llm, df)
run
```

## Recipe: Comparing Multiple Models Side-by-Side

Typically, you should write your task for a single LLM using the
`kbench.llm` placeholder. This allows you to schedule runs across
multiple models using “Add Models” button on the Kaggle Task Detail page
without changing your code.

However, if you need to compare models directly within your notebook
(e.g., for debugging or immediate visualization), you can pass a list of
LLMs to `.evaluate()` to run them all.

``` python
models = [
    kbench.llms["google/gemini-2.5-flash"],
    kbench.llms["meta/llama-3.1-70b"]
]

# Runs the evaluation for BOTH models
results = solve_question.evaluate(llm=models, evaluation_data=df)
```

## Recipe: Leveraging Automatic Conversation History

By default, `llm.prompt()` maintains conversation history within the
same session. This makes multi-turn conversations natural and easy to
implement.

``` python
@kbench.task()
def chat_task(llm):
    # Turn 1
    llm.prompt("Hi, I'm looking for a book.")

    # Turn 2: The model "remembers" the previous turn automatically.
    llm.prompt("I like science fiction.")
```

## Recipe: Creating Isolated Conversations for Judges

Sometimes you need a side-conversation that shouldn’t be seen by the
main agent—for example, when a “Judge” LLM is evaluating the main
agent’s performance. Use `kbench.chats.new()` to create a clean slate.

``` python
@kbench.task()
def game_task(llm, judge_llm):
    # Main conversation with the player
    llm.prompt("Player move...")

    # Isolated conversation for the judge
    with kbench.chats.new("judging"):
        # This prompt is NOT added to the 'llm' history
        judge_llm.prompt("Did the player make a valid move?")
```

Here is another example showing how to benchmark a 20 question using isolated conversations.

``` python
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
def play_20(llm, judge_llm, category: str, target: str) -> bool:
    """Checks LLMs ability to play 20 question game."""

    rules = f"""
Let's play 20 questions! I'm thinking of {category}.
You have 20 questions to guess what it is.
Ask me yes or no questions, about anything you want.
"""
    # 1. Conversation Management:
    # When `llm.prompt` is called multiple times on the same `llm` object,
    # the conversation history is automatically preserved.
    # The LLM "remembers" previous turns (rules, questions, answers).
    response = llm.prompt(rules, schema=Response)

    for i in range(20):
        if response.guess:
            assertions.assert_in(
                target,
                response.guess.lower(),
                f"Guessed '{response.guess}' is incorrect. The right answer was '{target}'.",
            )
            return True

        # 2. Isolated Conversations:
        # We use `chats.new` to create a *temporary, isolated* conversation context.
        # This is crucial here because the Judge LLM knows the secret `target` word.
        # If we didn't isolate this, or if we used the main `llm` for judging,
        # the secret word might leak into the main conversation history,
        # ruining the game.
        with chats.new("Checking the question with another LLM"):
            yes = judge_llm.prompt(
                f"""
I'm playing 20 questions with someone.
I'm thinking of a {target}.
Here's their question: {response.question}.""",
                schema=bool,
            )

        answer = "Yes" if yes else "No"

        # Continue the main conversation. The history now includes:
        # - Rules
        # - Previous Q&A pairs
        # - The new question (from the previous `llm.prompt` call)
        # - This new answer
        response = llm.prompt(
            f"My answer is {answer}. Are you ready to guess or would like to ask another question (you have {19 - i} left)?",
            schema=Response,
        )

    system.send(f"Failed to guess `{target}` in 20 questions.")
    return False


play_20.run(llm, llm, category="an animal", target="dog")
```

## Recipe: Managing Multi-Agent Conversations

In complex multi-agent scenarios, you often need to maintain separate
conversation histories for each agent. The *Dungeon Adventure* example
demonstrates how to do this by creating a dedicated `Chat` object for
each agent and using the `contexts.enter()` context manager to switch
between them. This allows each agent to have its own isolated
conversation history, which is crucial for role-playing and other
complex interactions.

``` python
from kaggle_benchmarks import chats, contexts, LLMChat, task


@task()
def multi_chat_task(llm):
    # Create two separate, isolated chat contexts
    chat1 = chats.Chat(name="First Conversation")
    chat2 = chats.Chat(name="Second Conversation")

    # Get the main chat to add our sub-chats to for visualization
    current_chat = chats.get_current_chat()
    current_chat.append(chat1)
    current_chat.append(chat2)

    # --- Interaction 1 ---
    # Enter the context of the first chat
    with contexts.enter(chat=chat1):
        # Any LLM calls or messages sent here will be recorded in 'chat1'
        # This message is in the first conversation.
        llm.prompt("Hello from chat 1")

    # --- Interaction 2 ---
    # Now, enter the context of the second chat
    with contexts.enter(chat=chat2):
        # Any LLM calls or messages sent here will be recorded in 'chat2'
        # This message is now in the completely separate second conversation.
        llm.prompt("Hello from chat 2")

    # --- Interaction 3 ---
    # We can switch back to the first chat at any time
    with contexts.enter(chat=chat1):
        # We are back in the first conversation.
        llm.prompt("Still in chat 1")

multi_chat_task.run(kbench.llm)
```

## Recipe: Sending Images to Multimodal Models

You can send images to vision-capable models using either a direct URL
or a Base64 string.

``` python
from kaggle_benchmarks.content_types import images
import kaggle_benchmarks as kbench

# Send image as an URL
@kbench.task()
def vision_task(llm):
    # Load image from a URL
    img = images.from_url("https://www.kaggle.com/static/images/site-logo.png")

    # Pass the image alongside the text prompt
    response = llm.prompt("Describe this image.", image=img)

# Send image as base64 string
@kbench.task("Describe Image (Base64)")
def describe_image_base64(llm):
    """Sends a base64 encoded image with explicit format specification."""

    image_b64 = images.image_url_to_base64("https://www.kaggle.com/static/images/site-logo.png")

    # Create Image object from Base64, specifying the format as 'png'
    # The 'format' parameter is important when the image is not a JPEG (default)
    image = images.from_base64(image_b64, format="png")

    response = llm.prompt("What color is this image?", image=image)

    kbench.assertions.assert_contains_regex(
        r"(?i)blue",
        response,
        expectation="LLM should identify the color blue.",
    )
```

## Recipe: Writing Reusable Custom Assertions

Keep your code clean by encapsulating complex checks into reusable
assertion functions using the `@assertion_handler` decorator.

``` python
from kaggle_benchmarks.assertions import assertion_handler, AssertionResult

@assertion_handler()
def assert_is_even(value: int, expectation: str) -> AssertionResult:
    # Return a structured result object
    return AssertionResult(
        passed=(value % 2 == 0),
        expectation=expectation
    )

@kbench.task()
def even_task(llm):
    num = int(llm.prompt("Pick a not odd number", schema=int))
    # Use your custom assertion just like built-in ones
    assert_is_even(num, expectation="Number should be even")
```


## Recipe: Using the Built-in Python Script Runner

We provide some helper functions to make it easy for the LLM to write
and execute Python code to solve problems. The library provides tools to
extract and run this code safely.

``` python
@kbench.task()
def python_task(llm):
    prompt = "Calculate the 10th Fibonacci number using Python."
    response = llm.prompt(prompt)

    # 1. Extract code from the markdown response
    code = kbench.tools.python.extract_code(response)

    # 2. Run the code
    result = kbench.tools.python.script_runner.run_code(code)

    # 3. Verify the output
    kbench.assertions.assert_contains_regex("55", result.stdout)
```

A complete example of asking LLM to generate Python code for the Tower of Hanoi problem is like this.

```python
import kaggle_benchmarks as kbench
import re

@kbench.task(name="hanoi_python_solver")
def hanoi_python_solver(llm):
    """
    Asks the LLM to generate Python code for the Tower of Hanoi problem
    and verifies its correctness by executing it.
    """
    prompt = """
    Write a Python function that solves the Tower of Hanoi problem.
    The function should be named `hanoi` and take four arguments:
    1. `n`: the number of disks
    2. `source`: the source peg
    3. `auxiliary`: the auxiliary peg
    4. `destination`: the destination peg

    The function should print each move required to solve the puzzle.
    For example: 'Move disk 1 from A to C'.

    Respond with function defintion only.
    """
    response = llm.prompt(prompt)

    # Extract the Python code block from the LLM's response.
    code = kbench.tools.python.extract_code(response)
    kbench.assertions.assert_not_empty(
        code,
        expectation="Response should contain a Python code block."
    )

    # Add driver code to call the generated function for a 3-disk problem.
    driver_code = "\nhanoi(n=3, source='A', auxiliary='B', destination='C')"
    full_code = code + driver_code

    # Execute the combined code.
    result = kbench.tools.python.script_runner.run_code(full_code)

    # Assert that the code ran successfully without any errors.
    kbench.assertions.assert_empty(
        result.stderr,
        expectation=f"The generated Python code should run without errors. Stderr: {result.stderr}"
    )

    # For n=3 disks, the solution always takes 2^3 - 1 = 7 moves.
    # We count the number of "Move disk" lines in the output.
    move_count = len(re.findall(r"(?i)move disk", result.stdout))
    kbench.assertions.assert_equal(
        move_count, 7,
        expectation="The solution for 3 disks must contain exactly 7 moves."
    )

    # Check for the crucial move of the largest disk from source to destination.
    kbench.assertions.assert_contains_regex(
        r"(?i)move disk 3 from A to C",
        result.stdout,
        expectation="The largest disk (3) must move from the source (A) to the destination (C) at some point."
    )

hanoi_python_solver.run(kbench.llm)
```

## Recipe: Equipping Models with Custom Tools

You can also pass your own Python functions as tools. The model can then
call these functions to retrieve information or perform actions. Please
note this is currently an experimental feature.

``` python
def get_weather(city: str) -> str:
    """Returns the weather for a city."""
    return "Sunny"

@kbench.task()
def weather_task(llm):
    # Pass the function directly to the 'tools' argument
    # Note: Works best with 'genai' API models
    llm.prompt("Weather in Tokyo?", tools=[get_weather])

# This feature currently only works with 'genai' API models
llm= models.load_model(
    model_name=kbench.llm.name,
    api="genai",
)

weather_task.run(llm)
```
