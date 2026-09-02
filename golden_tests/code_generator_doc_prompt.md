# Quick Start Guide

Welcome to the Kaggle Benchmarks Cookbook! This guide provides a
collection of “recipes” — practical examples and patterns to help you
get the most out of the library.

## Recipe: Defining a Simple Pass/Fail Task

The most fundamental task type is one that asserts a condition and
returns nothing. If all assertions pass, the task succeeds; otherwise,
it fails. This is perfect for simple Q&A checks. By default, assertions are non-halting, meaning all assertions in the task will run and record their results, and the task fails if any assertion failed.

``` python
import kaggle_benchmarks as kbench


@kbench.task(name="check_capital")
def check_capital(llm):
    # 1. Prompt the model
    response = llm.prompt("What is the capital of France?")

    # 2. Assert the correctness of the response
    # We use a regex for robust, case-insensitive matching.
    # Signature: assert_contains_regex(pattern: str | re.Pattern, text: str, expectation: str | None)
    kbench.assertions.assert_contains_regex(
        r"(?i)Paris", response, expectation="Model should answer Paris."
    )


check_capital.run(kbench.llm)
```

## Recipe: Using different types of assertions

Assertions are the core of the evaluation logic in kaggle-benchmarks. They allow you to validate the output of your Language Models (LLMs) against expected criteria. Each assertion returns an `AssertionResult` dataclass (containing `passed`, `expectation`, `actual`, and `expected` fields).

**Built-in Assertion Signatures:**
- `assert_true(expr: bool, expectation: str | None = None)`
- `assert_false(expr: bool, expectation: str | None = None)`
- `assert_equal(expected: Any, actual: Any, expectation: str | None = None)`
- `assert_in(member: Any, container: Any, expectation: str | None = None)`
- `assert_not_in(member: Any, container: Any, expectation: str | None = None)`
- `assert_empty(container: Any, expectation: str | None = None)`
- `assert_not_empty(container: Any, expectation: str | None = None)`
- `assert_contains_regex(pattern: str | re.Pattern[str], text: str, expectation: str | None = None)`
- `assert_not_contains_regex(pattern: str | re.Pattern[str], text: str, expectation: str | None = None)`
- `assert_raises_no_exceptions(callable_obj: Callable, expectation: str | None = None, *args, **kwargs)`
- `assert_fail(expectation: str | None = None)`: Unconditionally record a failure.

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
        expectation="The generated code should contain a function definition.",
    )

    # 2. Check for the correct function name
    assertions.assert_contains_regex(
        r"def add\s*\(",
        generated_code,
        expectation="The function should be named 'add'.",
    )

    # 3. Ensure no dangerous keywords are present
    forbidden_words = ["exec", "eval"]
    for word in forbidden_words:
        assertions.assert_not_in(
            word,
            generated_code,
            expectation=f"The output should not contain the dangerous keyword: {word}",
        )

    # 4. Safely test logic execution wrapper
    try:
        assertions.assert_contains_regex(
            r"return.*?\+.*?",
            generated_code,
            expectation="The function should return the sum of its arguments.",
        )
    except Exception as e:
        assertions.assert_fail(f"Could not validate the function's logic: {e}")


code_validation_task.run(llm)
```

## Recipe: Using a judge LLM

For complex, open-ended, or subjective tasks, simple assertions may not
suffice. You can leverage a “Judge” LLM to evaluate responses against a
list of criteria using `assess_response_with_judge`.

**Note:** Any exception during the judge invocation causes the function to return `None`. You must check for `None` before iterating over the results. The default return type is an `AssessReport` containing a list of `AssessResult` objects (with fields: `criterion`, `passed`, `reason`, `confidence`).

``` python
import kaggle_benchmarks as kbench


@kbench.task(name="haiku_evaluation")
def haiku_evaluation(llm, topic):
    response = llm.prompt(f"Write a haiku about {topic}.")

    # Assess the response using a judge
    # Signature: assess_response_with_judge(criteria: Iterable[str], response_text: str, judge_llm: Any, prompt_fn=None, output_schema=AssessReport)
    assessment = kbench.assertions.assess_response_with_judge(
        criteria=[
            "The poem must have exactly 3 lines.",
            "The syllable structure must be 5-7-5.",
            f"The poem must be about {topic}.",
        ],
        response_text=response,
        judge_llm=kbench.judge_llm,
    )

    # Handle judge failure gracefully
    if assessment is None:
        kbench.assertions.assert_fail(
            expectation="Judge LLM failed to return a valid assessment."
        )
        return

    # Convert judge results into formal assertions
    for result in assessment.results:
        kbench.assertions.assert_true(
            result.passed,
            expectation=f"Criterion '{result.criterion}' failed: {result.reason}",
        )


haiku_evaluation.run(kbench.judge_llm, topic="AGI")
```

You can provide a custom prompt function and output schema to tailor the judge’s
evaluation to your specific needs. The custom prompt function must accept `(criteria: Iterable[str], response_text: str) -> str`.

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
    """)


@kbench.task()
def critique_short_story(llm):
    story = llm.prompt(
        "Write a one-paragraph short story about a robot who discovers music."
    )

    critique = kbench.assertions.assess_response_with_judge(
        criteria=[
            "The story is exactly one paragraph.",
            "The main character is a robot.",
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
        kbench.assertions.assert_true(
            critique.overall_rating >= 3,
            expectation=f"Story rating was {critique.overall_rating}, which is below the acceptable threshold of 3.",
        )
        kbench.assertions.assert_not_in(
            "The story is inspired by a true story.",
            critique.passed_checks,
            expectation="The critique incorrectly passed a failing criterion.",
        )


critique_short_story.run(kbench.judge_llm)
```

## Recipe: Returning a Numerical Score

To track granular metrics like accuracy or a specific score, your task
should return a value. You **must** add a Python return type annotation to your function signature so the framework knows how to correctly serialize the result into the `BenchmarkTaskRun` protobuf.

Supported return type annotations:
- `None` (or no annotation): Task succeeds if no assertions fail.
- `-> bool`: Returns True for success (Pass) and False for failure (Fail).
- `-> int` / `-> float`: Returns a direct numeric score. It is best practice to normalize this to `[0.0, 1.0]`.
- `-> tuple[int, int]`: Returns a count of `(successes, total)` attempts.
- `-> tuple[float, float]`: Returns a `(score, confidence_interval)` pair.

```python
import kaggle_benchmarks as kbench


# Float (score) task
@kbench.task(name="conciseness_score")
def conciseness_score(llm) -> float:
    text = "The quick brown fox jumps over the lazy dog." * 5
    response = llm.prompt(f"Summarize this text in exactly 5 words: {text}")

    word_count = len(response.split())
    error = abs(word_count - 5)
    return max(0.0, 1.0 - (error / 5.0))  # Normalized between 0 and 1


# Tuple[int, int] (Count) Task
@kbench.task(name="math_quiz")
def math_quiz(llm) -> tuple[int, int]:
    questions = {"2+2": "4", "3*3": "9", "10/2": "5"}
    correct = 0
    for q, a in questions.items():
        if a in llm.prompt(f"What is {q}?"):
            correct += 1
    return correct, len(questions)
```

## Recipe: Enforcing Structured Output with Schemas

You can force the LLM to return data in a specific structure by
providing a `schema` to the `prompt` method. The framework will automatically handle generating the JSON schema, parsing the LLM response, and returning a typed Python object.

**Supported Schema Types:**
- Primitives: `int`, `float`, `bool`, `str`, `datetime.datetime` (Note: `str` returns the raw text response, skipping JSON parsing).
- Structures: `TypedDict`, `dataclasses.dataclass`, and `pydantic.BaseModel`.
*(Note: Do not pass generic `list` directly as a schema; wrap lists inside a dataclass or BaseModel).*

**Handling Errors:** Always wrap structured calls in a `try/except` block for `ResponseParsingError` if reliability is critical.

```python
import kaggle_benchmarks as kbench
from kaggle_benchmarks.prompting import ResponseParsingError
from pydantic import BaseModel, Field


# 1. Define the new schema with a list of strings
class LanguageList(BaseModel):
    languages: list[str] = Field(
        default_factory=list, description="A list of programming language names."
    )


@kbench.task(name="list_programming_languages")
def list_programming_languages(llm):
    # 2. Update the prompt to match the new topic and schema
    prompt = "List 5 of the most popular programming languages today. Your response must be a JSON object with a single key 'languages' which holds a list of the language names as strings."

    try:
        response = llm.prompt(prompt, schema=LanguageList)

        # 3. Normalize the output for a safer assertion check
        languages_lower = [lang.lower() for lang in response.languages]

        # 4. Assert that "python" made it into the LLM's list
        kbench.assertions.assert_in(
            "python",
            languages_lower,
            expectation="The list of popular programming languages should include 'Python'.",
        )

    except ResponseParsingError as e:
        # Catch any parsing or schema validation errors
        kbench.assertions.assert_fail(
            expectation=f"The output was not valid JSON or did not match the LanguageList schema. Error: {e.error}"
        )


# Run the task
list_programming_languages.run(kbench.llm)
```

```python
from pydantic import BaseModel, Field
from kaggle_benchmarks.prompting import ResponseParsingError
import kaggle_benchmarks as kbench


class Planet(BaseModel):
    name: str
    mass_earth_masses: float = Field(description="Mass relative to Earth")
    has_life: bool = Field(description="Whether the planet is known to have life")
    moons: list[str] = Field(default_factory=list, description="List of major moons")


@kbench.task(name="planet_info")
def planet_info(llm):
    """Retrieves planet information using a Pydantic model."""
    try:
        planet = llm.prompt(
            "Provide information about the planet Jupiter.", schema=Planet
        )
        kbench.assertions.assert_contains_regex(
            r"(?i)jupiter", planet.name, expectation="Planet name should be Jupiter."
        )
    except ResponseParsingError as e:
        kbench.assertions.assert_fail(
            expectation=f"Failed to parse structured output: {e.error}"
        )
```

## Recipe: Evaluating Performance on a Dataset

Instead of a single run, you can evaluate a task over many examples using the `.evaluate()` method. This maps pandas DataFrame columns directly to your task function's parameters.

**Full Signature:** `task.evaluate(llm, evaluation_data, n_jobs=1, timeout=None, stop_condition=None, max_attempts=1, retry_delay=0, remove_run_files=False)`

**Important Note:** `max_attempts` must be 1 for evaluations nested in other tasks.

``` python
import pandas as pd
import kaggle_benchmarks as kbench

# 1. Prepare your dataset columns (maps directly to task parameters)
df = pd.DataFrame(
    [
        {"question": "2+2", "answer": "4"},
        {"question": "Capital of UK", "answer": "London"},
    ]
)


# 2. Define a task that accepts the row columns as arguments
@kbench.task(name="single_qa_task", store_task=False)
def single_qa_task(llm, question, answer) -> bool:
    response = llm.prompt(question)
    return answer.lower() in response.lower()


# 3. Define the aggregation task
@kbench.task(name="multi_qa_task")
def multi_qa_task(llm, df) -> float:
    with kbench.client.enable_cache():
        # Evaluate runs the task in parallel (n_jobs) and handles retries
        runs = single_qa_task.evaluate(
            llm=[llm],
            evaluation_data=df,
            n_jobs=2,  # Parallel workers
            retry_delay=5,
            remove_run_files=True,  # Clean up intermediate run files
        )

    # 4. Analyze the returned `Runs` collection
    eval_df = runs.as_dataframe()
    accuracy = float(eval_df.result.mean())
    return accuracy


multi_qa_task.run(kbench.llm, df)
```

## Recipe: Comparing Multiple Models Side-by-Side

You can pass a list of LLM instances to `.evaluate()` to run the cross-product of `(models × rows)`. The resulting `Runs` object contains built-in rendering helpers for organizing results.

``` python
models = [kbench.llms["google/gemini-2.5-flash"], kbench.llms["meta/llama-3.1-70b"]]

# Runs the evaluation for BOTH models across the entire dataset
results = single_qa_task.evaluate(llm=models, evaluation_data=df)

# Group or pivot results for easy comparison in the notebook
display(results.pivot(by="llm", mode="columns"))
```

## Recipe: Leveraging Automatic Conversation History

By default, the system automatically creates a `Chat` object at task execution start. Every `llm.prompt()` call appends messages to this chat's history, enabling seamless multi-turn conversations.

``` python
@kbench.task()
def chat_task(llm):
    # Turn 1: Appends user and assistant messages to active Chat
    llm.prompt("Hi, I'm looking for a book.")

    # Turn 2: The LLM receives the full history automatically
    llm.prompt("I like science fiction.")

    # Respond based on existing context without appending a new user prompt
    final_thought = llm.respond()
```

## Recipe: Creating Isolated Conversations for Judges

When you need a side-conversation that shouldn't contaminate the main agent's history, use `kbench.chats.new("name")`. This creates a temporary `Chat` context for the duration of the `with` block.

``` python
@kbench.task()
def game_task(llm, judge_llm):
    # Appends to the default root chat
    llm.prompt("Player move: pawn to e4")

    # Opens an isolated chat context
    with kbench.chats.new(
        "judging_context", system_instructions="You are a strict chess arbiter."
    ):
        # This interaction is saved in 'judging_context' and hidden from 'llm'
        valid = judge_llm.prompt("Is pawn to e4 a valid opening move?", schema=bool)

    # We are back in the root chat; 'llm' doesn't know the judge evaluated it
    llm.prompt("What is your counter-move?")
```

## Recipe: Managing Multi-Agent Conversations

For complex interactions where multiple agents require long-lived, independent histories, instantiate explicit `chats.Chat` objects and use `contexts.enter(chat=...)` to switch between them.

``` python
from kaggle_benchmarks import chats, contexts, task


@task()
def multi_chat_task(llm1, llm2):
    # Create persistent, isolated chat objects
    chat1 = chats.Chat(name="Agent 1 History")
    chat2 = chats.Chat(name="Agent 2 History")

    # Agent 1 takes a turn
    with contexts.enter(chat=chat1):
        move1 = llm1.prompt("Make your opening statement.")

    # Agent 2 takes a turn, isolated from Agent 1's history
    with contexts.enter(chat=chat2):
        move2 = llm2.prompt(f"The opponent said: {move1}. Formulate a rebuttal.")

    # Switch back to Agent 1 seamlessly
    with contexts.enter(chat=chat1):
        llm1.prompt(f"The opponent replied: {move2}. Your response?")
```

## Recipe: Sending Images to Multimodal Models

The framework provides robust `ImageContent` objects. You can create these from URLs, local paths, NumPy arrays, or Base64 strings using the `images` factory functions.

**Important Note:**
- `llm.prompt(image=img)` automatically downloads URL-based images and converts them to Base64 to ensure maximum compatibility with all LLMs.
- `user.send(img)` sends the exact raw format without conversion.

``` python
from kaggle_benchmarks.content_types import images
import kaggle_benchmarks as kbench
import numpy as np


@kbench.task("Analyze Images")
def vision_task(llm):
    # 1. From a URL (Auto-converted to base64 by llm.prompt)
    url_img = images.from_url(
        "https://www.kaggle.com/static/images/site-logo.png", caption="Logo"
    )
    response1 = llm.prompt("Describe this image.", image=url_img)

    # 2. From a local file (Read and encoded to base64 immediately)
    local_img = images.from_path("/path/to/chart.png")

    # 3. From a NumPy array
    array_img = images.from_array(np.zeros((100, 100, 3), dtype=np.uint8))

    # You can also preload multi-image context
    kbench.user.send(local_img)
    kbench.user.send(array_img)
    response2 = llm.prompt("Compare the two images I just sent.")
```

## Recipe: Using the Built-in Python Script Runner

The library provides tools to extract and safely execute python code generated by the LLM.
- `extract_code(text, all_blocks=False)` pulls out Python code from fenced markdown blocks.
- `script_runner.run_code()` executes the code in a subprocess and returns a `ScriptOutput` containing `stdout`, `stderr`, and the `exit_code`.

```python
import kaggle_benchmarks as kbench


@kbench.task(name="python_math")
def python_math(llm):
    prompt = "Write a python script that prints 'Hello World'."
    response = llm.prompt(prompt)

    # 1. Extract code (strips markdown formatting)
    code = kbench.tools.python.extract_code(response)

    kbench.assertions.assert_not_empty(
        code, expectation="Response should contain a Python code block."
    )

    # 2. Run the code using the subprocess ScriptRunner
    result = kbench.tools.python.script_runner.run_code(code)

    # 3. Validate execution metrics (exit code 0 indicates success)
    kbench.assertions.assert_equal(
        0,
        result.exit_code,
        expectation=f"Code should execute successfully. Stderr: {result.stderr}",
    )
    kbench.assertions.assert_contains_regex(
        r"(?i)hello world",
        result.stdout,
        expectation="Standard output should contain 'Hello World'.",
    )
```
