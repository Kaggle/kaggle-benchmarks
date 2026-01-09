# Quick Start: Using Kaggle Benchmarks


- [1. Introduction](#1-introduction)
- [2. Setup in a Kaggle Notebook](#2-setup-in-a-kaggle-notebook)
- [3. Core Concepts](#3-core-concepts)
- [4. Creating Your First Benchmark](#4-creating-your-first-benchmark)
  - [Basic Task](#basic-task)
  - [Structured Output](#structured-output)
- [5. Advanced Features](#5-advanced-features)
  - [Image Input](#image-input)
  - [Using Multiple LLMs](#using-multiple-llms)
  - [Using Tools: The Python Script
    Runner](#using-tools-the-python-script-runner)
- [6. Creative and Robust Assertions](#6-creative-and-robust-assertions)
  - [Using External Libraries](#using-external-libraries)
- [7. Evaluating on a Dataset](#7-evaluating-on-a-dataset)
- [8. Using IPython magics](#8-using-ipython-magics)
  - [8.1 Using `%autopilot` to Generate
    Code](#81-using-autopilot-to-generate-code)
  - [8.2 Using `%choose` to Select a Notebook’s
    Task](#82-using-choose-to-select-a-notebooks-task)
- [9. Referring to examples](#9-referring-to-examples)

This guide will walk you through the basics of using the
`kaggle-benchmarks` library to create, run, and evaluate LLMs directly
within a Kaggle notebook.

## 1. Introduction

`kaggle-benchmarks` is a Python library designed to help you rigorously
evaluate AI models on tasks that matter to you. It provides a structured
framework for defining tasks, interacting with models, and asserting the
correctness of their outputs. This is especially useful for:

- **Reproducibility:** Capture the exact inputs, outputs, and model
  interactions for later review.
- **Complex Evaluations:** Go beyond simple string matching to test for
  code execution, tool use, and multi-turn conversational abilities.
- **Rapid Prototyping:** Quickly test a model’s capabilities on a new,
  creative task you’ve designed.

## 2. Setup in a Kaggle Notebook

To use `kaggle-benchmarks` within a Kaggle notebook, you don’t need to
install anything. For early access, simply navigate to
https://www.kaggle.com/benchmarks/tasks/new. This will automatically
create a new Kaggle notebook with `kaggle-benchmarks` and its
dependencies pre-installed.

You can install additional dependencies using `!pip` magic, just as you
would in a standard Kaggle notebook.

## 3. Core Concepts

Here are some core concepts for using the library effectively:

- **`Task`**: The fundamental unit of evaluation. A task is a Python
  function decorated with `@kbench.task` that defines a specific problem
  for the model to solve. It takes inputs (like an LLM and a prompt) and
  can optionally return a value. If no value is returned, the task is
  graded Pass/Fail based on its assertions. The first parameter must
  always be the LLM being tested; additional parameters are optional.
- **`LLM`**: An object representing a large language model you can
  interact with. You can access available Kaggle models via
  `kbench.llms["vendor/model-name"]`.
- **`Chat` and `Actor`**: The library represents interactions as a
  conversation. When you call `llm.prompt()` or `kbench.user.send()`, a
  `Message` is added to the current `Chat`. The `Actor` (e.g.,
  `kbench.user` or an `LLM` instance) defines who is sending the
  message.
- **`Assertion`**: A check to verify the model’s output. If an assertion
  fails, it’s recorded in the run results. You can make assertions using
  `kbench.assertions.assert_that(..., expectation)`, where `expectation`
  is a message summarizing the assertion that will be displayed on the
  leaderboard. You can also write your own assertions by using the
  `@kbench.assertions.assertion_handler` decorator.
- **`Run`**: A recorded execution of a `Task`. It captures everything:
  the task definition, input parameters, the full chat history, any
  assertion results, and the final return value.

## 4. Creating Your First Benchmark

Let’s start with a simple task.

### Basic Task

Here, we define a task that asks a model a classic riddle and checks if
the answer is correct.

``` python
import kaggle_benchmarks as kbench

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
```

### Structured Output

For more complex analysis, you can ask the model to return a structured
object instead of plain text. The library supports `dataclasses`,
`pydantic` models, dictionaries, and primitive types. You can specify
the desired output format using the `schema` parameter in
`llm.prompt()`.

Here’s an example that extracts information into a `dataclass`:

``` python
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
```

Structured output support varies depending on the LLM and user prompt.
To improve results, be specific in prompts and consider providing
examples.

## 5. Advanced Features

Now let’s explore some of the more powerful features of the library.

### Image Input

You can send images to multimodal models. The library provides helpers
to load images from files or base64 strings.

``` python
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
```

### Using Multiple LLMs

You can easily compare multiple models within a single task by creating
separate `Chat` contexts for each one.

``` python
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
```

### Using Tools: The Python Script Runner

One of the most powerful features is giving an LLM access to tools. The
built-in Python Script Runner allows the model to execute code to solve
problems.

``` python
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
```

## 6. Creative and Robust Assertions

Go beyond simple equality checks to create more meaningful evaluations.

### Using External Libraries

You can use any Python library to help validate a model’s response. For
example, use `Levenshtein` distance for fuzzy string matching.

``` python
# Install and use extra dependencies.
%pip install -q python-Levenshtein
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
```

## 7. Evaluating on a Dataset

You can run a task over an entire dataset (e.g., a pandas DataFrame) to
get aggregate performance metrics. The `.evaluate()` method runs the
task for each row of the data and can run each row in parallel.

``` python
import pandas as pd

import kaggle_benchmarks as kbench

# Dataset to evaluate.
df = pd.DataFrame(
    [
        {
            "question": "What's the capital of Singapore",
            "answer": "Singapore",
        },
        {
            "question": "What's the capital of France",
            "answer": "Paris",
        },
    ]
)

# First define the task for a single row of the dataset.
@kbench.task(store_task=False)
def single_qa_task(llm, question, answer) -> dict:
    response = llm.prompt(question)
    return {
        "question": question,
        "gold_target": answer,
        "predicted_answer": response,
        "is_correct": answer.lower() in response.lower(),
    }

# Define the task for the entire dataset.
@kbench.task()
def multi_qa_task(llm, df) -> tuple[float, float]:
    with kbench.client.enable_cache():
        runs = single_qa_task.evaluate(
            stop_condition=lambda runs: len(runs) == df.shape[0],
            max_attempts=50,
            retry_delay=15,
            llm=[llm],
            evaluation_data=df,
            n_jobs=2,
            timeout=120,
            remove_run_files=True,  # Optionally remove sub runs files.
        )
    eval_df = runs.as_dataframe()

    # Use float() to convert from np.float.
    accuracy = float(eval_df.result.str.get("is_correct").mean())
    std = float(eval_df.result.str.get("is_correct").std())
    return accuracy, std


run = multi_qa_task.run(kbench.llm, df)
run
```

The `.evaluate()` method is powerful for batch processing. When running
in a notebook, it will typically display a progress bar, giving you a
visual indicator of how many tasks have been completed. It returns a
`Runs` object, which is a list of individual `Run` objects for each row
in your dataset.

For large datasets, the detailed evaluation of individual items can be
verbose and slow. To dramatically speed up the process and remove this
detail, simply set the environment variable:
`os.environ["RENDER_SUBRUNS"] = "False"`.

## 8. Using IPython magics

Several IPython magics are available for you to use within Kaggle
notebooks.

### 8.1 Using `%autopilot` to Generate Code

To accelerate your workflow, you can use the `%autopilot` line magic in
a Kaggle notebook. This command opens a simple UI that automatically
generates boilerplate code, letting you focus on the unique logic of
your benchmark. Using this function will count against your user quota.

``` python
%autopilot
```

### 8.2 Using `%choose` to Select a Notebook’s Task

Currently, the leaderboard only supports a single Task as the output of
a notebook. However, it’s common to generate multiple tasks and run
files as intermediate results (e.g., when evaluating a dataset, the
evaluation of each row could generate its own intermediate result).

To make it convenient for users to choose the main task and its
corresponding run file to upload to a leaderboard, you can use the
`%choose` magic at the end of the notebook.

For example, given the following tasks:

``` python
@kbench.task()
def sub_task1(): ...

@kbench.task()
def sub_task2(): ...

@kbench.task()
def main_task():
   sub_task1.evaluate(...)
```

You can select the main task like this:

``` python
# Run this to keep a benchmark task and its latest run task.
# This will remove all other *run.json and *task.json files
# from the working directory (/kaggle/working).
%choose main_task
```

## 9. Referring to examples

You can quickly browse and run examples directly within a Kaggle
notebook.

To see all available examples, list the contents of the examples
directory:

``` python
# Browse examples
!ls /benchmarks/documentation/examples/
```

To load and run a specific example, like the one for game benchmarking,
use the %load magic command:

``` python
# Load and run the example for game benchmarking.
%load /benchmarks/documentation/examples/tic_tac_toe_game.py
```
