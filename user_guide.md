# Kaggle Benchmarks: A User Guide


- [1. Writing a Task](#1-writing-a-task)
  - [The `@kbench.task` Decorator](#the-kbenchtask-decorator)
  - [Task Function Parameters](#task-function-parameters)
  - [Task Return Types and Result
    Inference](#task-return-types-and-result-inference)
- [2. Interacting with LLMs](#2-interacting-with-llms)
  - [`llm.prompt()`](#llmprompt)
  - [`llm.prompt()` with Tool Calling](#llmprompt-with-tool-calling)
  - [`kbench.user.send()` and Multi-Turn
    Chats](#kbenchusersend-and-multi-turn-chats)
  - [Other Interaction Methods: `send` and
    `respond`](#other-interaction-methods-send-and-respond)
  - [Managing Chat Context](#managing-chat-context)
  - [Tracking Token Usage and Costs](#tracking-token-usage-and-costs)
- [3. Writing Assertions](#3-writing-assertions)
  - [Built-in Assertions](#built-in-assertions)
  - [The `expectation` Parameter](#the-expectation-parameter)
  - [Custom Assertions](#custom-assertions)
  - [Model-based Assertions with
    `assess_response_with_judge`](#model-based-assertions-with-assess_response_with_judge)
- [4. Evaluating Multiple Models on
  Kaggle](#4-evaluating-multiple-models-on-kaggle)

This guide provides a deeper dive into the core APIs of the
`kaggle-benchmarks` library. We’ll cover how to define tasks, interact
with Large Language Models (LLMs), and write powerful assertions to
evaluate their performance.

## 1. Writing a Task

The core of any benchmark is the `Task`. It’s a Python function that
defines a problem for an LLM to solve.

### The `@kbench.task` Decorator

You can define a task by decorating a Python function with
`@kbench.task`.

``` python
import kaggle_benchmarks as kbench

@kbench.task(name="my_first_task")
def my_task(llm):
    # ... task logic ...
    pass
```

### Task Function Parameters

A task function always accepts an `LLM` object as its first argument.
This object is your interface to the model being evaluated. You can
define additional parameters to make your task more flexible.

``` python
@kbench.task(name="simple_riddle")
def solve_riddle(llm, riddle: str, answer: str):
    """Asks a riddle and checks for a keyword in the answer."""
    response = kbench.llm.prompt(riddle)
    kbench.assertions.assert_contains_regex(
        f"(?i){answer}", response, expectation="The model should answer the riddle correctly."
    )
```

When you execute the task using `.run()`, you pass in the values for
these parameters:

``` python
solve_riddle.run(
    llm=kbench.llm,  # Use the default LLM
    riddle="What gets wetter as it dries?",
    answer="Towel",
)
```

### Task Return Types and Result Inference

The `kaggle-benchmarks` framework grades your task based on its Python
return type annotation. Getting this right is key to ensuring your
benchmark is rendered correctly on the leaderboard.

- **Pass/Fail Tasks**: For tasks where success is determined only by
  assertions, the function should not return a value. The framework
  treats your task as `Pass/Fail` if it:
  - Has no return type annotation (e.g., `def my_task(llm):`). By
    default, these are treated as `-> None` tasks.
  - Is explicitly annotated with `-> None` (e.g.,
    `def my_task(llm) -> None:`).
- **Tasks with Return Values**: To get a score beyond a simple pass or
  fail, your function can return a value. The return type hint is
  required for scoring. Supported types include:
  - `bool`: Explicitly marks the task as a success (`True`) or failure
    (`False`).
  - `int` or `float`: Represents a numerical score like pass rate.
  - `tuple[int, int]`: Can represent a pass count, like
    `(passed_tests, total_tests)`.
  - `tuple[float, float]`: Can represent a metric with its confidence
    interval, like `(accuracy, accuracy_ci)`.

**A mismatch between the annotated return type and the actual value
returned by your function will raise a warning like
`Wrong return type <class ...>. Expected <class ...>. This may need to lead to unexpected task behavior.`.**

As an example, a task returning a boolean can be used to score
performance across a dataset:

``` python
@kbench.task(name="evaluate_riddles")
def solve_and_check_riddle(llm, riddle: str, answer_keyword: str) -> bool:
    """A task that returns True if the model gets the riddle right."""
    response = kbench.llm.prompt(riddle)
    is_correct = answer_keyword.lower() in response.lower()
    return is_correct
```

When this task is run on a dataset using `.evaluate()`, the boolean
return values are collected, allowing for aggregate scoring.

``` python
import pandas as pd

# Create a dataset of riddles
riddle_data = {
    "riddle": [
        "I have cities, but no houses. I have mountains, but no trees. I have water, but no fish. What am I?",
        "What has an eye, but cannot see?",
    ],
    "answer_keyword": ["map", "needle"],
}
riddle_df = pd.DataFrame(riddle_data)

# The .evaluate() method runs the task for each row in the DataFrame
# and collects the boolean results.
runs = solve_and_check_riddle.evaluate(
    llm=[kbench.llm],
    evaluation_data=riddle_df,
)

# You can then analyze the results, for example, by calculating the success rate.
# The boolean results are cast to 0s and 1s for easy calculation.
success_rate = runs.as_dataframe()["result"].mean()
print(f"Success rate: {success_rate:.2f}")
```

## 2. Interacting with LLMs

The `LLM` object provides several methods to communicate with the model.
These interactions are tracked as a conversation within a `Chat` object.

### `llm.prompt()`

This is the most common method for interacting with a model. It sends a
prompt and returns the model’s response as a string.

``` python
response = kbench.llm.prompt("What is the capital of France?")
```

You can also request a structured response by providing a `schema` (like
a `dataclass` or `pydantic` model). The library will attempt to parse
the model’s output into an instance of that schema.

``` python
from dataclasses import dataclass

@dataclass
class CapitalInfo:
    city: str
    country: str

info = kbench.llm.prompt("What is the capital of France?", schema=CapitalInfo)
# info is now an instance of CapitalInfo
# e.g., CapitalInfo(city='Paris', country='France')
```

You can include images in your prompt using the `image` parameter, such
as

``` python
# Pass the image directly to the prompt
response = kbench.llm.prompt(
    "What is the animal in the picture?",
    image=images.from_url(image_url)
)
```

You can also include videos using the `video` parameter. Currently,
only YouTube URLs are supported.

``` python
from kaggle_benchmarks.content_types import videos

response = kbench.llm.prompt(
    "What is this video about?",
    video=videos.from_url("https://www.youtube.com/watch?v=aqz-KE-bpKQ")
)
```

> **Note:** Video support depends on the model. Currently, only select
> models support video inputs. Models that don't support video will
> return an error.

You can also include audio using the `audio` parameter. Audio can be
loaded from a file, a URL, or a base64-encoded string.

``` python
from kaggle_benchmarks.content_types import audio

# From a local file
response = kbench.llm.prompt(
    "Transcribe this audio.",
    audio=audio.from_path("speech.mp3")
)

# From a URL
response = kbench.llm.prompt(
    "Transcribe this audio.",
    audio=audio.from_url("https://example.com/speech.mp3")
)

# From base64
response = kbench.llm.prompt(
    "Transcribe this audio.",
    audio=audio.from_base64(b64_string, format="mp3")
)
```

> **Note:** Audio support depends on the model. Currently, only select
> models support audio inputs. Models that don't support audio will
> return an error.

### `llm.prompt()` with Tool Calling

You can allow the LLM to use Python functions as tools by passing them
in a list to the `tools` parameter.

**Note: Automatic tool calling currently requires loading the model with
`api="genai"`.**

``` python
import kaggle_benchmarks as kbench
from kaggle_benchmarks.kaggle import models


def weird_multiply(a: int, b: int) -> int:
    "Multiplies two integers."
    return 42


# NOTE: Automatic tool calling requires the `genai` API.
# For `openai` API, tools must be called manually:
# e.g., see example `use_calculator_tool.py`
llm_with_genai_api = models.load_model(
    model_name=kbench.llm.name,
    api="genai",
)

response = llm_with_genai_api.prompt(
    "Answer user questions by only using the provided tool: what is 2 times 3?",
    tools=[weird_multiply],
)
```

### `kbench.user.send()` and Multi-Turn Chats

For multi-turn conversations or multimodal input, you can use
`kbench.user.send()` to add a user message to the chat history before
prompting the LLM. This is how you can send images, for example.

``` python
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
image = kbench.content_types.images.from_base64(kbench.content_types.images.image_url_to_base64(image_url))
with kbench.chats.new("image_chat"):
    # Send an image first
    kbench.user.send(image)
    # Then ask a question about it
    response = kbench.llm.prompt("What do you see in this image?")
```

### Other Interaction Methods: `send` and `respond`

While `prompt()` is the primary method for single-turn interactions,
`send()` and `respond()` offer more fine-grained control for complex,
multi-step conversations.

- **`llm.send()`**: Adds a message to the current chat history to be
  sent to the model. Unlike `prompt()`, it does not immediately wait for
  a response, allowing you to build up conversational context before the
  LLM generates a reply.
- **`llm.respond()`**: Similar to `prompt()`, this method is used to get
  a response from the model. It is best used to continue an existing
  conversation, making it explicit that the LLM is responding to prior
  messages from the user or a tool.

For most use cases, `llm.prompt()` will cover your needs.

### Managing Chat Context

When interacting with an LLM, it’s crucial to manage the chat context to
avoid sending an ever-growing history of messages. Sending the full chat
history in a loop can lead to excessive token usage and slower
performance, as the context window of the LLM fills up with repeated
information.

To prevent this, use `kbench.chats.new()` to create a new, isolated chat
for each iteration of the loop. This ensures that each prompt is sent in
a clean context, without the history of previous iterations.

Consider this simple example where we want to ask an LLM about a list of
items:

``` python
import kaggle_benchmarks as kbench

@kbench.task()
def fun_fact_task(llm):
    items_to_ask_about = ["the sun", "a black hole", "the moon"]
    responses = []

    for item in items_to_ask_about:
        # By creating a new chat for each item, we keep the context clean
        # and avoid sending the previous questions and answers with each new prompt.
        with kbench.chats.new(f"chat about {item}"):
            prompt = f"Tell me a fun fact about {item}."
            response = llm.prompt(prompt)
            responses.append(response)

    # Now, 'responses' contains one fun fact for each item, and each was requested
    # in a separate, clean chat context, which is much more efficient.

fun_fact_task.run(kbench.llm)
```

### Tracking Token Usage and Costs

The library provides access to token usage, costs, and latency metrics
through the `Usage` dataclass. You can access this data at both
the individual message level and aggregated across an entire chat.

> **Note:** Usage metrics are only available when using the Kaggle Model
> Proxy. Otherwise, these fields will return `None`.

#### Accessing Usage from Individual Messages

Each message has a `usage` property that returns a `Usage`
object:

``` python
with kbench.chats.new("Conversation") as chat:
    response = llm.prompt("What is machine learning?")

    for msg in chat.messages:
        if msg.sender.role == "assistant":
            print(f"Input tokens: {msg.usage.input_tokens}")
            print(f"Output tokens: {msg.usage.output_tokens}")
            print(f"Input cost (nanodollars): {msg.usage.input_tokens_cost_nanodollars}")
            print(f"Output cost (nanodollars): {msg.usage.output_tokens_cost_nanodollars}")
            print(f"Backend latency (ms): {msg.usage.total_backend_latency_ms}")
```

#### Accessing Aggregated Usage from a Chat

For convenience, the `Chat` object also provides a `usage` property that
aggregates usage across all assistant messages in the conversation:

``` python
with kbench.chats.new("Conversation") as chat:
    llm.prompt("What is machine learning?")
    llm.prompt("Can you give me an example?")

    # Get aggregated metrics across all LLM responses
    print(f"Total input tokens: {chat.usage.input_tokens}")
    print(f"Total output tokens: {chat.usage.output_tokens}")
    print(f"Total input cost: {chat.usage.input_tokens_cost_nanodollars}")
    print(f"Total output cost: {chat.usage.output_tokens_cost_nanodollars}")
    print(f"Total latency: {chat.usage.total_backend_latency_ms}")
```

#### Usage Fields

The `Usage` dataclass contains the following fields:

- **`input_tokens`**: Number of input tokens sent to the model.
- **`output_tokens`**: Number of output tokens generated by the model.
- **`input_tokens_cost_nanodollars`**: Cost of input tokens in
  nanodollars (10⁻⁹ dollars).
- **`output_tokens_cost_nanodollars`**: Cost of output tokens in
  nanodollars.
- **`total_backend_latency_ms`**: Total backend processing time in
  milliseconds.

All fields return `None` if the metric is not available from the model
provider.

## 3. Writing Assertions

Assertions are how you verify that a model’s output is correct.

### Built-in Assertions

The library comes with a set of common assertion functions. Here are
some of the ones you’ll use most often:

- `assert_equal(expected, actual, ...)`: Checks if two values are equal.
- `assert_true(value, ...)`: Checks if a value is `True`.
- `assert_false(value, ...)`: Checks if a value is `False`.
- `assert_in(member, container, ...)`: Checks if a member is in a
  container (e.g., a substring in a string, or an item in a list).
- `assert_not_in(member, container, ...)`: Checks if a member is not in
  a container.
- `assert_contains_regex(pattern, text, ...)`: Checks if a regular
  expression pattern is found in the given text.
- `assert_not_contains_regex(pattern, text, ...)`: Checks if a regular
  expression pattern is not found in the given text.
- `assert_empty(container, ...)`: Checks if a container (like a string
  or list) is empty.
- `assert_not_empty(container, ...)`: Checks if a container is not
  empty.
- `assert_fail(expectation)`: Signals a test failure unconditionally,
  with an optional message.

Here is a complete example showing a simple task that uses an assertion
with a descriptive `expectation` message.

``` python
@kbench.task(name="capital_city_check")
def check_capital(llm, country: str, capital: str):
    """Asks for the capital of a country and checks the answer."""
    prompt = f"What is the capital of {country}?"
    response = llm.prompt(prompt)

    # We use assert_contains_regex for a case-insensitive match.
    # The 'expectation' message is crucial for clear reporting on the leaderboard.
    kbench.assertions.assert_contains_regex(
        f"(?i){capital}",
        response,
        expectation=f"The model should identify {capital} as the capital of {country}.",
    )

# Run the task
check_capital.run(
    llm=kbench.llm,
    country="France",
    capital="Paris",
)
```

### The `expectation` Parameter

When writing assertions, it is highly recommended to include the
`expectation` parameter, as seen in the examples.

``` python
kbench.assertions.assert_equal(
    "610",
    code_output,
    expectation="The code should print the 15th Fibonacci number."
)
```

This parameter is a human-readable string that describes what the
assertion is testing. **This text is what will be displayed on the
benchmark leaderboard**. Providing a clear expectation makes your
results much easier to interpret for yourself and others. It explains
*why* a task failed, not just *that* it failed.

### Custom Assertions

You can easily create your own reusable assertions using the
`@kbench.assertions.assertion_handler` decorator. This allows you to
encapsulate custom validation logic.

Here’s an example of a custom assertion that checks if a number is
positive:

``` python
from kaggle_benchmarks.assertions import assertion_handler, AssertionResult

@assertion_handler()
def assert_is_positive(value: float, expectation: str) -> AssertionResult:
    """Custom assertion to check if a number is positive."""
    # Custom assertion logic
    passed = value > 0

    # Simply return the result to keep it recorded in the run results
    return kbench.assertions.AssertionResult(
        passed=passed,
        expectation=expectation
        or f"Expected {value} to be positive.",
    )

# Using the custom assertion in a task
@kbench.task()
def check_number(llm):
    response = kbench.llm.prompt("Give me a positive number.", schema=float)
    assert_is_positive(response, expectation="LLM should return a positive number.")

check_number.run(kbench.llm)
```

### Model-based Assertions with `assess_response_with_judge`

For complex assessments that cannot be captured by simple checks, you
can use `assess_response_with_judge`. This helper uses a separate
“judge” LLM to evaluate a response against a set of detailed, natural
language criteria. It is particularly useful for tasks where quality
requires nuanced understanding.

The function returns an `AssessReport` object containing a list of
results, one for each criterion. You can then iterate over these results
and make explicit assertions to record the pass/fail status for the
benchmark.

Here is an example of how to use it to evaluate an explanation of a
technical concept:

``` python
import kaggle_benchmarks as kbench

@kbench.task()
def summarize_story(llm):
    # The prompt asks for a summary of a classic story
    response = llm.prompt(
        "Summarize the story of '''Little Red Riding Hood''' in two sentences."
    )

    # 1. Get the assessment report from the judge
    assess_report = kbench.assertions.assess_response_with_judge(
        criteria=(
            "The summary must mention the main character, Little Red Riding Hood.",
            "The summary must mention the antagonist, the Wolf.",
            "The summary should mention the Woods or Forest as the setting.",
            "The summary must be two sentences long.",
        ),
        response_text=response,
        judge_llm=kbench.judge_llm,
    )

    # 2. Iterate over the results and assert on each one
    for result in assess_report.results:
        kbench.assertions.assert_true(
            result.passed,
            expectation=f"Criterion: {result.criterion}. Reason: {result.reason}",
        )

summarize_story.run(kbench.llm)
```

#### Advanced Usage: Custom Prompts and Schemas

You can customize how the judge evaluates the response by providing a
`prompt_fn` (to generate the prompt sent to the judge) and an
`output_schema` (to define the structure of the judge’s response).

This is useful if you need the judge to return specific feedback.

``` python
import dataclasses
from typing import Iterable, List
import textwrap

@dataclasses.dataclass
class StoryCritique:
    """A custom schema for receiving a story critique."""
    overall_rating: int  # 1-5
    feedback: str
    passed_checks: List[str]

def custom_story_prompt(criteria: Iterable[str], response_text: str) -> str:
    """Custom prompt for the judge."""
    formatted_criteria = "\n".join(f"- {c}" for c in criteria)
    return textwrap.dedent(f"""
        Evaluate this story:
        {response_text}

        Criteria:
        {formatted_criteria}

        Return a JSON with:
        - overall_rating (1-5)
        - feedback (string)
        - passed_checks (list of strings for met criteria)
    """)

@kbench.task()
def critique_story(llm):
    story = llm.prompt("Write a very short story about a robot.")

    critique = kbench.assertions.assess_response_with_judge(
        criteria=["Must be about a robot."],
        response_text=story,
        judge_llm=kbench.judge_llm,
        prompt_fn=custom_story_prompt,
        output_schema=StoryCritique,
    )

    kbench.assertions.assert_true(
        critique.overall_rating >= 4,
        expectation=f"Story rating {critique.overall_rating} should be at least 4."
    )

critique_story.run(kbench.llm)
```

## 4. Evaluating Multiple Models on Kaggle

You can easily benchmark multiple models by using your main task as a
template.

1.  Define your task function with `llm` as the first parameter, passing
    `kbench.llm` to it.

2.  On the created task page, use “Evaluate More Models” button to
    select the models you want to evaluate. Kaggle will then run your
    task, swapping in each model you choose to generate the leaderboard.
