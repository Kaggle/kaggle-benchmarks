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
  - [Multi-Agent Conversations with
    `ChatRoom`](#multi-agent-conversations-with-chatroom)
- [3. Writing Assertions](#3-writing-assertions)
  - [Built-in Assertions](#built-in-assertions)
  - [The `expectation` Parameter](#the-expectation-parameter)
  - [Custom Assertions](#custom-assertions)
  - [Model-based Assertions with
    `assess_response_with_judge`](#model-based-assertions-with-assess_response_with_judge)
- [4. Evaluating Multiple Models on
  Kaggle](#4-evaluating-multiple-models-on-kaggle)
- [5. Choosing a UI](#5-choosing-a-ui)

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

When running on Kaggle, the `name` and `description` arguments are
subject to platform length limits; the decorator raises `ValueError`
immediately if they are exceeded so you don't waste a run.

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

By default, `.evaluate()` raises on the first per-sample failure (the
right behavior for development). For production-scale evaluations where
some samples may hit transient errors (API timeouts, rate limits, etc.),
pass `on_failure="continue"` to collect failed runs into the returned
`Runs` instead of raising:

``` python
results = solve_and_check_riddle.evaluate(
    llm=[kbench.llm],
    evaluation_data=riddle_df,
    on_failure="continue",
)

# Split successes from failures
print(f"Completed: {len(results.completed_runs)}")
print(f"Errored:   {len(results.errored_runs)}")

# Always aggregate over completed_runs only — failed runs carry the
# results.FAILED sentinel which breaks .mean() / .sum().
success_rate = results.completed_runs.as_dataframe()["result"].mean()
```

Combine with `max_attempts > 1` and `enable_cache()` to retry only the
failed samples on subsequent attempts. See the cookbook recipe "Best
Practices for Large Datasets" for the full pattern.

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
from kaggle_benchmarks.content_types import audios

# From a local file
response = kbench.llm.prompt(
    "Transcribe this audio.",
    audio=audios.from_path("speech.mp3")
)

# From a URL
response = kbench.llm.prompt(
    "Transcribe this audio.",
    audio=audios.from_url("https://example.com/speech.mp3")
)

# From base64
response = kbench.llm.prompt(
    "Transcribe this audio.",
    audio=audios.from_base64(b64_string, format="mp3")
)
```

> **Note:** Audio support depends on the model. Currently, only select
> models support audio inputs. Models that don't support audio will
> return an error.

### Reasoning

The `reasoning` parameter controls how much reasoning the model
performs. The SDK maps this to the correct provider-specific parameter
automatically (`reasoning_effort` for OpenAI, `thinking_config` for
GenAI).

```python
response = llm.prompt("Solve this math problem.", reasoning="high")
```

Valid values: `"none"`, `"low"`, `"medium"`, `"high"`.

> **Note:** Not all models support reasoning. Models that don't support
> it will return an error.

Since `prompt()` returns a plain string, use `kbench.last_reasoning_traces()`
to access the model's reasoning traces from the most recent response:

```python
response = llm.prompt("How many r's are in 'strawberry'?", reasoning="high")
traces = kbench.last_reasoning_traces()  # model's internal reasoning
```

### `llm.prompt()` with Tool Calling

You can allow the LLM to use Python functions as tools by passing them
in a list to the `tools` parameter. The library handles multi-turn tool
calling automatically: it sends the tools to the model, executes any
requested tool calls, feeds the results back, and repeats until the
model produces a final answer.

``` python
import kaggle_benchmarks as kbench


def weird_multiply(a: int, b: int) -> int:
    "Multiplies two integers."
    return 42


response = kbench.llm.prompt(
    "Answer user questions by only using the provided tool: what is 2 times 3?",
    tools=[weird_multiply],
)
```

Tools can be combined with `schema` to get structured output after the
tool-calling phase completes:

``` python
from dataclasses import dataclass


@dataclass
class CityInfo:
    name: str
    population: int


def get_city_data(city_name: str) -> dict:
    """Returns data about a city."""
    return {"name": city_name, "population": 3_700_000}


result = kbench.llm.prompt(
    "Look up Berlin and return it as a CityInfo.",
    tools=[get_city_data],
    schema=CityInfo,
)
# result is a CityInfo instance
```

You can verify that a tool was actually called using the built-in
`assert_tool_was_invoked` assertion:

``` python
kbench.assertions.assert_tool_was_invoked(get_city_data)
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

### Multi-Agent Conversations with `ChatRoom`

For benchmarks where two or more LLMs need to talk to each other
(debate, negotiation, social deduction, cooperative games), use
`kbench.ChatRoom`. The room handles three things you would otherwise
hand-roll:

1. **Perspective projection**. Each LLM sees its own past messages as
   `assistant` and peers' messages as `user` with a name prefix.
2. **Shared identity routing**. One LLM instance can act as many
   participants. The room wraps each in a lightweight `Participant`
   with its own `name`, `avatar`, and `system_prompt`. The LLM is
   never cloned.
3. **A single ground-truth transcript** for post-room evaluation.

#### Basic usage: two primitives

Inside `with room:` you only ever do two things:

- `room.post(msg)` — narrator/system directive. **No LLM call.** Use
  for phase transitions, topic prompts, or game-state announcements.
- `participant.reply()` — the LLM takes a turn. **One LLM call.** The
  response is automatically added to the room transcript and attributed
  to the participant.

``` python
import kaggle_benchmarks as kbench

@kbench.task()
def debate_task(llm):
    room = kbench.ChatRoom(
        system_prompt="A short debate. Each speaker gets one turn.",
        name="Moderator",
    )
    pro = room.add_participant(
        llm, name="Pro", avatar="🔵",
        system_prompt="Argue IN FAVOR of the topic. Be concise.",
    )
    con = room.add_participant(
        llm, name="Con", avatar="🔴",
        system_prompt="Argue AGAINST the topic. Be concise.",
    )

    with room:
        room.post("Topic: Should we phase out fossil fuels by 2035?")
        pro_argument = pro.reply()
        con_argument = con.reply()

    # room.messages is the full transcript, in order.
    return {"pro": pro_argument, "con": con_argument}
```

#### Structured replies

`reply(schema=...)` works the same way as `llm.prompt(schema=...)` —
the returned value is the parsed object.

``` python
import dataclasses

@dataclasses.dataclass
class Move:
    row: int
    col: int

move = player.reply(schema=Move)  # returns Move instance
```

#### Private channels

For multi-turn private conversations (e.g. werewolves coordinating at
night), create a sub-room with `room.private_channel(...)`. Members
see both the parent room and the channel interleaved in time; non-members
never see channel messages.

``` python
wolf_chat = room.private_channel(
    [alice, bob], name="Werewolf Night Chat"
)
with wolf_chat:
    wolf_chat.post("Pick a villager to eliminate.")
    for wolf in [alice, bob]:
        wolf.reply()
```

For a one-off private directive (not a multi-turn discussion), pass
`visible_to=[...]` to `room.post()`:

``` python
room.post("Your secret role is Werewolf.", visible_to=[alice])
```

#### Removing participants

When a participant leaves the game (e.g. a werewolf player is
eliminated), call `room.remove_participant(p)`. The framework's
roster stops listing them, so surviving LLMs no longer see them as a
peer. Their past messages remain attributed to them in the transcript.

``` python
room.remove_participant(victim)
```

A removed participant raises `RuntimeError` if you call `.reply()` on
them inside the room.

#### One-shot calls outside the room

A judge that scores the finished debate is **not** a room participant.
Use the raw `LLMChat` directly after the room exits:

``` python
with room:
    # ... debate happens ...
    ...

transcript = "\n".join(str(m) for m in room.messages)
verdict = judge_llm.prompt(
    f"Who argued better?\n\n{transcript}",
)
```

> **Rule of thumb**: if a model does not call `.reply()` inside the
> room, do not register it with `add_participant`. Registering puts
> it in every peer's roster (changing how the peers behave) without
> a real role for it to play. Post-room judging reads `room.messages`;
> live observation can subscribe to the `new_message` event.

#### Full examples

Five worked examples live in `documentation/examples/`:

- `chatroom_llm_debate.py` — two-sided structured debate with an
  external judge.
- `chatroom_pizza_order.py` — customer-vs-clerk negotiation under
  budget, allergy, and pricing constraints.
- `chatroom_synthetic_turing_test.py` — judge tries to detect a
  human-impersonating subject.
- `chatroom_tic_tac_toe.py` — structured-output game moves between
  two LLMs.
- `chatroom_werewolf.py` — 7-player social deduction with private
  channels and eliminations.

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
- `assert_tool_was_invoked(tool, ...)`: Checks if a specific tool
  function was invoked during the current task. Accepts either a
  callable or a tool name string.
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

## 5. Choosing a UI

`kaggle-benchmarks` renders runs differently depending on where you
execute them:

- **Notebook kernels** (Jupyter, Kaggle, VSCode) display each `Run` via
  its cell output (Panel-based rich widgets). No live event-bound UI is
  attached by default, so heavy benchmarks (e.g. video QA) won't tax
  the kernel with continuous redraws.
- **Terminals / scripts** auto-bind `ConsoleUI`, which prints a
  structured transcript of each run.

### Opting in to live PanelUI streaming in notebooks

If you want the live, streaming Panel UI inside a notebook (e.g. while
iterating on a small task and you want to watch tokens arrive), opt in
explicitly:

``` python
import kaggle_benchmarks as kbench

kbench.config.enable_interactive_mode()
```

Equivalently, set `INTERACTIVE_UI=True` in your environment before
importing the library.

### Console output

In a terminal, `ConsoleUI` is on by default. To force it on (for
example, when running inside a notebook for log-friendly output) or
tune its behavior, use:

``` python
kbench.config.enable_console_mode(quiet=False, color=None)
```

The corresponding environment variables are `BENCHMARK_CONSOLE_UI`,
`BENCHMARK_CONSOLE_QUIET`, and `BENCHMARK_CONSOLE_COLOR`.

## What's Next?

For more practical examples and best practices — including recipes for
caching, retries, multimodal inputs, custom tools, and more — check out
the [Cookbook](cookbook.md).
