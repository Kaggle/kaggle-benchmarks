

- [Kaggle Benchmarks Cookbook](#kaggle-benchmarks-cookbook)
  - [Basics](#basics)
    - [Recipe: Defining a Simple Pass/Fail
      Task](#recipe-defining-a-simple-passfail-task)
    - [Recipe: Using a judge LLM](#recipe-using-a-judge-llm)
    - [Recipe: Returning a Numerical
      Score](#recipe-returning-a-numerical-score)
    - [Recipe: Enforcing Structured Output with
      Schemas](#recipe-enforcing-structured-output-with-schemas)
    - [Recipe: Publishing Your Task to the
      Leaderboard](#recipe-publishing-your-task-to-the-leaderboard)
  - [Data & Evaluation](#data--evaluation)
    - [Recipe: Evaluating Performance on a
      Dataset](#recipe-evaluating-performance-on-a-dataset)
    - [Recipe: Comparing Multiple Models
      Side-by-Side](#recipe-comparing-multiple-models-side-by-side)
  - [Conversations](#conversations)
    - [Recipe: Leveraging Automatic Conversation
      History](#recipe-leveraging-automatic-conversation-history)
    - [Recipe: Creating Isolated Conversations for
      Judges](#recipe-creating-isolated-conversations-for-judges)
    - [Recipe: Managing Multi-Agent
      Conversations](#recipe-managing-multi-agent-conversations)
  - [Multimodal](#multimodal)
    - [Recipe: Sending Images to Multimodal
      Models](#recipe-sending-images-to-multimodal-models)
    - [Recipe: Sending Videos to Multimodal
      Models](#recipe-sending-videos-to-multimodal-models)
  - [Advanced Patterns](#advanced-patterns)
    - [Recipe: Implementing an Interactive Game
      Loop](#recipe-implementing-an-interactive-game-loop)
    - [Recipe: Writing Reusable Custom
      Assertions](#recipe-writing-reusable-custom-assertions)
    - [Recipe: Using the Built-in Python Script
      Runner](#recipe-using-the-built-in-python-script-runner)
    - [Recipe: Equipping Models with Custom
      Tools](#recipe-equipping-models-with-custom-tools)

# Kaggle Benchmarks Cookbook

Welcome to the Kaggle Benchmarks Cookbook! This guide provides a
collection of “recipes” — practical examples and patterns to help you
get the most out of the library.

For a comprehensive overview, refer to the [Quick Start](quick_start.md)
and [User Guide](user_guide.md). For starter, please take a look at this
[example
notebook](https://www.kaggle.com/code/nicholaskanggoog/starter-notebook-template-nick).

## Basics

### Recipe: Defining a Simple Pass/Fail Task

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

[See Example: Assertions
Notebook](https://www.kaggle.com/code/kerneler/kaggle-benchmark-cookbook-assertions)

### Recipe: Using a judge LLM

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

### Recipe: Returning a Numerical Score

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

**Supported Return Types:** - `bool`: Pass/Fail (True/False) - `float` /
`int`: A numerical score (e.g., 0.85) - `tuple[int, int]`: A count
(e.g., `(8, 10)` for 8 out of 10 passed) - `tuple[float, float]`: A
value with confidence interval (e.g., `(0.85, 0.05)`)

[See Example: Return Types
Notebook](https://www.kaggle.com/code/kerneler/kaggle-benchmark-cookbook-return-types)

### Recipe: Enforcing Structured Output with Schemas

You can force the LLM to return data in a specific structure by
providing a `schema` to the `prompt` method. This is essential for tasks
that require structured output.

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
        "Analyze this review: 'Fantastic movie!'",
        schema=MovieReview
    )
    print(f"Score: {result.score}, Sentiment: {result.sentiment}")
```

[See Example: Structured Output
Notebook](https://www.kaggle.com/code/kerneler/kaggle-benchmark-cookbook-structured-output)

### Recipe: Publishing Your Task to the Leaderboard

Currently our leaderboard only supports a single task per notebook (This
may change in the future). To submit your benchmark to a Kaggle
leaderboard, you need to designate one “main” task output using the
`%choose` magic command.

1.  **Define your tasks** in the notebook.
2.  **Select the main task** in the very last cell using `%choose`.
3.  **Save Version** of your notebook.

``` python
# In the final cell of your notebook:
%choose my_main_task
```

[See Example: Task Creation
Notebook](https://www.kaggle.com/code/kerneler/kaggle-benchmark-cookbook-task-creation)

------------------------------------------------------------------------

## Data & Evaluation

### Recipe: Evaluating Performance on a Dataset

Instead of a single run, you often want to evaluate a task over many
examples. Use the `.evaluate()` method to run your task against every
row in a pandas DataFrame.

``` python
import pandas as pd

# 1. Define a task that accepts row columns as arguments
@kbench.task()
def solve_question(llm, question, answer) -> bool:
    response = llm.prompt(question)
    return answer.lower() in response.lower()

# 2. Prepare your dataset
df = pd.DataFrame([
    {"question": "2+2", "answer": "4"},
    {"question": "Capital of UK", "answer": "London"},
])

# 3. Run evaluation
results = solve_question.evaluate(llm=[kbench.llm], evaluation_data=df)
print(results.as_dataframe())
```

### Recipe: Comparing Multiple Models Side-by-Side

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

[See Example: Dataset Evaluation
Notebook](https://www.kaggle.com/code/kerneler/kaggle-benchmark-cookbook-dataset-evaluation)

------------------------------------------------------------------------

## Conversations

### Recipe: Leveraging Automatic Conversation History

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

### Recipe: Creating Isolated Conversations for Judges

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

[See Example: Conversation Management
Notebook](https://www.kaggle.com/code/kerneler/kaggle-benchmark-cookbook-conversations)

### Recipe: Managing Multi-Agent Conversations

In complex multi-agent scenarios, you often need to maintain separate
conversation histories for each agent. The *Dungeon Adventure* example
demonstrates how to do this by creating a dedicated `Chat` object for
each agent and using the `contexts.enter()` context manager to switch
between them. This allows each agent to have its own isolated
conversation history, which is crucial for role-playing and other
complex interactions.

``` python
# Create isolated chat contexts for each agent
dm_chat = chats.Chat(name="Dungeon Master")
player1_chat = chats.Chat(name="Player 1")

# Create the agents with their own chat contexts
dungeon_master = DungeonMasterAgent(dm_llm, dm_chat, story)
player1 = PlayerAgent(player1_llm, player1_chat, "Aragorn")

# --- In the PlayerAgent's __call__ method ---
with contexts.enter(chat=self.player_chat):
    # This prompt is only added to the player's chat history
    action = self.llm.prompt(prompt)

# --- In the DungeonMasterAgent's __call__ method ---
with contexts.enter(chat=self.dm_chat):
    # This prompt is only added to the Dungeon Master's chat history
    self.story = self.llm.prompt(prompt)
```

[See Example: Dungeon Adventure
Notebook](https://www.kaggle.com/code/kerneler/kaggle-benchmark-cookbook-dungeon-adventure)

------------------------------------------------------------------------

## Multimodal

### Recipe: Sending Images to Multimodal Models

Support for image inputs varies across Large Language Models (LLMs).
While some models do not accept images at all, multimodal models
generally fall into two categories:

- **Base64 Only**: The model requires the image data encoded as a
  string.
- **Base64 & URL**: The model can accept encoded strings or access
  images from a URL (requires internet access).

The `kaggle-benchmarks` SDK unifies these inputs using the `images`
module, but behaves differently depending on whether you are sending a
single prompt or building conversation history.

**Constructing Image Objects:** Regardless of the sending method, you
initiate images using three factory methods:

``` python
# 1. From URL (MIME type guessed from URL)
img_url = images.from_url("https://example.com/image.jpg")

# 2. From Local Path (MIME type guessed from extension; converted to Base64 automatically)
img_local = images.from_path("local/image/path.png")

# 3. From Base64 String (Default format is "jpeg")
img_b64 = images.from_base64(image_b64_str, format="png")
```

**Sending Methods: `prompt` vs `send`** The SDK provides two methods to
interact with models. It is crucial to understand how they handle
**Image URLs** differently.

| Method | Use Case | Behavior with Image URLs |
|:---|:---|:---|
| `llm.prompt` | Single message & immediate response. | **Auto-converts to Base64.** The SDK downloads the image and sends the data. Safe for models that don’t support URLs. |
| `user.send` | Building multi-turn chat history. | **Sends Raw URL.** The SDK passes the URL directly. This is useful for testing if a model natively supports URL fetching. |

*1.Single Turn: Using `llm.prompt`* Use this for straightforward
interactions. The SDK handles the heavy lifting of downloading and
encoding URLs, ensuring maximum compatibility.

``` python
# Create an image input from a URL
image = images.from_url("https://example.com/cat.jpg")

# The SDK downloads the image, converts it to Base64, and sends it with the text.
# The auto conversion makes it easier for users since it's the most common use case.
response = llm.prompt("Describe this image", image=image)
```

*2.Multi-Turn: Using `user.send`* Use `user.send` to stack multiple
messages or images before asking for a response. Note that this method
acts as a low-level pass-through for URLs.

``` python
# 1. Add a local image to history
# This is read and converted to Base64 immediately.
user.send(images.from_path("local/chart.png"))

# 2. Add an image URL to history
# This is NOT converted. The raw URL is sent to the model.
# This will fail if the model does not support URL inputs.
# However you can still use the local image or raw Base64.
user.send(images.from_url("https://example.com/diagram.jpg"))

# 3. Trigger the response based on the history accumulated above
response = llm.prompt("Compare these two images")
```

[See Example: Sending Images
Notebook](https://www.kaggle.com/code/kerneler/kaggle-benchmark-cookbook-images)

### Recipe: Sending Videos to Multimodal Models

Some models (currently select Gemini models) support video inputs. The
SDK provides a `videos` module for this, following the same pattern as
images.

**Constructing Video Objects:**

``` python
from kaggle_benchmarks.content_types import videos

# From a YouTube URL
video = videos.from_url("https://www.youtube.com/watch?v=aqz-KE-bpKQ")
```

**Sending a Video with `llm.prompt`:**

``` python
from kaggle_benchmarks.content_types import videos

video = videos.from_url("https://www.youtube.com/watch?v=aqz-KE-bpKQ")
response = llm.prompt("What is this video about?", video=video)
```

> **Note:** Video support depends on the model. Currently, only select
> models support video inputs, and only YouTube URLs are
> supported. Models that don't support video will return an error.

### Recipe: Sending Audio to Multimodal Models

Some models (currently select Gemini models) support audio inputs. The
SDK provides an `audio` module for this, following the same pattern as
images and videos.

**Constructing Audio Objects:**

``` python
from kaggle_benchmarks.content_types import audios

# From a local file
audio_content = audios.from_path("speech.mp3")

# From a URL
audio_content = audios.from_url("https://example.com/speech.mp3")

# From a base64-encoded string
audio_content = audios.from_base64(b64_string, format="mp3")
```

**Sending Audio with `llm.prompt`:**

``` python
from kaggle_benchmarks.content_types import audios

audio_content = audios.from_path("speech.mp3")
response = llm.prompt("Transcribe this audio.", audio=audio_content)
```

> **Note:** Audio support depends on the model. Currently, only select
> models support audio inputs. Models that don't support audio will
> return an error.

------------------------------------------------------------------------

## Advanced Patterns

### Recipe: Implementing an Interactive Game Loop

For interactive environments like Tic-Tac-Toe or Chess, use a `while`
loop to alternate between the LLM’s move and the game state update.

``` python
@kbench.task()
def play_game(llm):
    game = TicTacToe()

    # Loop until the game ends
    while not game.is_over():
        # Render board and ask for move
        move = llm.prompt(f"Board:\n{game.render()}\nYour move?")

        # Update game state
        game.apply_move(move)

    return game.winner == "AI"
```

[See Example: Games
Notebook](https://www.kaggle.com/code/kerneler/kaggle-benchmark-cookbook-games)

[Another Game Example for Valid
Spymaster](https://www.kaggle.com/code/goefft/codenames-valid-spymaster-clues-task)

### Recipe: Writing Reusable Custom Assertions

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

[See Example: Custom Assertions
Notebook](https://www.kaggle.com/code/kerneler/kaggle-benchmark-cookbook-customized-assertions)

### Recipe: Using the Built-in Python Script Runner

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

### Recipe: Equipping Models with Custom Tools

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

llm= models.load_model(
    model_name=kbench.llm.name,
    api="genai",
)

weather_task.run(llm)
```

[See Example: Using Tools
Notebook](https://www.kaggle.com/code/kerneler/kaggle-benchmark-cookbook-using-tools)
