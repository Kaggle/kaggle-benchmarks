---
name: kaggle-benchmarks
version: 0.5.0
description: Write benchmark tasks to evaluate LLMs using the kaggle_benchmarks Python library. Covers task decorators, structured outputs, assertions, tools, dataset evaluation, and multi-turn conversations.
---

# Skill: Writing Kaggle Benchmarks Tasks

> This skill file teaches you how to write high-quality benchmark tasks using the `kaggle-benchmarks` Python library (version 0.5.0+).
> Always verify patterns against the actual source code in `src/kaggle_benchmarks/` when in doubt.

## Quick Reference

```python
import kaggle_benchmarks as kbench
```

| Symbol | Purpose |
|--------|---------|
| `kbench.task` / `kbench.benchmark` | Decorator to define a benchmark task |
| `kbench.llm` | Default LLM actor (available when Kaggle is configured) |
| `kbench.judge_llm` | Judge LLM for evaluation |
| `kbench.llms` | Dict of all available models (e.g. `kbench.llms["google/gemini-2.5-flash"]`) |
| `kbench.assertions` | Module with all assertion functions |
| `kbench.chats` | Conversation/chat context management |
| `kbench.tools` | Built-in tools (Python runner, etc.) |
| `kbench.user` / `kbench.actors.user` | Send user messages to conversation |
| `kbench.system` / `kbench.actors.system` | Send system-level messages |
| `kbench.last_reasoning_traces()` | Access reasoning traces from last prompt |
| `kbench.content_types.images` | Image input helpers |
| `kbench.content_types.videos` | Video input helpers |
| `kbench.content_types.audios` | Audio input helpers |
| `kbench.client` | Client for caching, storage |

## Minimal Examples

### Simple assertion check

```python
import kaggle_benchmarks as kbench

@kbench.task(name="geography_quiz")
def geography_quiz(llm):
    response = llm.prompt("What is the longest river in the world?")
    kbench.assertions.assert_contains_regex(
        r"(?i)nile", response,
        expectation="Should mention the Nile river."
    )

geography_quiz.run(kbench.llm)
```

### Evaluating a list of questions

```python
import kaggle_benchmarks as kbench
import pandas as pd

@kbench.task(name="math_qa", store_task=False)
def math_qa(llm, question, expected) -> bool:
    answer = llm.prompt(question + "\nAnswer with just the number.", schema=int)
    kbench.assertions.assert_equal(expected, answer)
    return answer == expected

# %%
df = pd.DataFrame([
    {"question": "What is 15% of 200?", "expected": 30},
    {"question": "What is 7 × 8?", "expected": 56},
])

@kbench.task(name="math_benchmark")
def math_benchmark(llm) -> float:
    results = math_qa.evaluate(llm=[llm], evaluation_data=df, n_jobs=2)
    scores = results.as_dataframe()
    return float(scores.result.mean())

math_benchmark.run(kbench.llm)
```

## Key Rules

- The first parameter of every task function **must** be the LLM actor.
- If your task returns a value, you **MUST** add a return type annotation (`-> float`, `-> bool`, `-> dict`, etc.).
- Use `kbench.assertions.*` instead of Python `assert` — library assertions are recorded and tracked.
- Always check `assess_response_with_judge` for `None` before using the result.
- Do NOT wrap `.run()` or `.evaluate()` calls inside `if __name__ == "__main__":`. Benchmark files are notebook-style scripts — all code runs at the top level.
- Use `# %%` cell markers to create logical sections in benchmark files.
- Prefer `# !pip install ...` (commented) over `!pip install ...` so the file works everywhere.
- Use `store_task=False` for sub-tasks called inside other tasks.

## Common Mistakes to Avoid

| Mistake | Correct Approach |
|---------|-----------------|
| Missing return type annotation on scoring task | Add `-> float`, `-> bool`, `-> dict`, etc. |
| Using Python `assert` instead of `kbench.assertions.*` | Use library assertions — they're recorded and tracked |
| Not checking `assess_response_with_judge` for `None` | Always check: `if assessment is None:` |
| Using `kbench.llm` locally without Kaggle configured | Run `kaggle benchmarks init` to configure, or set env vars |
| Forgetting `schema=` when needing structured output | Pass `schema=MyDataclass` to `llm.prompt()` |
| Wrapping `.run()` / `.evaluate()` in `if __name__ == "__main__":` | Place them at module top level — benchmark files are scripts, not importable modules |
| Using `user.send()` with image URLs | `user.send()` passes URLs as-is; prefer `llm.prompt(image=)` for auto-conversion |
| Not isolating judge conversations | Use `with kbench.chats.new("judge"):` |
| Multiple tasks sharing conversation history | Each `.run()` creates its own conversation |
| Using `store_task=True` for sub-tasks | Set `store_task=False` for helper tasks called inside other tasks |
| Using `!pip install` without commenting | Use `# !pip install -q pkg` — uncommented magics break local execution |
| Forgetting `last_reasoning_traces()` can be `None` | Always check: `traces = kbench.last_reasoning_traces(); if traces: ...` |

---

## §1. Import Styles

There are two main import styles. **Prefer Style A** for clarity.

### Style A: Module import (Preferred)
```python
import kaggle_benchmarks as kbench

@kbench.task(name="my_task")
def my_task(llm):
    response = llm.prompt("Question?")
    kbench.assertions.assert_true(True)
```

### Style B: Direct imports
```python
from kaggle_benchmarks import assertions, chats, llm, task, system, user

@task("my_task")
def my_task(llm):
    response = llm.prompt("Question?")
    assertions.assert_true(True)
```

Style B is shorter but risks name collisions (e.g., `llm` is both a module-level variable and a task parameter).

### File Structure: Cell Markers

Benchmark files are Python scripts (`.py`), but use `# %%` cell markers to create logical sections. This makes them runnable as both standalone Python files and as interactive notebooks (via Jupyter/VS Code cell execution).

```python
# %%
import kaggle_benchmarks as kbench

# %%
@kbench.task()
def my_task(llm):
    response = llm.prompt("Hello!")
    kbench.assertions.assert_not_empty(response)

my_task.run(kbench.llm)

# %%
@kbench.task()
def another_task(llm) -> float:
    ...
```

**IPython magics (`!pip install`, `%time`, etc.):** These work on Kaggle notebooks but NOT when running as standalone Python files. If you need a magic command (e.g., to install a dependency), comment it out so the file remains runnable locally:

```python
# %%
# !pip install -q pronouncing syllables   # Uncomment on Kaggle
import pronouncing
```

> **Rule:** Prefer `# !pip install ...` (commented) over `!pip install ...` so the file works everywhere. Only use uncommented magics when the file is exclusively for Kaggle notebook execution.

**IMPORTANT — No `if __name__` guards.** Think of benchmark `.py` files as notebooks, not modules. They are never imported — they are always executed directly.

```python
# ❌ WRONG — do not do this
if __name__ == "__main__":
    my_task.run(kbench.llm)

# ✅ CORRECT — top-level, in its own cell
# %%
my_task.run(kbench.llm)
```

---

## §2. Defining Tasks

### `@kbench.task()` Parameters

```python
@kbench.task(
    name="optional_name",         # Defaults to function name, title-cased
    description="What it does",   # Defaults to docstring
    version=1,                    # Task version
    store_task=True,              # Set False for sub-tasks
    store_run=True,               # Set False to skip storing results
)
def my_task(llm):
    ...
```

`@kbench.benchmark()` is an exact alias for `@kbench.task()`.

### Task First Parameter

The first parameter **must** be the LLM actor. It receives the model to test.

```python
@kbench.task()
def my_task(llm):           # ✅ Correct
    ...

@kbench.task()
def my_task(llm, judge_llm): # ✅ Also fine — second LLM for judging
    ...
```

### Task Additional Parameters

Extra parameters are passed via `.run()` kwargs:

```python
@kbench.task()
def check_knowledge(llm, question, expected_answer):
    response = llm.prompt(question)
    kbench.assertions.assert_contains_regex(
        rf"(?i){expected_answer}", response
    )

check_knowledge.run(kbench.llm, question="Capital of Japan?", expected_answer="Tokyo")
```

### Return Types

**If your task returns a value, you MUST add a return type annotation.**

| Annotation | Result Type | Meaning |
|------------|-------------|---------|
| (none) or `-> None` | PassFail | Pass if no exceptions, based on assertions |
| `-> bool` | Boolean | True = pass, False = fail |
| `-> float` | Score | Numerical score |
| `-> int` | Numerical | Integer value |
| `-> dict` | Dictionary | Arbitrary dict result |
| `-> tuple[int, int]` | PassCount | Count (e.g., `(8, 10)`) |
| `-> tuple[float, float]` | MetricWithCI | Value ± confidence interval |

> **Note:** `-> None` is equivalent to omitting the annotation — both produce PassFail.

```python
# Score task
@kbench.task()
def accuracy(llm) -> float:
    return 0.85

# Count task
@kbench.task()
def count_correct(llm) -> tuple[int, int]:
    return (8, 10)  # 8 out of 10 passed

# Dict task (for rich results)
@kbench.task()
def detailed_result(llm) -> dict:
    return {"accuracy": 0.9, "latency": 1.2, "is_correct": True}
```

---

## §3. Running Tasks

### Running a Task

```python
# Single run — returns a Run object
run = my_task.run(kbench.llm)

# With extra parameters
run = my_task.run(kbench.llm, question="What is Python?")

# Multiple models
run1 = my_task.run(kbench.llm)         # Default model
run2 = my_task.run(kbench.judge_llm)   # Judge model
```

**Available models (loaded from Kaggle environment):**
- `kbench.llm` — default model
- `kbench.judge_llm` — judge model
- `kbench.llms` — list of ALL available models (useful for multi-model comparison)

### Run Object Properties

The `Run` object returned by `.run()` has useful attributes:

```python
run = my_task.run(kbench.llm)

run.passed              # bool — True if result + all assertions passed
run.result              # The returned value (type depends on task return annotation)
run.assertion_results   # list[AssertionResult] — all recorded assertions
run.status              # Status enum (PENDING, DONE, FAILED)
run.chat                # The conversation log
```

This is especially useful in sub-task composition:
```python
runs = [subtask.run(llm, q=q) for q in questions]
accuracy = sum(r.passed for r in runs) / len(runs)
```

### Batch Evaluation: `.evaluate()`

```python
import pandas as pd

results = my_task.evaluate(
    llm=[kbench.llm],                    # List of models
    evaluation_data=df,                   # DataFrame of test cases
    n_jobs=3,                             # Parallel workers (default: 1)
    timeout=120,                          # Per-job timeout in seconds
    max_attempts=3,                       # Retry count
    retry_delay=15,                       # Seconds between retries
    stop_condition=lambda runs: len(runs) == df.shape[0],  # Early stop
    remove_run_files=True,                # Clean up after
)

# Access results
results.as_dataframe()
```

> **Note:** Any extra keyword arguments (beyond `llm`, `evaluation_data`, etc.) are forwarded to the task function. For example, if your task has a `critic` parameter, pass `critic=[critic_llm]` to `.evaluate()`.

### Multi-Model Comparison

```python
models = [
    kbench.llms["google/gemini-2.5-flash"],
    kbench.llms["meta/llama-3.1-70b"],
]

# When using stop_condition with multiple models, account for all combinations:
n_total = len(models) * df.shape[0]
results = my_task.evaluate(
    llm=models,
    evaluation_data=df,
    n_jobs=3,
    stop_condition=lambda runs: len(runs) == n_total,
)
```

### Sub-Tasks Pattern

For nested evaluation (task calling sub-task):

```python
@kbench.task(name="single_qa", store_task=False)  # store_task=False for sub-tasks
def single_qa(llm, question, answer) -> dict:
    response = llm.prompt(question)
    return {"is_correct": answer.lower() in response.lower()}

@kbench.task(name="full_eval")
def full_eval(llm, df) -> tuple[float, float]:
    with kbench.client.enable_cache():
        runs = single_qa.evaluate(
            llm=[llm], evaluation_data=df,
            n_jobs=2, timeout=120, max_attempts=1,
            remove_run_files=True,
        )
    eval_df = runs.as_dataframe()
    accuracy = float(eval_df.result.str.get("is_correct").mean())
    std = float(eval_df.result.str.get("is_correct").std())
    return accuracy, std
```

---

## §4. LLM Interaction

### `llm.prompt()` — Primary method

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| *text* | `str` | — | The prompt text (required, first positional arg) |
| `schema` | `Type` | `str` | Structured output type (returns parsed object, not string) |
| `image` | `Image` | `None` | Image content |
| `video` | `Video` | `None` | Video content |
| `audio` | `Audio` | `None` | Audio content |
| `tools` | `list[Callable]` | `None` | Callable Python functions as tools |
| `reasoning` | `str` | `None` | Reasoning effort: `"none"`, `"low"`, `"medium"`, `"high"` |
| `seed` | `int` | `0` | Random seed for reproducibility |
| `temperature` | `float` | `0` | Temperature (0 = deterministic, higher = more creative) |

### Accessing Reasoning Traces

When using `reasoning=` parameter, access the model's thinking process:

```python
response = llm.prompt("Solve: 15 × 17", reasoning="medium")
traces = kbench.last_reasoning_traces()  # str | None — the model's "thinking"
kbench.assertions.assert_not_empty(response)
```

> `last_reasoning_traces()` returns `None` if the model didn't produce traces (e.g., reasoning was not enabled or the model doesn't support it).

```python
# Simple text → returns str
response = llm.prompt("What is 2+2?")

# Multi-turn: history maintained automatically within a task
llm.prompt("My name is Alice.")
response = llm.prompt("What is my name?")  # Remembers "Alice"
```

### Structured Output: Four Schema Styles

**Style 1: Dataclass (Preferred for complex types)**
```python
from dataclasses import dataclass

@dataclass
class Sentiment:
    label: str
    score: float

result = llm.prompt("Analyze: 'I love this!'", schema=Sentiment)
print(result.label, result.score)  # "positive", 0.95
```

**Style 2: Inline dict schema (Quick & simple)**
```python
result = llm.prompt(
    "9.9 - 9.11 = ?",
    schema={"answer": bool, "explanation": str},
)
print(result.answer, result.explanation)
```

**Style 3: Primitive type**
```python
count = llm.prompt("How many letters in 'hello'?", schema=int)  # returns int
is_yes = llm.prompt("Is the sky blue?", schema=bool)             # returns bool
text = llm.prompt("Summarize briefly.", schema=str)               # returns str
```

**Style 4: Pydantic model (with Field descriptions)**
```python
import pydantic

class Review(pydantic.BaseModel):
    sentiment: str = pydantic.Field(description="positive, negative, or neutral")
    score: float = pydantic.Field(description="confidence score 0-1")
    key_phrases: list[str] = pydantic.Field(description="notable phrases from the text")

result = llm.prompt("Analyze: 'Great movie!'", schema=Review)
# result.sentiment, result.score, result.key_phrases are all typed
```

> **Tip:** `Field(description=...)` helps the LLM understand what each field expects, improving extraction accuracy for complex schemas.

**When to use which:**
- **Dict schema**: Quick prototyping, simple key-value results
- **Dataclass**: Complex types with enums, nested types, or frozen immutability
- **Pydantic**: When you need validation rules or `Field(description=...)` hints
- **Primitive**: When you need a single value (bool, int, str)

### Multimodal Inputs

**Images — Two approaches:**

```python
from kaggle_benchmarks.content_types import images

# Approach A: via prompt() — PREFERRED (auto-converts URL to Base64)
img = images.from_url("https://example.com/photo.jpg")
response = llm.prompt("Describe this image", image=img)

# Approach B: via user.send() — for multi-turn / stacking multiple images
kbench.user.send(images.from_url("https://example.com/photo.jpg"))
kbench.user.send(images.from_path("local/chart.png"))
response = llm.prompt("Compare these images")
```

> **Prefer Approach A** — `llm.prompt(image=)` auto-converts URLs to Base64 for maximum compatibility.
> **Use Approach B** when you need to stack multiple images or build complex conversation history.
> Note: `user.send()` passes URLs as-is — the model must natively support URL inputs.

Image factories:
```python
img = images.from_url("https://example.com/photo.jpg")   # From URL
img = images.from_path("local/photo.png")                 # From local file
img = images.from_base64(b64_str, format="png")           # From Base64
img = images.from_array(numpy_array)                      # From NumPy array (requires Pillow)
b64 = images.image_url_to_base64("https://...")            # Download + convert helper
```

**Videos** (limited to specific models — Gemini 2.5+):
```python
from kaggle_benchmarks.content_types import videos
video = videos.from_url("https://www.youtube.com/watch?v=...")
response = llm.prompt("What happens in this video?", video=video)
```

**Audio** (limited to specific models — Gemini 2.0+):
```python
from kaggle_benchmarks.content_types import audios

# Three factory methods:
audio = audios.from_path("speech.mp3")                               # From local file
audio = audios.from_base64(b64_string, format="mp3")                  # From Base64
audio = audios.from_url("https://example.com/speech.mp3")             # From URL

response = llm.prompt("Transcribe this audio.", audio=audio)
```

### System Messages

**Two approaches:**

```python
# Approach A: via kbench.system.send() inside a task — PREFERRED for in-task system prompts
@kbench.task()
def code_analysis(llm):
    kbench.system.send("You are an expert Python programmer.")
    response = llm.prompt("Check this code for bugs...")

# Approach B: via chats.new(system_instructions=) — for new isolated conversations
with kbench.chats.new("pirate_chat", system_instructions="You are a pirate."):
    response = llm.prompt("Tell me about treasure.")
```

### Streaming

```python
llm.stream_responses = True  # Enable streaming before prompting
response = llm.prompt("Write a long story...")
```

### Temperature Control

```python
# Default: temperature=0 (deterministic, reproducible output)
response = llm.prompt("What is 2+2?")

# Higher temperature = more creative/varied responses
response = llm.prompt("Write a creative story about a cat.", temperature=0.7)

# Use temperature=0 (default) for factual/deterministic tasks
# Use temperature=0.5-1.0 for creative/generative tasks
```

### Reasoning Control

```python
response = llm.prompt("Solve: 127 * 53?", reasoning="high")
# Valid: "none", "low", "medium", "high"

traces = kbench.last_reasoning_traces()  # Access model's reasoning
```

---

## §5. Assertions

All assertions are under `kbench.assertions`. They **do NOT raise exceptions** by default — they record pass/fail results and execution continues.

### Built-in Assertions

```python
# Equality & Truth
kbench.assertions.assert_equal(expected, actual, expectation="...")
kbench.assertions.assert_true(expr, expectation="...")
kbench.assertions.assert_false(expr, expectation="...")

# Membership
kbench.assertions.assert_in(member, container, expectation="...")
kbench.assertions.assert_not_in(member, container, expectation="...")

# Emptiness
kbench.assertions.assert_empty(container, expectation="...")
kbench.assertions.assert_not_empty(container, expectation="...")

# Regex
kbench.assertions.assert_contains_regex(pattern, text, expectation="...", flags=re.NOFLAG)
kbench.assertions.assert_not_contains_regex(pattern, text, expectation="...", flags=re.NOFLAG)

# Exception safety
kbench.assertions.assert_raises_no_exceptions(callable_obj, expectation="...", *args, **kwargs)

# Unconditional failure
kbench.assertions.assert_fail(expectation="...")
```

### Choosing the Right Assertion

| Goal | Preferred Assertion |
|------|-------------------|
| Check exact value | `assert_equal(expected, actual)` |
| Check keyword in response | `assert_contains_regex(r"(?i)keyword", response)` — use `(?i)` for case-insensitive |
| Check absence of keyword | `assert_not_contains_regex(r"(?i)badword", response)` |
| Check membership | `assert_in("item", collection)` |
| Validate boolean condition | `assert_true(condition)` / `assert_false(condition)` |
| Signal unconditional failure | `assert_fail("reason")` — useful as fallback (e.g., judge returns None) |
| Validate no errors | `assert_raises_no_exceptions(fn)` |
| Subjective/open-ended evaluation | `assess_response_with_judge(criteria, response, judge)` |

### Assertions vs Python `assert`

```python
# ❌ Python assert — stops execution, not tracked
assert "Paris" in response

# ✅ Library assertion — recorded, execution continues
kbench.assertions.assert_in("Paris", response, expectation="Should mention Paris")

# Note: Python assert IS caught by the task runner (doesn't crash),
# but it won't be recorded with proper tracking.
```

### LLM-as-Judge (for subjective evaluation)

**Default schema (AssessReport):**
```python
assessment = kbench.assertions.assess_response_with_judge(
    criteria=[
        "The poem has exactly 3 lines.",
        "The syllable structure is 5-7-5.",
    ],
    response_text=response,
    judge_llm=kbench.judge_llm,
)

# ALWAYS check for None — returns None on failure
if assessment is None:
    kbench.assertions.assert_fail("Judge failed to respond.")
else:
    for result in assessment.results:
        kbench.assertions.assert_true(
            result.passed,
            expectation=f"'{result.criterion}': {result.reason}"
        )
```

**Custom schema:**
```python
@dataclasses.dataclass
class StoryCritique:
    overall_rating: int
    feedback: str
    passed_checks: list[str]

assessment = kbench.assertions.assess_response_with_judge(
    criteria=[...],
    response_text=story,
    judge_llm=kbench.judge_llm,
    prompt_fn=custom_prompt_fn,       # Custom prompt generator
    output_schema=StoryCritique,       # Custom output type
)
```

### Custom Assertions

```python
from kaggle_benchmarks.assertions import assertion_handler, AssertionResult

@assertion_handler()
def assert_word_count(text: str, min_w: int, max_w: int, expectation: str) -> AssertionResult:
    count = len(text.split())
    return AssertionResult(
        passed=(min_w <= count <= max_w),
        expectation=expectation,
    )

# Use like built-in assertions:
assert_word_count(response, 10, 100, "Response should be 10-100 words")
```

**Rules:**
- Return type **must** be annotated as `-> AssertionResult`
- Use `@assertion_handler(raises_assertion_error=True)` to raise on failure
- **Normalize inputs** inside your custom assertion (e.g., `.lower()`, `.strip()`) to make checks robust

---

## §6. Conversation Management

### Default: Automatic History

Within a task, `llm.prompt()` calls share history:

```python
@kbench.task()
def multi_turn(llm):
    llm.prompt("My favorite color is blue.")
    response = llm.prompt("What's my favorite color?")
    kbench.assertions.assert_contains_regex(r"(?i)blue", response)
```

### `chats.new()` — Isolated Conversation

Creates a clean conversation (no shared history):

```python
with kbench.chats.new("evaluation") as chat:
    judge_llm.prompt("Rate this response...")  # Clean slate
```

Parameters:
```python
kbench.chats.new(
    name="chat_name",                    # Display name
    system_instructions="You are ...",   # Optional system prompt
    orphan=False,                        # If True, don't nest in parent chat history
)
```

### `chats.fork()` — Copy Current History

Creates a new conversation starting with the current chat's history (the original chat is unaffected):

```python
# Build up some context
llm.prompt("My name is Alice and I'm a data scientist.")
llm.prompt("I work on NLP projects.")

# Branch the conversation — fork has full history, original continues separately
with kbench.chats.fork("hypothesis") as branch:
    # This prompt sees "Alice" + "NLP" context
    response = llm.prompt("Given my background, suggest a research topic.")
    # Anything said here does NOT affect the original conversation

# Back in original — still only has the two original messages
response = llm.prompt("What's my name?")  # Still remembers "Alice"
```

### `contexts.enter()` — Multi-Agent

For complex multi-agent scenarios:

```python
from kaggle_benchmarks import chats, contexts

agent_a_chat = chats.Chat(name="Agent A")
agent_b_chat = chats.Chat(name="Agent B")

with contexts.enter(chat=agent_a_chat):
    response_a = llm_a.prompt("Agent A's prompt...")

with contexts.enter(chat=agent_b_chat):
    response_b = llm_b.prompt("Agent B's prompt...")
```

### Choosing Conversation Strategy

| Scenario | Method |
|----------|--------|
| Default multi-turn | Automatic — just call `llm.prompt()` repeatedly |
| Judge evaluation | `chats.new("judge")` — no history leakage |
| System instructions for a section | `chats.new(system_instructions="...")` |
| Continue with shared history | `chats.fork("branch")` |
| Multiple agents with separate histories | `contexts.enter(chat=...)` |

---

## §7. Tools

### Python Code Execution — Two Approaches

**Approach A: Extract + Run (Preferred for code generation tasks)**
```python
response = llm.prompt("Write Python to calculate factorial of 10.")
code = kbench.tools.python.extract_code(response)
result = kbench.tools.python.script_runner.run_code(code)
kbench.assertions.assert_contains_regex("3628800", result.stdout)
kbench.assertions.assert_empty(result.stderr.strip(), "No errors expected")

# For programs that read stdin:
result = kbench.tools.python.script_runner.run_code(code, input="test input\n")
```

**Approach B: IPythonREPL (for expression evaluation)**
```python
repl = kbench.tools.python.IPythonREPL()
output = repl.invoke("2 + 2", is_visible_to_llm=False)
kbench.assertions.assert_equal(4, float(output.output))
```

### Web/HTML Testing

```python
with kbench.tools.web.Browser() as browser:
    html_code = kbench.tools.web.extract_html(response)
    snapshot = browser.take_snapshot(html_code, wait_before=5000, full_page=True)
    # snapshot.html — rendered HTML
    # snapshot.logs — console logs
```

### Custom Function Tools

Define plain Python functions with type hints and docstrings. Pass them via `tools=`.

```python
def run_simple_calculator(a: float, b: float, operator: str) -> float:
    """Calculates the result of an arithmetic operation. Supported operators: + - * /"""
    if operator == "+": return a + b
    if operator == "-": return a - b
    if operator == "*": return a * b
    if operator == "/": return a / b
    raise ValueError(f"Unknown operator: {operator}")

@kbench.task()
def calc_task(llm):
    response = llm.prompt("What is 50 plus 25?", tools=[run_simple_calculator])
    kbench.assertions.assert_contains_regex(r"75", response)
```

**Multiple tools — LLM selects the right one:**
```python
def add_tool(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b

def multiply_tool(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b

@kbench.task()
def multi_tool_task(llm):
    response = llm.prompt(
        "What is 12 multiplied by 34?",
        tools=[add_tool, multiply_tool],
    )
    kbench.assertions.assert_contains_regex(r"408", response)
```

**Tool error handling — tools can raise exceptions:**
```python
def flaky_tool() -> str:
    """This tool always fails with an error."""
    raise ValueError("Tool execution failed.")

@kbench.task()
def error_handling_task(llm):
    response = llm.prompt("Call the flaky_tool and report what happens.", tools=[flaky_tool])
    kbench.assertions.assert_contains_regex(r"(?i)error|failed", response)
```

> **Note:** Automatic tool calling is currently only supported via the `genai` API.
> For `openai` API, tools must be called manually (see `use_calculator_tool.py`).
>
> **Note:** `kbench.assertions.assert_tool_was_invoked(fn)` appears in golden tests
> but may not be available in all versions of the library. Check before using.

---

## §8. Model Loading, Dataset Evaluation, and Publishing

### Model Loading

#### Three approaches:

```python
# 1. Default model (Preferred — lets Kaggle platform manage model selection)
kbench.llm          # Default model
kbench.judge_llm    # Judge model

# 2. Named model from available models
kbench.llms["google/gemini-2.5-flash"]
kbench.llms["meta/llama-3.1-70b"]

# 3. Direct ModelProxy (for explicit API control)
from kaggle_benchmarks.kaggle import model_proxy
llm = model_proxy.ModelProxy(model="google/gemini-2.5-flash", api="genai")
llm = model_proxy.ModelProxy(model="google/gemini-2.5-flash", api="openai")
```

**When to use which:**
- **`kbench.llm`**: Default choice — portable across Kaggle's "Add Models" feature
- **`kbench.llms["..."]`**: When you need a specific model (e.g., vision, judge)
- **`ModelProxy`**: When you need to specify the API backend (genai vs openai)

### Dataset Evaluation

```python
import pandas as pd

@kbench.task()
def qa_task(llm, question, answer) -> bool:
    response = llm.prompt(question)
    return answer.lower() in response.lower()

df = pd.DataFrame([
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "Capital of France?", "answer": "Paris"},
])

# Task parameter names must match DataFrame column names
results = qa_task.evaluate(llm=[kbench.llm], evaluation_data=df)
print(results.as_dataframe())
```

### Caching

```python
with kbench.client.enable_cache():
    results = my_task.evaluate(llm=[kbench.llm], evaluation_data=df)
```

### Publishing to Leaderboard

```python
# Final cell of Kaggle notebook:
%choose my_main_task
```

Currently only one task per notebook for leaderboards.

### Environment Variables

- `MODEL_PROXY_URL` — Model proxy endpoint
- `MODEL_PROXY_API_KEY` — API key
- `KBENCH_EXECUTION_MODE` — `testing` for test mode
- `KBENCH_UI_MODE` — `panel`, `console`, or `none`

### Testing Your Tasks

#### Running with uv

```bash
source ~/ws/uv/bin/activate

uv pip install -e .
uv run python documentation/examples/simple_task.py
uv run --group test pytest tests/ -v
```

#### MockedChat for Unit Tests

```python
from tests.mocks import MockedChat

mock = MockedChat(responses=["Paris", "42"])
response1 = mock.prompt("Capital of France?")  # Returns "Paris"
response2 = mock.prompt("What is 6*7?")         # Returns "42"

# Verify what was sent
assert mock.invocations[0].messages[0].content == "Capital of France?"
```

---

## §9. Complete Example Patterns

### Pattern A: Simple Q&A — Regex Check

The most basic pattern. Good for factual questions with known keywords.

```python
import kaggle_benchmarks as kbench

@kbench.task(name="geography_quiz")
def geography_quiz(llm):
    response = llm.prompt("What is the longest river in the world?")
    kbench.assertions.assert_contains_regex(
        r"(?i)nile", response,
        expectation="Should mention the Nile river."
    )

geography_quiz.run(kbench.llm)
```

### Pattern B: Structured Output + Validation

For tasks needing parsed, validated responses.

```python
import kaggle_benchmarks as kbench
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    occupation: str

@kbench.task(name="extract_person")
def extract_person(llm, bio: str):
    person = llm.prompt(
        f"Extract the name, age, and occupation:\n\n{bio}",
        schema=Person
    )
    kbench.assertions.assert_equal("Marie Curie", person.name)
    kbench.assertions.assert_equal(66, person.age)
    kbench.assertions.assert_in("physicist", person.occupation.lower())

extract_person.run(kbench.llm, bio="Marie Curie was a physicist... born 1867, died 1934 at 66.")
```

### Pattern C: Hallucination Detection (Structured + Negative Assert)

Combining structured output with negative assertions to catch model hallucinations.

```python
@kbench.task("hallucination_check")
def check_hallucination(llm):
    response = llm.prompt(
        "When Richard Feynman mentioned gravity-light-contraction theory in his Nobel speech, did he think it was important?",
        schema={"answer": bool, "explanation": str},
    )
    kbench.assertions.assert_false(
        response.answer,
        expectation="Model should recognize fictitious theory.",
    )
    kbench.assertions.assert_contains_regex(
        r"(not|never|no|didn't)", response.explanation.lower(),
        expectation="Explanation should deny the theory exists.",
    )
```

### Pattern D: Judge-Based Evaluation with Error Handling

For open-ended tasks where deterministic checks aren't possible.

```python
@kbench.task(name="story_quality")
def story_quality(llm):
    story = llm.prompt("Write a one-paragraph story about a cat detective.")

    assessment = kbench.assertions.assess_response_with_judge(
        criteria=[
            "The story is exactly one paragraph.",
            "The main character is a cat.",
            "The cat is a detective.",
        ],
        response_text=story,
        judge_llm=kbench.judge_llm,
    )

    if assessment is None:
        kbench.assertions.assert_fail("Judge failed to respond.")
    else:
        for result in assessment.results:
            kbench.assertions.assert_true(
                result.passed,
                expectation=f"'{result.criterion}': {result.reason}"
            )

story_quality.run(kbench.llm)
```

### Pattern E: Code Generation + Execution

Combines prompting, code extraction, and programmatic validation.

```python
@kbench.task(name="solve_with_python")
def solve_with_python(llm):
    response = llm.prompt(
        "What is the 15th Fibonacci number? Write Python to calculate and print it."
    )
    code = kbench.tools.python.extract_code(response)
    result = kbench.tools.python.script_runner.run_code(code)

    kbench.assertions.assert_empty(
        result.stderr.strip(), "Code should run without errors."
    )
    kbench.assertions.assert_equal(
        "610", result.stdout.strip(), "Should print 610."
    )

solve_with_python.run(kbench.llm)
```

### Pattern F: Multi-Turn Game Loop

Interactive game with state tracking.

```python
@kbench.task(name="twenty_questions")
def twenty_questions(llm, judge_llm, target: str):
    from dataclasses import dataclass

    @dataclass
    class Response:
        question: str = ""
        guess: str = ""

    rules = f"Let's play 20 questions! I'm thinking of an animal. Ask yes/no questions."
    response = llm.prompt(rules, schema=Response)

    for i in range(20):
        if response.guess:
            kbench.assertions.assert_in(target, response.guess.lower())
            return True

        with kbench.chats.new("Answering"):
            yes = judge_llm.prompt(
                f"I'm thinking of {target}. Question: {response.question}",
                schema=bool,
            )

        answer = "Yes" if yes else "No"
        response = llm.prompt(f"{answer}. Guess or ask another?", schema=Response)

    return False

twenty_questions.run(kbench.llm, kbench.judge_llm, target="dog")
```

### Pattern G: Multi-Model Judging with Isolated Chats

Multiple judges scoring the same output, each in isolation.

```python
from dataclasses import dataclass

@dataclass
class PoemScore:
    score: float

@kbench.task(name="judge_poem")
def judge_poem(llm, question: str) -> float:
    judge1 = kbench.llms["google/gemini-2.5-pro"]
    judge2 = kbench.llms["meta/llama-3.1-70b"]

    with kbench.chats.new("writing"):
        poem = llm.prompt(question)

    with kbench.chats.new("judge_1"):
        score1 = judge1.prompt(f"Rate this poem 0-10:\n{poem}", schema=PoemScore)

    with kbench.chats.new("judge_2"):
        score2 = judge2.prompt(f"Rate this poem 0-10:\n{poem}", schema=PoemScore)

    return (score1.score + score2.score) / 2

judge_poem.run(kbench.llm, question="Write a haiku about clouds.")
```

### Pattern H: Dataset Evaluation with Parallel Execution

```python
import pandas as pd

@kbench.task()
def riddle_solver(llm, riddle: str, answer_keyword: str) -> bool:
    response = llm.prompt(riddle)
    is_correct = answer_keyword.lower() in response.lower()
    kbench.assertions.assert_true(is_correct)
    return is_correct

df = pd.DataFrame({
    "riddle": ["I have cities but no houses. What am I?", "What has an eye but cannot see?"],
    "answer_keyword": ["map", "needle"],
})

runs = riddle_solver.evaluate(
    llm=[kbench.llm], evaluation_data=df, n_jobs=3
)
runs.as_dataframe()
```

### Pattern I: Code Analysis with System Prompt + Tools

Combining system messages, structured output, and code execution.

```python
from dataclasses import dataclass

@dataclass
class CodeAnalysis:
    has_bugs: bool
    fixed_code: str

@kbench.task("code_analysis")
def analyze_code(llm):
    buggy_code = """
fruits = ['apple', 'orange' 'banana', 'peach']
print(len(fruits))
"""
    kbench.system.send("You are an expert Python programmer.")
    response = llm.prompt(
        f"Does this code have bugs? Fix it.\n{buggy_code}",
        schema=CodeAnalysis,
    )
    kbench.assertions.assert_true(response.has_bugs, "Should detect the missing comma.")

    fixed = kbench.tools.python.extract_code(response.fixed_code)
    output = kbench.tools.python.script_runner.run_code(fixed)
    kbench.assertions.assert_equal("4", output.stdout.strip(), "Fixed code outputs 4.")
```

---

## Related Skills

- **`kaggle-cli`** — Covers using the `kaggle` CLI to manage datasets, notebooks, and submit benchmarks to Kaggle. Use that skill after writing your benchmark code with this one.
