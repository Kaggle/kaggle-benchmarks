# LLM Interaction

## `llm.prompt()` — Primary method

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

## Accessing Reasoning Traces

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

## Structured Output: Four Schema Styles

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

## Multimodal Inputs

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

## System Messages

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

## Streaming

```python
llm.stream_responses = True  # Enable streaming before prompting
response = llm.prompt("Write a long story...")
```

## Temperature Control

```python
# Default: temperature=0 (deterministic, reproducible output)
response = llm.prompt("What is 2+2?")

# Higher temperature = more creative/varied responses
response = llm.prompt("Write a creative story about a cat.", temperature=0.7)

# Use temperature=0 (default) for factual/deterministic tasks
# Use temperature=0.5-1.0 for creative/generative tasks
```

## Reasoning Control

```python
response = llm.prompt("Solve: 127 * 53?", reasoning="high")
# Valid: "none", "low", "medium", "high"

traces = kbench.last_reasoning_traces()  # Access model's reasoning
```
