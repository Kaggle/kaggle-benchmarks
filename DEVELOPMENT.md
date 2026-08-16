# Development Guide for Kaggle Benchmarks

A practical guide for contributors — covering architecture, conventions, and common pitfalls.

---

## 1. Project Overview

`kaggle-benchmarks` is a Python **library** (not an application or service) for evaluating LLMs using decorators, assertions, and tool-augmented interactions. Users write decorated functions (`@kbench.task`) that prompt LLMs and assert outputs; the library handles orchestration, caching, serialization, and UI rendering. The codebase is organized around reusable primitives, not around endpoints or CLI commands.

### Key Subsystems

The source lives in `src/kaggle_benchmarks/`. Here are the main subsystems — browse the directory for the full picture.

| Subsystem | Key Modules | Responsibility |
|-----------|------------|----------------|
| **Core primitives** | `tasks.py`, `messages.py`, `runs.py`, `results.py` | Task/benchmark decorators, message types, execution records |
| **LLM interaction** | `actors/llms.py` (+ `actors/openai.py`, `actors/genai.py`), `chats.py`, `prompting.py` | LLM abstraction — base `LLMChat` (`llms.py`) with the `OpenAI` and `GoogleGenAI` backends in their own modules (re-exported from `llms`), conversation management, schema processing |
| **Serialization** | `serializers/` | Translating messages to/from provider-specific API payloads |
| **Content types** | `content_types/` | Provider-agnostic data models for images, audio, video |
| **Configuration** | `_config.py` | Centralized `Config` dataclass, `ExecutionMode` enum |
| **Platform integration** | `kaggle/`, `envs/` | Model loading, protobuf serialization, execution environments |
| **Tooling** | `tools/` | Python interpreter, web search, tool invocation |
| **UI** | `ui/`, `events.py` | Panel-based notebook rendering, lifecycle event hooks |

Other top-level directories: `tests/` (mirrors `src/` structure), `golden_tests/` (end-to-end tests requiring API keys), `protos/` (Protocol Buffer schemas), `documentation/` (Quarto-based docs).

### Key Mental Model

The library has a clear **layered architecture** with strict separation of concerns:

```
User Code  →  @task / @benchmark decorators
                   ↓
             LLMChat.prompt()  →  Actor/LLM abstraction
                   ↓
             Serializers       →  Message ↔ API payload translation
                   ↓
             Provider SDKs     →  OpenAI / Google GenAI
```

Each layer has a single responsibility. Understanding which layer owns a piece of logic is critical for placing new code correctly.

---

## 2. Architecture & Design Principles

### Content Types Are Provider-Agnostic

Content objects (`ImageContent`, `VideoContent`, `AudioContent` in `content_types/`) represent **user data**. They must never contain provider-specific fields. If OpenAI needs a `detail` parameter on images or GenAI needs a `video_metadata` dict, that logic belongs in the corresponding serializer — not in the content type itself.

```python
# ❌ DON'T — this leaks OpenAI's API into a shared type
class ImageContent:
    detail: str = "auto"  # OpenAI-specific

# ✅ DO — content types stay clean; serializers handle provider logic
class ImageContent:
    """Provider-agnostic image representation."""
    ...
```

### Serialization Belongs in Serializers

The codebase has a dedicated serializer layer built on `BaseSerializer` (in `serializers/base.py`). All message-to-API-payload translation happens here via dynamic dispatch on content type (`dump_text_message`, `dump_image`, `dump_video`, `dump_audio`, etc.). Never put serialization logic inline in client code.

```python
# ❌ DON'T — serialization inline in the client
class MyClient:
    def send(self, msg):
        payload = {"parts": [{"text": msg.content}]}

# ✅ DO — delegate to the serializer
class MyClient:
    def send(self, msg):
        payload = list(self.serializer.dump_messages([msg]))
```

To add support for a new content type:
1. Define the content class in `content_types/`
2. Add a `dump_*` method to `BaseSerializer`
3. Implement it in each concrete serializer (`GenAISerializer`, `ModelProxyOpenAISerializer`)

### Prefer Subclasses Over Generic `**kwargs`

When adding new modality support or specialized behavior, prefer **dedicated subclasses** with explicit parameters over generic `**kwargs` passthrough. This gives you stronger typing, better discoverability, and a clear place for specialized logic.

```python
# ❌ DON'T — generic kwargs are hard to validate and discover
video = videos.from_url("https://youtube.com/...", video_metadata={"start_offset": "0s"})

# ✅ DO — dedicated factory with explicit parameters
video = videos.from_youtube("https://youtube.com/...", start_offset=0, end_offset=10)
```

### LLMChat Is Stateless

`LLMChat` (in `actors/llms.py`) does not store conversation history, system instructions, or temperature settings. All state lives in the `chats.Chat` object, enabling clean nested threads where inner history is invisible to outer conversations. Respect this design — don't add stateful fields to `LLMChat`.

### Keep the Public API Surface Small and Intentional

Everything exported from `__init__.py` is the library's public contract. Once a parameter is added to `prompt()` or other user-facing methods, it's very hard to remove. Think carefully before expanding the API surface.

```python
# ⚠️ THINK TWICE — hard to change once adopted
llm.prompt("...", api_params={"detail": "low"})

# 💡 Consider alternatives for experimental features (context managers, config, etc.)
```

### Avoid Backend-Specific Coupling

The library supports multiple LLM backends (OpenAI, GenAI). Don't hardcode assumptions about a single provider. Configuration validation should happen at the point of use (e.g., when an LLM is instantiated), not at import time.

### Abstract Provider Differences Behind Unified Parameters

When the same concept has different names across providers (e.g., OpenAI's `reasoning_effort` vs GenAI's `thinking_config`), introduce a single unified parameter that the SDK maps to provider-specific configurations internally. Users should never need `if/elif` branches per provider.

```python
# ❌ DON'T — forces users to know provider internals
if llm.api == "openai":
    llm.prompt("...", reasoning_effort="high")
elif llm.api == "genai":
    llm.prompt("...", thinking_config={"thinking_level": "HIGH"})

# ✅ DO — SDK handles the mapping
llm.prompt("...", reasoning="high")
```

### Keep Response Content Clean

Response content (`message.content`) is consumed by structured output parsers, assertions, and user code. Be very careful not to pollute it with metadata or internal artifacts — anything mixed in can break JSON parsing or produce misleading assertion results.

**Example:** When LLMs produce reasoning/thinking traces, they must be stored in `LLMMessage.thinking` (see `llm_messages.py`), not appended to `content`. If thinking text like `<think>The answer might be 42...</think>` ends up in `content`, a structured output parser will fail on the invalid JSON, and an `assert_contains("42", response)` will pass even when the final answer is wrong.

---

## 3. Configuration

### Use the `Config` Dataclass

All configuration flows through the central `Config` dataclass in `_config.py`. Don't scatter `os.getenv()` calls across modules. The established pattern uses `dataclasses.field(default_factory=...)` to read env vars with sensible defaults:

```python
@dataclasses.dataclass()
class Config:
    # ✅ Established pattern — centralized, with a default
    max_name_length: int = dataclasses.field(
        default_factory=lambda: int(os.getenv("KBENCH_MAX_NAME_LENGTH", "100"))
    )
```

The config singleton is created at module level (`config = Config()`) and applied via `config.apply()` during library initialization. The `.env` file is loaded automatically via `python-dotenv`.

### ExecutionMode

The `ExecutionMode` enum (in `_config.py`) controls library behavior across different contexts:

| Mode | Use Case |
|------|----------|
| `NOTEBOOK` / `INTERACTIVE` | Interactive notebook sessions |
| `RUN` / `TRIAL` | Batch evaluation runs |
| `TESTING` | Unit tests (mocked clients, no UI) |
| `DEV` | Local development |
| `DOC` | Documentation generation |

In tests, always set `config.execution_mode = ExecutionMode.TESTING` (the `conftest.py` fixture does this automatically).

---

## 4. Testing

### Test Structure

Tests live in `tests/` and mirror the source layout. The naming convention is `test_<module>.py` for top-level modules and subdirectories for subsystems (e.g., `tests/serializers/`, `tests/actors/`).

### The `context` Fixture

Every test automatically uses the autouse `context` fixture from `tests/conftest.py`, which:
- Enters a fresh execution context
- Sets `ExecutionMode.TESTING`
- Disables interactive mode and UI
- Patches the client with `InMemoryClient`

You don't need to set any of this up manually in your tests.

### Mocking LLMs

Use `MockedChat` (in `tests/mocks.py`) to simulate LLM responses without making real API calls:

```python
from tests.mocks import MockedChat

# Single response
llm = MockedChat.from_contents(["the answer is 42"])

# Cycling responses (for multi-turn)
llm = MockedChat.from_contents(["first", "second"], cycle=True)

# JSON responses
llm = MockedChat.from_contents_data([{"name": "Alice", "age": 30}])
```

The `conftest.py` provides pre-built fixtures: `duck` (responds "quack") and `goose` (responds "honk").

### Cross-Provider Tests

When a feature needs to work across both OpenAI and GenAI backends, write separate test cases for each serializer:

```
tests/serializers/test_genai_serializer.py
tests/serializers/test_openai_serializer.py
tests/test_genai_client.py
tests/test_openai_client.py
```

### Golden Tests

Golden tests (in `golden_tests/test_cookbook_examples.py`) validate end-to-end behavior against real LLM APIs. They are **not** part of the standard CI/CD pipeline (they require API keys).

If your change touches serializers, actors, or the prompt pipeline, verify that golden tests still pass:

```bash
# Run all golden tests
uv run pytest golden_tests/test_cookbook_examples.py

# Run and update the report
uv run pytest golden_tests/test_cookbook_examples.py --generate-report

# Filter by API
uv run pytest golden_tests/test_cookbook_examples.py -k "genai"
```

### Running Unit Tests

```bash
# All unit tests
uv run --group test pytest tests

# Specific file
uv run --group test pytest tests/test_assertions.py

# Verbose
uv run --group test pytest tests -v
```

---

## 5. Code Style & Tooling

This project follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) as its baseline coding standard. The tooling below automates and enforces many of these conventions.

### Formatting and Linting

The project uses **ruff** for both formatting and linting. Always run these before committing:

```bash
ruff format .          # Format code
ruff check --fix .     # Lint + auto-fix (includes import sorting via isort)
```

Key ruff configuration from `pyproject.toml`:
- `extend-select = ["I"]` — isort-compatible import sorting is enforced
- `extend-exclude` — auto-generated `*_pb2.py` files are excluded
- `__init__.py` files suppress `F401` (unused imports) and `F402` (import shadowing)

### Pre-commit Hooks

Install and use pre-commit hooks — they run `ruff`, `ruff-format`, and `addlicense` automatically:

```bash
pre-commit install
pre-commit run --all-files
```

### License Headers

Every `.py` file **must** have the Apache 2.0 license header. The `addlicense` pre-commit hook handles this automatically. The header format:

```python
# Copyright <year> Kaggle Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
```

### Type Checking

The project uses **mypy**. Generated protobuf files (`*_pb2.py`) are excluded from type checking. Run with:

```bash
mypy src/
```

### Dependency Management

Use **`uv`** for all dependency operations — never `pip` directly. Dependencies are organized into groups in `pyproject.toml`:

| Group | Purpose |
|-------|---------|
| `test` | pytest and test utilities |
| `kaggle` | Kaggle platform integration |
| `docs` | Documentation example dependencies |
| `dev` | All of the above + mypy, ruff, pre-commit |

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run a command with a specific group
uv run --group test pytest tests
```

> **Supply chain hardening:** The project pins an `exclude-newer` date in `pyproject.toml` to refuse packages published after that date. To upgrade dependencies, bump the date, then run `uv lock --upgrade`.

---

## 6. Import Conventions

### Lazy Imports to Avoid Circular Dependencies

The codebase uses lazy imports inside methods to break circular dependencies — especially for `ui`, `events`, `contexts`, and `llm_messages`:

```python
def respond(self, ...):
    from kaggle_benchmarks import contexts, llm_messages  # Lazy import
    ...
```

### `TYPE_CHECKING` for Type-Only Imports

Use `if TYPE_CHECKING:` for imports that are only needed for type annotations. This is the standard pattern throughout the codebase:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaggle_benchmarks import actors
```

### Gotcha: Conditional Objects in `__init__.py`

Some module-level objects (`kbench.llm`, `kbench.judge_llm`, `kbench.llms`) are only created when the Kaggle platform is configured. If you're writing examples or tests that reference these, be aware they may not exist in all environments:

```python
# ✅ Always works — import the submodule directly
from kaggle_benchmarks import actors
llm_chat = actors.LLMChat(...)

# ⚠️ Only works when Kaggle is configured
import kaggle_benchmarks as kbench
kbench.llm  # AttributeError if not configured
```

---

## 7. Error Handling Conventions

### Exception Hierarchy

The codebase defines a few custom exceptions with specific semantics:

| Exception | When to Use |
|-----------|-------------|
| `NonRecoverableError` | Stops task execution entirely — the task cannot proceed |
| `SchemaError` | Response parsing failures (e.g., LLM returned invalid JSON) |
| `UnsupportedMessageFormat` | Serializer encounters an unknown message/content type |
| `GetAssertExpressionError` | Internal — assertion source code retrieval failed |

### Use `warnings.warn()` for Non-Fatal Issues

For degraded functionality or deprecated behavior, use `warnings.warn()` instead of silently dropping data or raising an exception:

```python
# ✅ User sees a clear message about what happened
warnings.warn("Reasoning traces are not available in streaming mode.")

# ❌ DON'T — silently dropping data is confusing
if not supported:
    return  # Where did my data go?
```

### Use `logging` for Operational Messages

Use the standard `logging` module for informational and diagnostic messages. Create a module-level logger:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Loading environment variables from %s", path)
logger.warning("Reading cached run failed: %s", error)
```

---

## 8. Protocol Buffers

Proto definitions live in `protos/`. The generated Python files (`*_pb2.py`) are output to `src/kaggle_benchmarks/kaggle/`.

**Never manually edit `*_pb2.py` files.** They are auto-generated. After editing any `.proto` file, rebuild with:

```bash
cd protos && ./build.sh
```

The build script runs `protoc` with both `--python_out` and `--mypy_out` to generate type stubs alongside the implementation.

---

## 9. Writing New Assertions

Custom assertions use the `@assertion_handler` decorator (in `assertions.py`). The decorated function **must** be annotated to return `AssertionResult`:

```python
@assertions.assertion_handler()
def assert_response_valid(response: str, expectation: str = "") -> assertions.AssertionResult:
    """Verifies that the response is well-formed."""
    passed = len(response) > 0 and not response.startswith("Error")
    return assertions.AssertionResult(
        passed=passed,
        expectation=expectation or "Response should be non-empty and not an error",
    )
```

The decorator automatically:
- Captures the source code line that called the assertion
- Reports results to the current run's `assertion_results`
- Optionally raises `AssertionError` if `raises_assertion_error=True`

---

## 10. General Principles

These are the recurring themes in code reviews. The examples are illustrative — the underlying principles apply broadly.

### Respect Separation of Concerns

Put code in the layer that owns that responsibility. If you're unsure, refer to the layered architecture diagram in §1.

- **Example:** Provider-specific logic (like OpenAI's `detail` parameter on images) belongs in serializers, not in content types. Content types are provider-agnostic data containers.
- **Example:** Scattered `os.getenv()` calls belong in the centralized `Config` dataclass, not spread across individual modules.

### Don't Repeat Expensive Work

If something can be computed once and reused, do so — especially for module-level setup.

```python
# ✅ Compile once at module level
_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)

def strip_tags(text):
    return _PATTERN.sub("", text)
```

### Validate Ambiguous Inputs

When a function accepts the same setting through multiple paths (e.g., a named parameter and a raw dict), validate for conflicts rather than silently picking one. Silent precedence rules are a source of bugs.

### Document Workarounds

If you introduce a temporary pattern or known limitation, leave a clear comment explaining why it exists and what the long-term fix looks like:

```python
# TODO: Remove this _meta workaround once respond() creates LLMMessage
# instead of Message. At that point, LLMMessage.reasoning_traces
# (the dataclass field) will be the single source of truth.
msg._meta["reasoning_traces"] = traces
```

### Keep Documentation in Sync

User-facing and public-API changes should include corresponding documentation updates in the same PR. Internal refactors don't necessarily need doc changes.

---

## Quick Reference Checklist

Before opening a PR, verify:

- [ ] `ruff format .` and `ruff check --fix .` pass cleanly
- [ ] `uv run --group test pytest tests` — all unit tests pass
- [ ] Golden tests verified if you touched serializers/actors/prompt pipeline
- [ ] License header present on all `.py` files (`pre-commit run --all-files`)
- [ ] New code lives in the correct architectural layer
- [ ] Public API additions are intentional — they're permanent once shipped
- [ ] Cross-provider tests exist for features that touch the LLM pipeline
- [ ] Workarounds have explanatory comments
- [ ] User-facing changes include documentation updates
- [ ] Protobuf files regenerated if you edited any `.proto` file
