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

## Minimal Example

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
| Using `kbench.llm` locally without Kaggle configured | Use model proxy or set env vars |
| Forgetting `schema=` when needing structured output | Pass `schema=MyDataclass` to `llm.prompt()` |
| Wrapping `.run()` / `.evaluate()` in `if __name__ == "__main__":` | Place them at module top level — benchmark files are scripts, not importable modules |
| Using `user.send()` with image URLs | `user.send()` passes URLs as-is; prefer `llm.prompt(image=)` for auto-conversion |
| Not isolating judge conversations | Use `with kbench.chats.new("judge"):` |
| Multiple tasks sharing conversation history | Each `.run()` creates its own conversation |
| Using `store_task=True` for sub-tasks | Set `store_task=False` for helper tasks called inside other tasks |
| Using `!pip install` without commenting | Use `# !pip install -q pkg` — uncommented magics break local execution |
| Forgetting `last_reasoning_traces()` can be `None` | Always check: `traces = kbench.last_reasoning_traces(); if traces: ...` |

## Detailed References

For full API details, patterns, and examples, read the reference files in `references/`:

| File | Contents |
|------|----------|
| `references/tasks_and_running.md` | Import styles, `@kbench.task()` parameters, return types, `.run()`, `.evaluate()`, sub-tasks |
| `references/llm_interaction.md` | `llm.prompt()` API, structured output (dataclass/dict/pydantic), multimodal (images/video/audio), system messages, streaming, reasoning |
| `references/assertions.md` | All built-in assertions, LLM-as-judge, custom assertions |
| `references/conversations.md` | `chats.new()`, `chats.fork()`, `contexts.enter()`, conversation strategies |
| `references/tools.md` | Python code execution, web/HTML testing, custom function tools |
| `references/advanced.md` | Model loading, dataset evaluation, caching, leaderboard publishing, testing, environment variables |
| `references/examples.md` | 9 complete example patterns (simple Q&A, structured output, hallucination detection, judge evaluation, code generation, multi-turn games, multi-model judging, dataset evaluation, code analysis) |

## Related Skills

- **`kaggle-cli`** — Covers using the `kaggle` CLI to manage datasets, notebooks, and submit benchmarks to Kaggle. Use that skill after writing your benchmark code with this one.
