# Reviewing Code for Kaggle Benchmarks

A guide for reviewers — what to look for, how to give feedback, and when to block.

---

## The Reviewer's Job

Your goal is to keep the codebase **correct, consistent, and maintainable** while unblocking the contributor as quickly as possible. A good review does three things:

1. **Catches bugs and design problems** before they ship
2. **Teaches** — the contributor (and future readers) learn from your feedback
3. **Stays focused** — reviews the code that's there, not the code you wish were there

---

## What to Look For

### 1. Architecture & Layering

This is the most common source of review feedback. The library has strict boundaries between layers:

| Layer | Owns | Does NOT Own |
|-------|------|-------------|
| `content_types/` | Data representation (images, audio, video) | Provider-specific serialization |
| `serializers/` | Message ↔ API payload translation | Business logic, LLM interaction |
| `actors/llms.py` | LLM abstraction, `prompt()`, `respond()` | Conversation state (that's `chats.py`) |
| `_config.py` | Centralized configuration | Per-module env var reads |

**Ask yourself:** Does this code live in the right layer? If a provider-specific concept (like OpenAI's `detail` on images) is showing up in `content_types/`, that's a red flag — it belongs in the serializer.

### 2. API Surface

Changes to public-facing methods (`prompt()`, `assert_*`, `@task`) are effectively permanent. Scrutinize them more carefully than internal changes.

**Questions to ask:**
- Is this parameter necessary, or can the same goal be achieved via configuration or a context manager?
- Does the parameter name match existing conventions? (e.g., `reasoning`, not `reasoning_effort` or `thinking_config`)
- Will this be confusing alongside existing parameters?
- If this is experimental, is it marked as such?

### 3. Correctness

- **Edge cases:** What happens with `None`, empty strings, malformed input? Ask explicitly — e.g., "if the input is malformed, does this return empty or raise?"
- **Streaming vs non-streaming:** Features that work in non-streaming mode may silently break in streaming. If the PR touches the response pipeline, check both paths.
- **Thinking traces:** If the change involves reasoning/thinking, verify that traces are separated from response content. Mixing them corrupts structured outputs and breaks assertions.
- **Cross-provider behavior:** If it works on OpenAI, does it also work on GenAI? Are there tests for both serializer paths?

### 4. Code Quality

- **Workarounds documented:** If the code uses a temporary pattern or known limitation, is there a comment (e.g., `TODO`) explaining why and what the long-term fix looks like? Undocumented workarounds become mysteries for the next contributor.
- **Performance-conscious patterns:** Watch for repeated work that could be done once (e.g., compiling regex patterns, re-reading config, redundant API calls). These aren't always blockers, but worth a `nit:` comment.
- **No conflicting inputs:** If a function accepts the same setting through multiple paths (e.g., both a named parameter and a raw dict), is there validation to catch conflicts? Silent precedence rules are a source of bugs.
- **Error handling:** Non-fatal issues should use `warnings.warn()`. Internal diagnostics should use `logging`. Recoverable errors should raise specific exceptions, not generic `Exception`.

### 5. Tests

- **Mocked correctly?** Tests should use `MockedChat`, not real API calls (those belong in `golden_tests/`).
- **Cross-provider coverage?** Changes to the LLM pipeline should have tests in both `test_genai_serializer.py` and `test_openai_serializer.py`.
- **Golden tests acknowledged?** If the change touches serializers/actors/prompt pipeline, the contributor should confirm golden tests still pass (even if they can't run in CI).

### 6. Documentation

- **Docs updated?** Every user-visible change needs corresponding documentation updates. Don't accept "I'll do it in a follow-up."
- **Right location?** Stable features go in the cookbook. Experimental features go in an advanced usage section with a disclaimer.
- **Examples work?** If the PR adds or modifies example code, verify the imports are correct (watch out for conditional initialization in `__init__.py`).

---

## How to Give Feedback

### Be Specific and Actionable

```
# ❌ Vague
"This doesn't feel right."

# ✅ Specific
"I think `api_params` belongs in the serializer, not in ImageContent.
The content type should stay provider-agnostic. Could you move the
parameter handling to GenAISerializer.dump_image() instead?"
```

### Distinguish Severity

Use prefixes to signal how important a comment is:

- **`nit:`** — Style or minor improvement. Not a blocker. (e.g., "nit: `Verifies` instead of `Verify`?")
- **`suggestion:`** — An alternative approach worth considering, but the current code is acceptable.
- **`question:`** — You need clarification before you can approve. (e.g., "Just to confirm, should this be `not in (None, 'none')`?")
- **(no prefix)** — A problem that should be fixed before merging.

### Ask Questions, Don't Just Assert

When you're unsure about intent, ask before prescribing:

```
# ✅ Good
"Just to confirm my understanding — we'll switch to reasoning_traces
when we move from Message to LLMMessage later? If so, worth a TODO here."

# ❌ Less good
"This is wrong, it should use LLMMessage.reasoning_traces."
```

### Acknowledge Good Work

A quick "LGTM — thanks for making this more robust!" goes a long way, especially on PRs that went through multiple revision rounds.

---

## Handling Scope and Disagreements

### Keep PRs Focused

If a reviewer spots an improvement opportunity that's unrelated to the PR's goal, don't ask the contributor to fix it in the same PR. Instead:

- File it as a separate issue or note it as a follow-up
- Comment: "This is unrelated to this PR, but we should discuss [X] separately"

### When You Disagree with the Approach

1. **State your concern clearly** with a concrete alternative
2. **Listen to the response** — the contributor may have context you don't
3. **Escalate only if it's architectural** — if it affects the library's public API or layering, bring in a second reviewer
4. **Accept "good enough"** for non-architectural concerns — perfect is the enemy of shipped

### When to Block vs. Approve-with-Comments

**Block (Request Changes)** when:
- The change breaks existing behavior or tests
- Provider-specific logic is leaking into the wrong layer
- The public API surface is expanding without clear justification
- Thinking traces are mixed into response content
- Tests are missing for a new code path

**Approve with Comments** when:
- Only nits and style suggestions remain
- The approach is sound but has minor rough edges
- Documentation could be slightly better but isn't misleading
- You've suggested a TODO and the contributor agrees to add it

---

## Review Checklist

A quick reference for each review pass:

```
Architecture
  □ Code lives in the correct layer (content types / serializers / actors)
  □ No provider-specific logic in content types
  □ LLMChat stays stateless — no new state stored on the actor

API Surface
  □ New public parameters are intentional and well-named
  □ Naming is consistent with existing conventions
  □ Experimental features are clearly marked

Correctness
  □ Edge cases handled (None, empty, malformed)
  □ Streaming and non-streaming paths both work
  □ Thinking traces separated from content
  □ No silent data loss

Tests
  □ Unit tests cover the new/changed behavior
  □ Cross-provider tests exist (GenAI + OpenAI)
  □ Golden test impact acknowledged

Code Quality
  □ Workarounds have explanatory comments (TODOs, etc.)
  □ No obvious repeated work that could be done once
  □ Config uses the Config dataclass, not scattered os.getenv()
  □ License header present

Documentation
  □ Docs updated alongside the feature
  □ Experimental features in advanced section, not cookbook
  □ Examples use correct imports
```
