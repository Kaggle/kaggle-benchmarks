# Assertions

All assertions are under `kbench.assertions`. They **do NOT raise exceptions** by default — they record pass/fail results and execution continues.

## Built-in Assertions

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

## Choosing the Right Assertion

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

## Assertions vs Python `assert`

```python
# ❌ Python assert — stops execution, not tracked
assert "Paris" in response

# ✅ Library assertion — recorded, execution continues
kbench.assertions.assert_in("Paris", response, expectation="Should mention Paris")

# Note: Python assert IS caught by the task runner (doesn't crash),
# but it won't be recorded with proper tracking.
```

## LLM-as-Judge (for subjective evaluation)

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

## Custom Assertions

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
