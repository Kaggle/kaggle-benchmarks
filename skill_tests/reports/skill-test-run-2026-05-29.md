# kaggle-benchmarks Skill Test Run — 2026-05-29

All **54** scenarios from [`skill_tests/agent_test_scenarios.md`](../agent_test_scenarios.md) were run end-to-end against [`skills/kaggle-benchmarks/SKILL.md`](../../skills/kaggle-benchmarks/SKILL.md). The skill file was loaded as the system prompt for a Claude Sonnet 4.6 sub-agent; each scenario's prompt was sent as the user message; each criterion was scored by a second Claude Sonnet 4.6 call grounded with the cited Source-of-Truth excerpt. Cat-1–3 code execution was **not** enabled (no live `kaggle_benchmarks` LLM backend was configured).

- Generated: 2026-05-29 21:53 UTC
- Agent model: `claude-sonnet-4-6@default`
- Judge model: `claude-sonnet-4-6@default`
- `--execute` enabled: no

## Headline numbers

| Rating | Count | % of 54 |
|---|---|---|
| ✅ Strong Pass | 28 | 52% |
| ⚠️ Partial Pass | 23 | 43% |
| ❌ Fail | 3 | 6% |

At the criterion level: **259 / 309 (84%)** individual checks passed.

### By category

| Cat | Description | ✅ | ⚠️ | ❌ | Criteria passed |
|---|---|---|---|---|---|
| 1 | Basic — Simple Task + Assertion | 8 | 5 | 0 | 58/64 (91%) |
| 2 | Basic — Function Tool Use | 3 | 0 | 0 | 14/14 (100%) |
| 3 | Basic — Code Execution | 1 | 1 | 0 | 11/12 (92%) |
| 4 | Dataset Evaluation | 2 | 2 | 1 | 28/36 (78%) |
| 5 | Medium — Combining Basics | 4 | 4 | 0 | 45/51 (88%) |
| 6 | Knowledge & Troubleshooting | 8 | 2 | 1 | 45/49 (92%) |
| 7 | Generalization — Domain-Specific Tasks | 2 | 9 | 1 | 58/83 (70%) |

## Top SKILL.md gaps surfaced by the failures

The following weaknesses recur across multiple failed criteria and represent
actionable improvements to `SKILL.md`:

1. **Judge-task signature & invocation.** When a task uses a judge LLM, the
   skill file shows `kbench.assertions.assess_response_with_judge(...)` but
   does not consistently demonstrate the **two-LLM-parameter pattern**: tasks
   that use a judge should declare `def task(llm, judge_llm)` and be invoked
   as `task.run(kbench.llm, kbench.judge_llm)`. In scenarios 5.1 and 7.8 the
   agent reached for `kbench.judge_llm` as a module-global inside the task
   body and called `.run(kbench.llm)` with one argument. **Fix:** add an
   explicit "Tasks that use a judge" subsection to §5 / §6 showing the
   parameter-passing form, and call this out in the Key Rules.

2. **Multi-model comparison with `.evaluate()`.** Scenarios 4.4 and 7.3
   both failed/regressed because the agent ran each model with a separate
   `.run()` instead of `.evaluate(llm=[model_a, model_b], evaluation_data=df)`.
   The skill file has multi-model in the reference list but lacks a full
   worked example with the `llms = [kbench.llm, kbench.judge_llm]`,
   `stop_condition=lambda runs: len(runs) == len(llms) * df.shape[0]`,
   per-model aggregation pattern. **Fix:** add the multi-model example
   directly into §3 (Dataset Evaluation).

3. **Manual judge pattern (`chats.new("judge")` + `judge_llm.prompt(...)`).**
   For scenarios that expect a *manual* judge — scoring inside an isolated
   chat with `schema=int` or `schema=bool` instead of
   `assess_response_with_judge` — the agent defaulted to the high-level
   helper every time (7.3, 7.8, 7.10 all picked the helper). The skill file
   should document **both** patterns and when to choose each: the helper for
   multi-criteria yes/no rubrics, the manual pattern for numeric scoring or
   when you need full control over the judge's prompt.

4. **`store_run=False` and decorator-name conventions.** Scenario 6.6 ("how
   to hide a sub-task from the leaderboard") missed `store_run=False`
   entirely, and several agents introduced ad-hoc `name=` strings that the
   judge then dinged. The Common Mistakes table covers `store_task=False`
   but `store_run=False` is undocumented. **Fix:** add a one-line row for
   `store_run=False` and clarify that `name=` is optional (the function name
   is used when omitted).

5. **Pydantic `Field(description=...)` and reasoning-mode discoverability.**
   Generalization scenarios that asked for nested Pydantic models (7.6,
   1.6, 1.7) frequently skipped `Field(description=...)` annotations — the
   skill file mentions `Field()` exists but doesn't show why it matters
   for extraction quality. The reasoning parameter (1.10, 7.4) and the
   `temperature=` parameter (6.9, 7.7) are similarly under-emphasised: each
   gets one line in the reference table but no worked example explaining
   *when* to reach for them. **Fix:** promote each of `Field(description=)`,
   `reasoning=`, and `temperature=` from one-liners to short worked
   examples in §4.


> ℹ️ A subset of failing criteria are **judge-strictness artifacts** (e.g., the criterion quotes `@kbench.task()` but the agent legitimately wrote `@kbench.task(name="...")`, or asks for `temperature=0` when the agent relied on the documented default of 0). These do not represent actual SKILL.md weaknesses; they are noted in the per-scenario detail where they occur.

## Filled-in Summary Table

Links jump to per-scenario detail.

### Summary Table

| # | Scenario | Category | Difficulty | Result |
|---|----------|----------|------------|--------|
| 1.1 | Simple Q&A regex | Basic Task | Basic | [✅ 7/7](#scenario-1-1) |
| 1.2 | Extract int | Basic Structured | Basic | [⚠️ 4/5](#scenario-1-2) |
| 1.3 | Extract bool | Basic Structured | Basic | [✅ 4/4](#scenario-1-3) |
| 1.4 | Extract dict | Basic Structured | Basic | [✅ 4/4](#scenario-1-4) |
| 1.5 | Extract dataclass | Basic Structured | Basic | [⚠️ 4/5](#scenario-1-5) |
| 1.6 | Extract pydantic | Basic Structured | Basic | [⚠️ 3/5](#scenario-1-6) |
| 1.7 | Composite pydantic | Basic Structured | Basic | [⚠️ 4/5](#scenario-1-7) |
| 1.8 | Multi-turn memory | Basic Conversation | Basic | [✅ 5/5](#scenario-1-8) |
| 1.9 | Simple greeting | Basic Task | Basic | [✅ 5/5](#scenario-1-9) |
| 1.10 | Reasoning param | Basic Task | Basic | [✅ 4/4](#scenario-1-10) |
| 1.11 | Image URL input | Basic Multimodal | Basic | [✅ 4/4](#scenario-1-11) |
| 1.12 | Audio input | Basic Multimodal | Basic | [⚠️ 5/6](#scenario-1-12) |
| 1.13 | Video input (URL) | Basic Multimodal | Basic | [✅ 5/5](#scenario-1-13) |
| 2.1 | Simple function tool | Basic Tool | Basic | [✅ 5/5](#scenario-2-1) |
| 2.2 | Multiple tools | Basic Tool | Basic | [✅ 4/4](#scenario-2-2) |
| 2.3 | Tool error handling | Basic Tool | Basic | [✅ 5/5](#scenario-2-3) |
| 3.1 | Extract + run code | Basic Code | Basic | [✅ 6/6](#scenario-3-1) |
| 3.2 | Code + subchat combo | Basic Code | Basic | [⚠️ 5/6](#scenario-3-2) |
| 4.1 | DataFrame eval (bool) | Dataset Eval | Medium | [✅ 7/7](#scenario-4-1) |
| 4.2 | Sub-task + accuracy | Dataset Eval | Medium | [⚠️ 8/9](#scenario-4-2) |
| 4.3 | Eval with full params | Dataset Eval | Medium | [⚠️ 4/7](#scenario-4-3) |
| 4.4 | Multi-model comparison | Dataset Eval | Medium | [❌ 3/7](#scenario-4-4) |
| 4.5 | Math word problems | Dataset Eval | Medium | [✅ 6/6](#scenario-4-5) |
| 5.1 | Structured + judge | Combined | Medium | [⚠️ 5/7](#scenario-5-1) |
| 5.2 | Hallucination detect | Combined | Medium | [⚠️ 4/5](#scenario-5-2) |
| 5.3 | System + struct + code | Combined | Medium | [✅ 7/7](#scenario-5-3) |
| 5.4 | Custom assertion + task | Combined | Medium | [✅ 7/7](#scenario-5-4) |
| 5.5 | Game loop + judge | Combined | Advanced | [⚠️ 7/9](#scenario-5-5) |
| 5.6 | Sub-task composition | Combined | Medium | [✅ 6/6](#scenario-5-6) |
| 5.7 | Code gen CSV filter | Combined | Medium | [✅ 6/6](#scenario-5-7) |
| 5.8 | Positive + negative | Combined | Medium | [⚠️ 3/4](#scenario-5-8) |
| 6.1 | Assertions vs assert | Knowledge | Basic | [✅ 5/5](#scenario-6-1) |
| 6.2 | Four schema styles | Knowledge | Basic | [✅ 6/6](#scenario-6-2) |
| 6.3 | Missing return anno | Troubleshooting | Basic | [✅ 4/4](#scenario-6-3) |
| 6.4 | Judge returns None | Troubleshooting | Basic | [⚠️ 3/4](#scenario-6-4) |
| 6.5 | new vs fork | Knowledge | Basic | [✅ 4/4](#scenario-6-5) |
| 6.6 | Sub-task leaderboard | Troubleshooting | Basic | [❌ 1/3](#scenario-6-6) |
| 6.7 | llm must be list | Troubleshooting | Basic | [✅ 3/3](#scenario-6-7) |
| 6.8 | -> None return type | Knowledge | Basic | [✅ 4/4](#scenario-6-8) |
| 6.9 | Temperature param | Knowledge | Basic | [⚠️ 3/4](#scenario-6-9) |
| 6.10 | Cell markers + magics | Knowledge | Basic | [✅ 6/6](#scenario-6-10) |
| 6.11 | No `__name__` guard | Knowledge | Basic | [✅ 6/6](#scenario-6-11) |
| 7.1 | Sentiment pipeline | Generalization | Advanced | [⚠️ 7/8](#scenario-7-1) |
| 7.2 | Code review benchmark | Generalization | Advanced | [⚠️ 6/7](#scenario-7-2) |
| 7.3 | Translation quality | Generalization | Advanced | [❌ 1/8](#scenario-7-3) |
| 7.4 | Reasoning + verify | Generalization | Advanced | [✅ 5/5](#scenario-7-4) |
| 7.5 | FAQ chatbot + fork | Generalization | Advanced | [⚠️ 4/6](#scenario-7-5) |
| 7.6 | Job posting extraction | Generalization | Advanced | [⚠️ 6/7](#scenario-7-6) |
| 7.7 | Self-consistency check | Generalization | Advanced | [⚠️ 3/6](#scenario-7-7) |
| 7.8 | Tool + judge retrieval | Generalization | Advanced | [⚠️ 4/7](#scenario-7-8) |
| 7.9 | Competitive coding | Generalization | Advanced | [⚠️ 7/8](#scenario-7-9) |
| 7.10 | Tricky questions eval | Generalization | Advanced | [⚠️ 4/7](#scenario-7-10) |
| 7.11 | Hallucination competition | Generalization | Advanced | [⚠️ 4/7](#scenario-7-11) |
| 7.12 | Run introspection | Generalization | Advanced | [✅ 7/7](#scenario-7-12) |

---

## Per-scenario detail

<a id="scenario-1-1"></a>
### ✅ Scenario 1.1 — Simple Q&A with Regex Check (7/7 criteria · Strong Pass)

**Category:** 1 — Basic — Simple Task + Assertion  
**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 149–180 (`assess_with_judge_task`); `documentation/examples/quick_start_examples.py` lines 29–38  

<details><summary>Prompt</summary>

```
Write a kaggle-benchmarks task that asks the LLM "What is Kaggle?" and asserts the response contains the word "platform" (case-insensitive). Run it with `kbench.llm`.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench


@kbench.task(name="what_is_kaggle")
def what_is_kaggle(llm):
    response = llm.prompt("What is Kaggle?")
    kbench.assertions.assert_contains_regex(
        r"(?i)platform",
        response,
        expectation="Response should contain the word 'platform'.",
    )


what_is_kaggle.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Uses `import kaggle_benchmarks as kbench`
  - _judge:_ The agent's response contains `import kaggle_benchmarks as kbench` exactly as required.
- ✅ Uses `@kbench.task()` decorator
  - _judge:_ The code uses `@kbench.task(name="what_is_kaggle")` which matches the `@kbench.task()` decorator pattern required by the criterion.
- ✅ First parameter of task function is `llm`
  - _judge:_ The task function `what_is_kaggle` has `llm` as its first parameter.
- ✅ Calls `llm.prompt("What is Kaggle?")`
  - _judge:_ The code explicitly calls `llm.prompt("What is Kaggle?")` on line 6 of the extracted Python block.
- ✅ Uses `kbench.assertions.assert_contains_regex(r"(?i)platform", response)` or `kbench.assertions.assert_in("platform", response.lower())`
  - _judge:_ The code uses `kbench.assertions.assert_contains_regex(r"(?i)platform", response, ...)` which matches the first allowed form exactly.
- ✅ Does NOT use plain Python `assert`
  - _judge:_ No plain Python `assert` statement found in the code. Only `kbench.assertions.assert_contains_regex` is used.
- ✅ Calls `.run(kbench.llm)` at the end
  - _judge:_ The code ends with `what_is_kaggle.run(kbench.llm)`, which satisfies `.run(kbench.llm)` being called at the end.

---

<a id="scenario-1-2"></a>
### ⚠️ Scenario 1.2 — Extract Integer with Structured Output (4/5 criteria · Partial Pass)

**Category:** 1 — Basic — Simple Task + Assertion  
**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 187–199 (`test_extract_int`)  

<details><summary>Prompt</summary>

```
Write a kaggle-benchmarks task that gives the LLM the text "The Apollo 11 mission landed on the Moon in 1969." and extracts the year as an integer using `schema=int`. Assert the year equals 1969.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench


@kbench.task(name="extract_apollo_year")
def extract_apollo_year(llm):
    year = llm.prompt(
        "Extract the year from this text as an integer:\n\n"
        "The Apollo 11 mission landed on the Moon in 1969.",
        schema=int,
    )
    kbench.assertions.assert_equal(1969, year, expectation="Extracted year should be 1969.")


extract_apollo_year.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ❌ Uses `@kbench.task()` decorator with `llm` as first param
  - _judge:_ The decorator used is `@kbench.task(name='extract_apollo_year')` with a `name` kwarg, not `@kbench.task()` with `llm` as the first param of the decorator call. Also, `llm` is a function param, not the decorator's first argument.
- ✅ Uses `llm.prompt("...", schema=int)`
  - _judge:_ The code contains `llm.prompt("...", schema=int)` with schema=int as required.
- ✅ Uses `kbench.assertions.assert_equal(1969, year)`
  - _judge:_ The code contains `kbench.assertions.assert_equal(1969, year,` which exactly matches the required criterion.
- ✅ Does NOT manually parse or cast the response to int
  - _judge:_ The code uses schema=int for structured output and never manually parses or casts the response with int(), int(year), etc.
- ✅ Calls `.run(kbench.llm)`
  - _judge:_ The code contains `extract_apollo_year.run(kbench.llm)` which satisfies the `.run(kbench.llm)` criterion.

---

<a id="scenario-1-3"></a>
### ✅ Scenario 1.3 — Extract Bool with Structured Output (4/4 criteria · Strong Pass)

**Category:** 1 — Basic — Simple Task + Assertion  
**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 206–218 (`test_extract_bool`)  

<details><summary>Prompt</summary>

```
Write a benchmark task that asks the LLM whether "I absolutely loved this movie! It was fantastic." is a positive review. Use `schema=bool` to get a boolean answer and assert it's True.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench


@kbench.task(name="sentiment_positive_review")
def sentiment_positive_review(llm):
    review = "I absolutely loved this movie! It was fantastic."
    is_positive = llm.prompt(
        f'Is the following review positive? Answer with true or false.\n\nReview: "{review}"',
        schema=bool,
    )
    kbench.assertions.assert_true(
        is_positive,
        expectation="The review expresses clear enthusiasm and should be classified as positive.",
    )


sentiment_positive_review.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Uses `schema=bool` in `llm.prompt()`
  - _judge:_ The code contains `schema=bool` in `llm.prompt()` call, matching the criterion exactly.
- ✅ Uses `kbench.assertions.assert_true(is_positive)`
  - _judge:_ The code contains `kbench.assertions.assert_true(is_positive, ...)` which matches the required criterion exactly.
- ✅ Does NOT parse the response string manually
  - _judge:_ The code uses schema=bool for structured output via llm.prompt() and contains no manual string parsing (no string splitting, regex, or conditional checks on response text).
- ✅ Has `@kbench.task()` decorator
  - _judge:_ The code contains `@kbench.task(name="sentiment_positive_review")` which matches the `@kbench.task()` decorator pattern required by the criterion.

---

<a id="scenario-1-4"></a>
### ✅ Scenario 1.4 — Extract Dict with Schema (4/4 criteria · Strong Pass)

**Category:** 1 — Basic — Simple Task + Assertion  
**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 225–246 (`test_extract_dict`)  

<details><summary>Prompt</summary>

```
Write a task that extracts person details (name, age, occupation) from the text "Contact info: John Doe, age 42, works as a Software Engineer." Use an inline dict schema `{"name": str, "age": int, "occupation": str}`. Assert each field.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench

@kbench.task(name="extract_person_details")
def extract_person_details(llm):
    bio = "Contact info: John Doe, age 42, works as a Software Engineer."
    person = llm.prompt(
        f"Extract the name, age, and occupation from this text:\n\n{bio}",
        schema={"name": str, "age": int, "occupation": str},
    )
    kbench.assertions.assert_equal("John Doe", person.name)
    kbench.assertions.assert_equal(42, person.age)
    kbench.assertions.assert_contains_regex(
        r"(?i)software engineer", person.occupation,
        expectation="Occupation should be 'Software Engineer'.",
    )

extract_person_details.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Uses `schema={"name": str, "age": int, "occupation": str}`
  - _judge:_ The code contains `schema={"name": str, "age": int, "occupation": str}` exactly as required.
- ✅ Accesses fields via `person.name`, `person.age`, `person.occupation` (dot notation, not dict indexing)
  - _judge:_ Code uses person.name, person.age, person.occupation — all dot notation, no dict indexing.
- ✅ Uses multiple kbench assertions (e.g., `assert_equal`, `assert_contains_regex`)
  - _judge:_ Code uses both assert_equal (twice) and assert_contains_regex, satisfying the 'multiple kbench assertions' criterion.
- ✅ Does NOT define a dataclass or pydantic model (uses inline dict)
  - _judge:_ No dataclass or pydantic model defined; uses inline dict schema={'name': str, 'age': int, 'occupation': str}.

---

<a id="scenario-1-5"></a>
### ⚠️ Scenario 1.5 — Extract Dataclass with Structured Output (4/5 criteria · Partial Pass)

**Category:** 1 — Basic — Simple Task + Assertion  
**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 253–281 (`test_extract_dataclass`)  

<details><summary>Prompt</summary>

```
Write a task that asks the LLM to generate a level 5 wizard character for a fantasy game. Define a `@dataclass` with fields: name (str), class_type (str), level (int), inventory (str). Assert the class_type contains "wizard" and level equals 5.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench
from dataclasses import dataclass


@dataclass
class Character:
    name: str
    class_type: str
    level: int
    inventory: str


@kbench.task(name="generate_wizard_character")
def generate_wizard_character(llm):
    character = llm.prompt(
        "Generate a level 5 wizard character for a fantasy game. "
        "Give the character a creative name, set class_type to 'wizard', "
        "set level to 5, and list a few starting inventory items.",
        schema=Character,
    )

    kbench.assertions.assert_contains_regex(
        r"(?i)wizard",
        character.class_type,
        expectation="class_type should contain 'wizard'.",
    )
    kbench.assertions.assert_equal(
        5,
        character.level,
        expectation="Character level should be exactly 5.",
    )


generate_wizard_character.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Defines a `@dataclass` with `name`, `class_type`, `level`, `inventory`
  - _judge:_ The @dataclass class Character defines all four required fields: name, class_type, level, and inventory.
- ❌ Uses `llm.prompt("...", schema=RPGCharacter)`
  - _judge:_ The agent uses `schema=Character` with a locally defined `Character` dataclass, not `schema=RPGCharacter` as required by the criterion.
- ✅ Uses `kbench.assertions.assert_contains_regex(r"(?i)wizard", character.class_type)`
  - _judge:_ The code contains exactly `kbench.assertions.assert_contains_regex(r"(?i)wizard", character.class_type,` as required by the criterion.
- ✅ Uses `kbench.assertions.assert_equal(5, character.level)`
  - _judge:_ The code contains `kbench.assertions.assert_equal(5, character.level,` which matches the required criterion exactly.
- ✅ Has `@kbench.task()` decorator
  - _judge:_ The code contains `@kbench.task(name="generate_wizard_character")` which satisfies the `@kbench.task()` decorator requirement.

---

<a id="scenario-1-6"></a>
### ⚠️ Scenario 1.6 — Extract Pydantic Model (3/5 criteria · Partial Pass)

**Category:** 1 — Basic — Simple Task + Assertion  
**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 288–313 (`test_extract_pydantic`)  

<details><summary>Prompt</summary>

```
Write a task that asks the LLM about planet Jupiter. Define a Pydantic `BaseModel` with fields: name (str), mass_earth_masses (float), has_life (bool), moons (list[str]). Assert mass > 300 and moons list is non-empty.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench
import pydantic


class JupiterFacts(pydantic.BaseModel):
    name: str = pydantic.Field(description="The planet's name")
    mass_earth_masses: float = pydantic.Field(
        description="The planet's mass expressed in Earth masses (e.g. 317.8)"
    )
    has_life: bool = pydantic.Field(
        description="Whether the planet is known to harbor life"
    )
    moons: list[str] = pydantic.Field(
        description="A list of named moons orbiting the planet (include at least the four Galilean moons)"
    )


@kbench.task(name="jupiter_facts")
def jupiter_facts(llm):
    """Ask the LLM about Jupiter and validate the structured response."""
    facts = llm.prompt(
        "Provide factual information about the planet Jupiter.",
        schema=JupiterFacts,
    )

    kbench.assertions.assert_equal(
        "Jupiter",
        facts.name,
        expectation="The planet name should be Jupiter.",
    )

    kbench.assertions.assert_true(
        facts.mass_earth_masses > 300,
        expectation=(
            f"Jupiter's mass should exceed 300 Earth masses "
            f"(got {facts.mass_earth_masses})."
        ),
    )

    kbench.assertions.assert_false(
        facts.has_life,
        expectation="Jupiter is not known to harbor life.",
    )

    kbench.assertions.assert_not_empty(
        facts.moons,
        expectation="Jupiter has many moons — the list must not be empty.",
    )


jupiter_facts.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Defines a Pydantic `BaseModel` (imports from `pydantic`)
  - _judge:_ The code imports pydantic and defines JupiterFacts as a subclass of pydantic.BaseModel, satisfying the criterion.
- ✅ Uses `Field()` with descriptions (optional but preferred)
  - _judge:_ The code uses `pydantic.Field(description=...)` with descriptions for all four model fields, satisfying the criterion.
- ❌ Uses `llm.prompt("...", schema=Planet)`
  - _judge:_ The agent uses `schema=JupiterFacts` not `schema=Planet`. The criterion requires the literal token `Planet` as the schema argument.
- ❌ Uses `kbench.assertions.assert_true(planet.mass_earth_masses > 300)`
  - _judge:_ Code uses `facts.mass_earth_masses > 300` not `planet.mass_earth_masses > 300` as required by the criterion.
- ✅ Uses `kbench.assertions.assert_true(len(planet.moons) > 0)` or `assert_not_empty`
  - _judge:_ The code uses `kbench.assertions.assert_not_empty(facts.moons, ...)` which satisfies the criterion requiring `assert_not_empty`.

---

<a id="scenario-1-7"></a>
### ⚠️ Scenario 1.7 — Composite Pydantic with Nested List (4/5 criteria · Partial Pass)

**Category:** 1 — Basic — Simple Task + Assertion  
**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 320–344 (`test_extract_composite_pydantic`)  

<details><summary>Prompt</summary>

```
Write a task that asks the LLM to list the 6 main characters of Friends. Define TWO Pydantic models: `Actor` with fields `actor_name` and `role_name`, and `Casting` with a field `actors: list[Actor]`. Assert there are exactly 6 actors and that "Jennifer" is among the actor names.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench
import pydantic


class Actor(pydantic.BaseModel):
    actor_name: str = pydantic.Field(description="The real name of the actor")
    role_name: str = pydantic.Field(description="The name of the character they play")


class Casting(pydantic.BaseModel):
    actors: list[Actor] = pydantic.Field(description="The 6 main cast members of Friends")


@kbench.task(name="friends_casting")
def friends_casting(llm):
    casting = llm.prompt(
        "List the 6 main characters of the TV show Friends, including the actor's real name and their character's name.",
        schema=Casting,
    )

    kbench.assertions.assert_equal(
        6,
        len(casting.actors),
        expectation="There should be exactly 6 main cast members.",
    )

    actor_names = [actor.actor_name for actor in casting.actors]
    kbench.assertions.assert_true(
        any("Jennifer" in name for name in actor_names),
        expectation="Jennifer Aniston should be among the actors.",
    )


friends_casting.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Defines two Pydantic models (nested structure)
  - _judge:_ Two Pydantic models defined: Actor (with actor_name, role_name) and Casting (with actors: list[Actor]), forming a nested structure.
- ✅ Uses `schema=Casting` in the prompt
  - _judge:_ The code contains `schema=Casting` literally in the llm.prompt() call.
- ✅ Uses `kbench.assertions.assert_equal(len(casting.actors), 6)` or similar
  - _judge:_ The agent uses `kbench.assertions.assert_equal(6, len(casting.actors), ...)` which matches the required pattern of assert_equal with len(casting.actors) and 6.
- ❌ Joins actor names and uses `kbench.assertions.assert_in("Jennifer", ...)`
  - _judge:_ The agent uses `kbench.assertions.assert_true(any(...))` instead of the required `kbench.assertions.assert_in("Jennifer", ...)` call.
- ✅ Does NOT return a flat list — uses nested Pydantic models
  - _judge:_ Response uses nested Pydantic models: Casting contains list[Actor], not a flat list. Both Actor and Casting are proper BaseModel subclasses.

---

<a id="scenario-1-8"></a>
### ✅ Scenario 1.8 — Multi-Turn Memory (5/5 criteria · Strong Pass)

**Category:** 1 — Basic — Simple Task + Assertion  
**Source of Truth:** `golden_tests/test_code_generator.py` lines 122–125 (`memory-alice`); `documentation/examples/simple_task.py` lines 33–36 (`subtask2`)  

<details><summary>Prompt</summary>

```
Write a benchmark task that tests multi-turn conversation memory. First tell the LLM "My name is Alice", then ask "What is my name?". Assert the response contains "Alice".
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench


@kbench.task(name="multi_turn_memory")
def multi_turn_memory(llm):
    """Tests whether the LLM retains information across conversation turns."""
    llm.prompt("My name is Alice.")
    response = llm.prompt("What is my name?")
    kbench.assertions.assert_contains_regex(
        r"(?i)alice",
        response,
        expectation="The LLM should remember the name 'Alice' from the previous turn.",
    )


multi_turn_memory.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Calls `llm.prompt()` twice in sequence (multi-turn)
  - _judge:_ llm.prompt() is called twice in sequence: first with 'My name is Alice.' and then with 'What is my name?'
- ✅ First call: `llm.prompt("My name is Alice.")`
  - _judge:_ The code contains exactly `llm.prompt("My name is Alice.")` as the first call.
- ✅ Second call: `response = llm.prompt("What is my name?")`
  - _judge:_ The code contains exactly `response = llm.prompt("What is my name?")` as the second call.
- ✅ Uses `kbench.assertions.assert_contains_regex(r"(?i)alice", response)` or `assert_in`
  - _judge:_ The code uses `kbench.assertions.assert_contains_regex(r"(?i)alice", response, ...)` which matches the required pattern exactly.
- ✅ Does NOT use `chats.new()` (relies on automatic history)
  - _judge:_ The code does not contain 'chats.new()' anywhere.

---

<a id="scenario-1-9"></a>
### ✅ Scenario 1.9 — Simple Greeting Assertion (5/5 criteria · Strong Pass)

**Category:** 1 — Basic — Simple Task + Assertion  
**Source of Truth:** `documentation/examples/simple_task.py` lines 27–30 (`subtask1`); `documentation/examples/simple_multiple_tasks.py` lines 37–40 (`task1`)  

<details><summary>Prompt</summary>

```
Write the simplest possible benchmark task: send "Hello!" to the LLM and assert the response is not empty.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench

@kbench.task()
def hello(llm):
    response = llm.prompt("Hello!")
    kbench.assertions.assert_not_empty(response)

hello.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Uses `@kbench.task()` decorator
  - _judge:_ The code contains `@kbench.task()` decorator explicitly on line 3 of the extracted code block.
- ✅ Calls `llm.prompt("Hello!")`
  - _judge:_ The code contains `llm.prompt("Hello!")` exactly as required.
- ✅ Uses `kbench.assertions.assert_not_empty(response)`
  - _judge:_ The code contains `kbench.assertions.assert_not_empty(response)` which satisfies the criterion.
- ✅ Calls `.run(kbench.llm)`
  - _judge:_ The code contains `hello.run(kbench.llm)` which satisfies the criterion of calling `.run(kbench.llm)`.
- ✅ No return type annotation (pass/fail task)
  - _judge:_ The function `hello(llm)` has no return type annotation, satisfying the criterion.

---

<a id="scenario-1-10"></a>
### ✅ Scenario 1.10 — Reasoning Parameter (4/4 criteria · Strong Pass)

**Category:** 1 — Basic — Simple Task + Assertion  
**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 601–619 (`test_reasoning_param`)  

<details><summary>Prompt</summary>

```
Write a task that asks "What is 2 + 2?" using reasoning mode set to "low". Assert the response contains "4".
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench


@kbench.task(name="simple_addition_reasoning")
def simple_addition_reasoning(llm):
    response = llm.prompt("What is 2 + 2?", reasoning="low")
    kbench.assertions.assert_contains_regex(
        r"4", response,
        expectation="Response should contain '4'."
    )


simple_addition_reasoning.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Uses `llm.prompt("...", reasoning="low")`
  - _judge:_ The code contains `llm.prompt("What is 2 + 2?", reasoning="low")` which matches the required pattern `llm.prompt("...", reasoning="low")`.
- ✅ Uses `kbench.assertions.assert_contains_regex(r"4", response)`
  - _judge:_ The code contains `kbench.assertions.assert_contains_regex(r"4", response,` which matches the required call with r"4" as first arg and response as second.
- ✅ Has `@kbench.task()` decorator
  - _judge:_ The code contains `@kbench.task(name="simple_addition_reasoning")` which matches the `@kbench.task()` decorator criterion.
- ✅ Does NOT set reasoning via a separate configuration
  - _judge:_ Reasoning is set inline as a parameter to llm.prompt(), not via a separate configuration object or method.

---

<a id="scenario-1-11"></a>
### ✅ Scenario 1.11 — Image Input (URL) (4/4 criteria · Strong Pass)

**Category:** 1 — Basic — Simple Task + Assertion  
**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 404–427 (`test_image_url`)  

<details><summary>Prompt</summary>

```
Write a task that sends the Kaggle logo image (URL: `https://www.kaggle.com/static/images/site-logo.png`) to the LLM and asks "What does this logo say?". Assert the response contains "kaggle" (case-insensitive).
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench
from kaggle_benchmarks.content_types import images

@kbench.task(name="kaggle_logo_recognition")
def kaggle_logo_recognition(llm):
    img = images.from_url("https://www.kaggle.com/static/images/site-logo.png")
    response = llm.prompt("What does this logo say?", image=img)
    kbench.assertions.assert_contains_regex(
        r"(?i)kaggle",
        response,
        expectation="Response should mention 'Kaggle'.",
    )

kaggle_logo_recognition.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Imports `images` from `kaggle_benchmarks.content_types`
  - _judge:_ The code contains `from kaggle_benchmarks.content_types import images` exactly as required.
- ✅ Uses `images.from_url("...")` to create image object
  - _judge:_ The agent's response contains `images.from_url("https://www.kaggle.com/static/images/site-logo.png")` which matches the required pattern.
- ✅ Uses `llm.prompt("...", image=image)` (preferred approach)
  - _judge:_ The code contains `llm.prompt("What does this logo say?", image=img)` which matches the required pattern `llm.prompt("...", image=image)`.
- ✅ Uses `kbench.assertions.assert_contains_regex(r"(?i)kaggle", response)`
  - _judge:_ The code contains exactly `kbench.assertions.assert_contains_regex(r"(?i)kaggle", response)` as the first two positional arguments, matching the criterion.

---

<a id="scenario-1-12"></a>
### ⚠️ Scenario 1.12 — Audio Input (5/6 criteria · Partial Pass)

**Category:** 1 — Basic — Simple Task + Assertion  
**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 490–516 (`test_audio_local_file`, `test_audio_url`, `test_audio_base64`); Skill file §4 Audio  

<details><summary>Prompt</summary>

```
Write a benchmark task that sends an MP3 audio file at path "speech.mp3" to the LLM and asks it to transcribe the audio. Assert the response is not empty.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench
from kaggle_benchmarks.content_types import audios

@kbench.task(name="audio_transcription")
def audio_transcription(llm):
    audio = audios.from_path("speech.mp3")
    response = llm.prompt("Please transcribe the audio.", audio=audio)
    kbench.assertions.assert_not_empty(response, expectation="Transcription should not be empty.")

audio_transcription.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Imports `audios` from `kaggle_benchmarks.content_types`
  - _judge:_ The code explicitly imports `audios` from `kaggle_benchmarks.content_types` on line 2.
- ✅ Uses `audios.from_path("speech.mp3")` to create audio object
  - _judge:_ The code contains exactly `audios.from_path("speech.mp3")` as required by the criterion.
- ❌ Uses `llm.prompt("Transcribe this audio.", audio=audio)`
  - _judge:_ The agent uses 'Please transcribe the audio.' as the prompt string, but the criterion requires exactly 'Transcribe this audio.' — the literal string does not match.
- ✅ Uses `kbench.assertions.assert_not_empty(response)`
  - _judge:_ The code explicitly calls `kbench.assertions.assert_not_empty(response, ...)` which matches the required criterion.
- ✅ Has `@kbench.task()` decorator
  - _judge:_ The code contains `@kbench.task(name="audio_transcription")` which matches the `@kbench.task()` decorator pattern required by the criterion.
- ✅ Does NOT try to use `user.send()` for the audio (uses `llm.prompt(audio=)` instead)
  - _judge:_ The code uses `llm.prompt(..., audio=audio)` and never calls `user.send()` for audio.

---

<a id="scenario-1-13"></a>
### ✅ Scenario 1.13 — Video Input (URL) (5/5 criteria · Strong Pass)

**Category:** 1 — Basic — Simple Task + Assertion  
**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 430–465 (`test_video_url`); Skill file §4 Videos  

<details><summary>Prompt</summary>

```
Write a benchmark task that sends a YouTube video URL to the LLM and asks "What is happening in this video?". Assert the response is not empty.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench

@kbench.task(name="video_understanding")
def video_understanding(llm):
    from kaggle_benchmarks.content_types import videos

    video = videos.from_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    response = llm.prompt("What is happening in this video?", video=video)
    kbench.assertions.assert_not_empty(response, expectation="Response should describe the video content.")

video_understanding.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Imports `videos` from `kaggle_benchmarks.content_types`
  - _judge:_ The code contains 'from kaggle_benchmarks.content_types import videos' which satisfies the criterion.
- ✅ Uses `videos.from_url("https://www.youtube.com/watch?v=...")` to create video object
  - _judge:_ The code contains `videos.from_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")` which matches the required pattern with a valid YouTube URL.
- ✅ Uses `llm.prompt("...", video=video)`
  - _judge:_ The code contains `llm.prompt("What is happening in this video?", video=video)` which matches the required pattern `llm.prompt("...", video=video)`.
- ✅ Uses `kbench.assertions.assert_not_empty(response)`
  - _judge:_ The code explicitly calls `kbench.assertions.assert_not_empty(response, ...)` which matches the required criterion.
- ✅ Has `@kbench.task()` decorator
  - _judge:_ The code contains `@kbench.task(name="video_understanding")` which matches the `@kbench.task()` decorator pattern (with optional arguments).

---

<a id="scenario-2-1"></a>
### ✅ Scenario 2.1 — Simple Function Tool (5/5 criteria · Strong Pass)

**Category:** 2 — Basic — Function Tool Use  
**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 660–685 (`test_simple_tool_use`); Skill file §7 "Custom Function Tools"  

<details><summary>Prompt</summary>

```
Write a benchmark task that provides a calculator tool to the LLM. The tool `run_simple_calculator(a: float, b: float, operator: str) -> float` supports +, -, *, /. Ask the LLM "What is 50 plus 25?" and pass the calculator as a tool. Assert the final answer contains "75".
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench


def run_simple_calculator(a: float, b: float, operator: str) -> float:
    """Calculates the result of an arithmetic operation. Supported operators: + - * /"""
    if operator == "+":
        return a + b
    if operator == "-":
        return a - b
    if operator == "*":
        return a * b
    if operator == "/":
        return a / b
    raise ValueError(f"Unknown operator: {operator}")


@kbench.task(name="calculator_tool")
def calculator_tool(llm):
    response = llm.prompt("What is 50 plus 25?", tools=[run_simple_calculator])
    kbench.assertions.assert_contains_regex(
        r"75",
        response,
        expectation="The final answer should contain 75.",
    )


calculator_tool.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Defines the `run_simple_calculator` function with proper type hints and docstring
  - _judge:_ The function `run_simple_calculator` is defined with type hints (a: float, b: float, operator: str) -> float and a docstring.
- ✅ Uses `llm.prompt("...", tools=[run_simple_calculator])`
  - _judge:_ The code contains `llm.prompt("What is 50 plus 25?", tools=[run_simple_calculator])` which satisfies the criterion of using `llm.prompt("...", tools=[run_simple_calculator])`.
- ✅ Uses `kbench.assertions` to check the response contains "75"
  - _judge:_ Code uses `kbench.assertions.assert_contains_regex` with `r"75"` to check response contains "75", satisfying the criterion.
- ✅ Does NOT manually parse tool call JSON (uses automatic tool calling)
  - _judge:_ The code uses `tools=[run_simple_calculator]` for automatic tool calling and contains no manual JSON parsing (no json.loads, json.dumps, or similar patterns).
- ✅ Has `@kbench.task()` decorator
  - _judge:_ @kbench.task() decorator is present (with optional name argument, which still satisfies the requirement of having the decorator).

---

<a id="scenario-2-2"></a>
### ✅ Scenario 2.2 — Multiple Tools Selection (4/4 criteria · Strong Pass)

**Category:** 2 — Basic — Function Tool Use  
**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 708–736 (`test_multiple_tool_selection`)  

<details><summary>Prompt</summary>

```
Write a benchmark task where the LLM is given TWO tools: `add_tool(a, b)` and `multiply_tool(a, b)`. Ask it "What is 12 multiplied by 34? Use the multiply_tool." Verify the correct tool was called.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench


def add_tool(a: float, b: float) -> float:
    """Adds two numbers together."""
    return a + b


def multiply_tool(a: float, b: float) -> float:
    """Multiplies two numbers together."""
    return a * b


@kbench.task(name="multiply_tool_selection")
def multiply_tool_selection(llm):
    response = llm.prompt(
        "What is 12 multiplied by 34? Use the multiply_tool.",
        tools=[add_tool, multiply_tool],
    )

    kbench.assertions.assert_tool_was_invoked(
        multiply_tool,
        expectation="The multiply_tool should have been called.",
    )
    kbench.assertions.assert_contains_regex(
        r"408",
        response,
        expectation="Response should contain the correct answer 408.",
    )


multiply_tool_selection.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Defines two separate tool functions with docstrings
  - _judge:_ Two separate tool functions (add_tool, multiply_tool) are defined, each with a docstring.
- ✅ Uses `llm.prompt("...", tools=[add_tool, multiply_tool])`
  - _judge:_ The code contains `llm.prompt("...", tools=[add_tool, multiply_tool])` with both tools listed exactly as required.
- ✅ Uses kbench assertion to verify the answer contains "408" (12*34)
  - _judge:_ The code uses kbench.assertions.assert_contains_regex with r"408" to verify the answer contains "408" (12*34).
- ✅ Has `@kbench.task()` decorator
  - _judge:_ The code contains `@kbench.task(name="multiply_tool_selection")` which matches the `@kbench.task()` decorator pattern required by the criterion.

---

<a id="scenario-2-3"></a>
### ✅ Scenario 2.3 — Tool Error Handling (5/5 criteria · Strong Pass)

**Category:** 2 — Basic — Function Tool Use  
**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 741–762 (`test_tool_error_handling`); Skill file §7  

<details><summary>Prompt</summary>

```
Write a benchmark task where the tool function raises a ValueError. The LLM should be asked to call the tool and then report the error. Verify the LLM mentions "error" or "failed" in its response.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench


def flaky_tool() -> str:
    """Calls an external service that is currently unavailable."""
    raise ValueError("Tool execution failed.")


@kbench.task(name="tool_error_reporting")
def tool_error_reporting(llm):
    response = llm.prompt(
        "Please call the flaky_tool and report what happened.",
        tools=[flaky_tool],
    )
    kbench.assertions.assert_contains_regex(
        r"(?i)error|failed",
        response,
        expectation="LLM should mention 'error' or 'failed' after the tool raises.",
    )


tool_error_reporting.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Defines a tool function that raises `ValueError`
  - _judge:_ The function `flaky_tool` explicitly raises `ValueError('Tool execution failed.')`.
- ✅ Uses `llm.prompt("...", tools=[flaky_tool])`
  - _judge:_ The code contains `llm.prompt("Please call the flaky_tool and report what happened.", tools=[flaky_tool],)` which matches the criterion `llm.prompt("...", tools=[flaky_tool])`
- ✅ Uses `kbench.assertions.assert_contains_regex(r"(?i)error|failed", response)`
  - _judge:_ The code contains exactly `kbench.assertions.assert_contains_regex(r"(?i)error|failed", response, ...)` as required by the criterion.
- ✅ Has `@kbench.task()` decorator
  - _judge:_ The code contains `@kbench.task(name="tool_error_reporting")` which satisfies the `@kbench.task()` decorator requirement.
- ✅ Does NOT try-catch around the `llm.prompt()` call (the library handles tool errors)
  - _judge:_ No try-catch block surrounds the llm.prompt() call. The code calls llm.prompt() directly without any exception handling.

---

<a id="scenario-3-1"></a>
### ✅ Scenario 3.1 — Extract and Run Code (6/6 criteria · Strong Pass)

**Category:** 3 — Basic — Code Execution  
**Source of Truth:** `golden_tests/test_code_generator.py` lines 112–116 (`factorial-10`); `documentation/examples/simple_multiple_tasks.py` lines 58–70 (`task4`)  

<details><summary>Prompt</summary>

```
Write a benchmark task that asks the LLM to write Python code to compute the factorial of 10 and print it. Extract the code, run it, and assert the output is "3628800" with no errors.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench


@kbench.task(name="factorial_code_gen")
def factorial_code_gen(llm):
    """Ask the LLM to write Python that computes and prints the factorial of 10."""
    response = llm.prompt(
        "Write a Python program that computes the factorial of 10 and prints the result. "
        "Output only the code, no explanation."
    )

    code = kbench.tools.python.extract_code(response)
    result = kbench.tools.python.script_runner.run_code(code)

    kbench.assertions.assert_empty(
        result.stderr.strip(),
        expectation="Code should run without any errors.",
    )
    kbench.assertions.assert_equal(
        "3628800",
        result.stdout.strip(),
        expectation="Output should be exactly '3628800'.",
    )


factorial_code_gen.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Calls `llm.prompt()` with a coding prompt
  - _judge:_ The agent calls `llm.prompt()` with a coding prompt asking to write a Python program to compute the factorial of 10.
- ✅ Uses `kbench.tools.python.extract_code(response)` to extract
  - _judge:_ The response contains `kbench.tools.python.extract_code(response)` which satisfies the criterion.
- ✅ Uses `kbench.tools.python.script_runner.run_code(code)` to execute
  - _judge:_ The agent's code contains exactly `kbench.tools.python.script_runner.run_code(code)` which matches the required call pattern.
- ✅ Asserts `result.stderr` is empty: `kbench.assertions.assert_empty(result.stderr.strip())`
  - _judge:_ The code contains `kbench.assertions.assert_empty(result.stderr.strip(), ...)` which matches the required pattern.
- ✅ Asserts `result.stdout` contains "3628800": `kbench.assertions.assert_equal("3628800", result.stdout.strip())`
  - _judge:_ The code contains `kbench.assertions.assert_equal("3628800", result.stdout.strip(), ...)` which matches the required assertion pattern.
- ✅ Uses kbench assertions, NOT plain assert
  - _judge:_ Code uses kbench.assertions.assert_empty and kbench.assertions.assert_equal, not plain Python assert statements.

---

<a id="scenario-3-2"></a>
### ⚠️ Scenario 3.2 — Code Extraction + Subchat in One Task (5/6 criteria · Partial Pass)

**Category:** 3 — Basic — Code Execution  
**Source of Truth:** `documentation/examples/simple_multiple_tasks.py` lines 58–71 (`task4`)  

<details><summary>Prompt</summary>

```
Write a benchmark task that does three things in sequence:
1. Sends "Hello!" and asserts the response is not empty.
2. Opens a subchat, sends "Hello in subchat!" and asserts that response is not empty.
3. Asks the LLM to write Python code to print 'hello world!', extracts the code, runs it, and asserts the output matches.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench

@kbench.task(name="hello_world_benchmark")
def hello_world_benchmark(llm):
    # Step 1: Send "Hello!" and assert the response is not empty
    response = llm.prompt("Hello!")
    kbench.assertions.assert_not_empty(response, expectation="Response to 'Hello!' should not be empty.")

    # Step 2: Open a subchat, send "Hello in subchat!" and assert the response is not empty
    with kbench.chats.new("subchat"):
        subchat_response = llm.prompt("Hello in subchat!")
        kbench.assertions.assert_not_empty(
            subchat_response,
            expectation="Response in subchat should not be empty.",
        )

    # Step 3: Ask the LLM to write Python code to print 'hello world!', run it, assert output
    code_response = llm.prompt("Write Python code to print 'hello world!'.")
    code = kbench.tools.python.extract_code(code_response)
    result = kbench.tools.python.script_runner.run_code(code)
    kbench.assertions.assert_empty(result.stderr.strip(), expectation="Code should run without errors.")
    kbench.assertions.assert_equal(
        "hello world!",
        result.stdout.strip(),
        expectation="Code output should be 'hello world!'.",
    )

hello_world_benchmark.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Has three sequential steps within one task
  - _judge:_ The task contains three sequential steps: (1) prompt 'Hello!' and assert not empty, (2) open subchat and assert response not empty, (3) request Python code, run it, and assert output.
- ❌ Uses `with kbench.chats.new("subchat1"):` for step 2
  - _judge:_ Agent uses `kbench.chats.new("subchat")` instead of the required `kbench.chats.new("subchat1")`.
- ✅ Uses `kbench.tools.python.extract_code()` and `script_runner.run_code()` for step 3
  - _judge:_ Both `kbench.tools.python.extract_code()` and `kbench.tools.python.script_runner.run_code()` appear literally in step 3 of the agent's code.
- ✅ Uses `kbench.assertions.assert_not_empty()` for steps 1 and 2
  - _judge:_ kbench.assertions.assert_not_empty() is called for both step 1 (response to 'Hello!') and step 2 (subchat_response) in the extracted code.
- ✅ Uses `kbench.assertions.assert_equal("hello world!", result.stdout.strip())` for step 3
  - _judge:_ The agent uses `kbench.assertions.assert_equal("hello world!", result.stdout.strip(), ...)` in step 3, matching the required pattern.
- ✅ All assertions are kbench assertions
  - _judge:_ All assertions use kbench.assertions.assert_not_empty, assert_empty, and assert_equal — all are kbench assertions.

---

<a id="scenario-4-1"></a>
### ✅ Scenario 4.1 — Basic DataFrame Evaluation with Bool Return (7/7 criteria · Strong Pass)

**Category:** 4 — Dataset Evaluation  
**Source of Truth:** `documentation/examples/quick_start_examples.py` lines 244–278 (`solve_and_check_riddle`)  

<details><summary>Prompt</summary>

```
Write a benchmark task that evaluates the LLM on a set of riddles. The task should:
1. Accept `llm`, `riddle`, and `answer_keyword` parameters.
2. Prompt the LLM with the riddle.
3. Return `True` if the answer_keyword is found in the response (case-insensitive), `False` otherwise.
4. Create a DataFrame with 3 riddles and their answer keywords.
5. Run `.evaluate()` on the DataFrame with `n_jobs=3`.
6. Print the results table.

Use these riddles:
- "I have cities but no houses. What am I?" → "map"
- "What has an eye but cannot see?" → "needle"
- "What has to be broken before you can use it?" → "egg"
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
# %%
import pandas as pd
import kaggle_benchmarks as kbench

# %%
@kbench.task(name="riddle_solver", store_task=False)
def riddle_solver(llm, riddle: str, answer_keyword: str) -> bool:
    response = llm.prompt(riddle)
    is_correct = answer_keyword.lower() in response.lower()
    kbench.assertions.assert_true(
        is_correct,
        expectation=f"Response should contain '{answer_keyword}'.",
    )
    return is_correct

# %%
df = pd.DataFrame([
    {"riddle": "I have cities but no houses. What am I?",         "answer_keyword": "map"},
    {"riddle": "What has an eye but cannot see?",                  "answer_keyword": "needle"},
    {"riddle": "What has to be broken before you can use it?",     "answer_keyword": "egg"},
])

# %%
results = riddle_solver.evaluate(
    llm=[kbench.llm],
    evaluation_data=df,
    n_jobs=3,
)

# %%
print(results.as_dataframe())
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Task has `-> bool` return type annotation
  - _judge:_ The function definition `def riddle_solver(llm, riddle: str, answer_keyword: str) -> bool:` contains the `-> bool` return type annotation.
- ✅ Task parameters are `llm, riddle, answer_keyword` (matching DataFrame columns)
  - _judge:_ Function signature uses `llm, riddle: str, answer_keyword: str` matching the required DataFrame columns `llm`, `riddle`, `answer_keyword`.
- ✅ Returns `answer_keyword.lower() in response.lower()`
  - _judge:_ The code contains `answer_keyword.lower() in response.lower()` exactly as required by the criterion.
- ✅ Creates `pd.DataFrame` with the 3 riddles
  - _judge:_ pd.DataFrame is created with all 3 riddles (map/needle/egg) and their answer_keywords, matching the required structure.
- ✅ Calls `.evaluate(llm=[kbench.llm], evaluation_data=df, n_jobs=3)`
  - _judge:_ The agent calls `.evaluate(llm=[kbench.llm], evaluation_data=df, n_jobs=3)` which matches the criterion exactly.
- ✅ `llm` parameter is passed as a LIST `[kbench.llm]`
  - _judge:_ The agent passes `llm=[kbench.llm]` as a list containing `kbench.llm`, matching the criterion exactly.
- ✅ Calls `.as_dataframe()` on the result
  - _judge:_ The agent calls `.as_dataframe()` on the results object: `results.as_dataframe()` is present in the code.

---

<a id="scenario-4-2"></a>
### ⚠️ Scenario 4.2 — Sub-Task + Main Task with Accuracy Reporting (8/9 criteria · Partial Pass)

**Category:** 4 — Dataset Evaluation  
**Source of Truth:** `documentation/examples/dataset_evaluation.py` lines 36–77; `golden_tests/test_cookbook_examples.py` lines 350–397  

<details><summary>Prompt</summary>

```
Write a two-level benchmark evaluation:
1. A sub-task (`store_task=False`) that takes `llm, question, answer` and returns a dict with "question", "gold_target", "predicted_answer", and "is_correct" fields.
2. A main task that receives `llm` and a `df` parameter, calls the sub-task's `.evaluate()` inside `kbench.client.enable_cache()`, and returns `tuple[float, float]` of (accuracy, std).
3. Use this data: "Capital of Singapore?" → "Singapore", "Capital of France?" → "Paris"
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench
import pandas as pd

# %%
@kbench.task(name="single_qa", store_task=False)
def single_qa(llm, question: str, answer: str) -> dict:
    response = llm.prompt(question)
    is_correct = answer.lower() in response.lower()
    kbench.assertions.assert_true(
        is_correct,
        expectation=f"Expected '{answer}' in response.",
    )
    return {
        "question": question,
        "gold_target": answer,
        "predicted_answer": response,
        "is_correct": is_correct,
    }

# %%
@kbench.task(name="qa_benchmark")
def qa_benchmark(llm, df: pd.DataFrame) -> tuple[float, float]:
    with kbench.client.enable_cache():
        runs = single_qa.evaluate(
            llm=[llm],
            evaluation_data=df,
            n_jobs=2,
            timeout=120,
            max_attempts=3,
            retry_delay=15,
            remove_run_files=True,
        )

    eval_df = runs.as_dataframe()
    correctness = eval_df.result.str.get("is_correct")
    accuracy = float(correctness.mean())
    std = float(correctness.std())
    return accuracy, std

# %%
df = pd.DataFrame([
    {"question": "Capital of Singapore?", "answer": "Singapore"},
    {"question": "Capital of France?", "answer": "Paris"},
])

qa_benchmark.run(kbench.llm, df=df)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Sub-task has `store_task=False` in decorator
  - _judge:_ The sub-task decorator `@kbench.task(name="single_qa", store_task=False)` contains `store_task=False` explicitly.
- ✅ Sub-task returns `-> dict` with all four fields
  - _judge:_ The sub-task `single_qa` has `-> dict` return annotation and returns a dict with all four fields: 'question', 'gold_target', 'predicted_answer', 'is_correct'.
- ✅ Sub-task checks `answer.lower() in response.lower()`
  - _judge:_ The code contains `answer.lower() in response.lower()` literally in the sub-task function body.
- ✅ Main task wraps evaluate in `with kbench.client.enable_cache():`
  - _judge:_ The main task `qa_benchmark` wraps `single_qa.evaluate(...)` inside `with kbench.client.enable_cache():` exactly as required.
- ✅ Main task calls `sub_task.evaluate(llm=[llm], evaluation_data=df, ...)`
  - _judge:_ The code calls `single_qa.evaluate(llm=[llm], evaluation_data=df, ...)` which matches the criterion pattern of `sub_task.evaluate(llm=[llm], evaluation_data=df, ...)`
- ✅ Main task uses `runs.as_dataframe()` to get results
  - _judge:_ The code contains `runs.as_dataframe()` explicitly in the main task `qa_benchmark`.
- ❌ Main task computes `eval_df.result.str.get("is_correct").mean()` and `.std()`
  - _judge:_ Agent stores result in `correctness` variable and calls `.mean()` and `.std()` on it, rather than chaining directly as `eval_df.result.str.get('is_correct').mean()` and `.std()` as required.
- ✅ Main task returns `-> tuple[float, float]`
  - _judge:_ The main task function `qa_benchmark` has the return type annotation `-> tuple[float, float]` explicitly present in the code.
- ✅ Uses `float()` to convert from numpy types
  - _judge:_ The code uses `float(correctness.mean())` and `float(correctness.std())`, which converts numpy types via `float()` as required.

---

<a id="scenario-4-3"></a>
### ⚠️ Scenario 4.3 — Evaluate with Stop Condition and Retry (4/7 criteria · Partial Pass)

**Category:** 4 — Dataset Evaluation  
**Source of Truth:** `documentation/examples/dataset_evaluation.py` lines 57–67; `src/kaggle_benchmarks/tasks.py` `evaluate()` lines 154–204  

<details><summary>Prompt</summary>

```
Show the full set of parameters available when calling `.evaluate()` on a dataset. Write a task that uses `stop_condition`, `max_attempts`, `retry_delay`, `timeout`, and `remove_run_files`.
```

</details>

<details><summary>Agent response</summary>

````
Here's the full parameter set for `.evaluate()` and a task demonstrating each one.

## Full `.evaluate()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `list[Actor]` | — | **Required.** List of LLM actors to evaluate (each model × each row = one job) |
| `evaluation_data` | `DataFrame` | — | **Required.** Pandas DataFrame; column names must match task parameter names |
| `n_jobs` | `int` | `1` | Parallel workers — set >1 to run rows concurrently |
| `timeout` | `int` | `None` | Per-job timeout in seconds; job is marked failed if exceeded |
| `max_attempts` | `int` | `3` | How many times to retry a failed/timed-out job before giving up |
| `retry_delay` | `int` | `15` | Seconds to wait between retry attempts |
| `stop_condition` | `Callable[[list[Run]], bool]` | `None` | Called after each job completes; evaluation stops when it returns `True` |
| `remove_run_files` | `bool` | `False` | Delete per-job result files after aggregation (keeps storage clean) |

> **Note:** Any extra keyword arguments beyond the above are forwarded as static parameters to the task function. For example, if your task has a `critic` parameter not in the DataFrame, pass `critic=[critic_llm]` to `.evaluate()`.

---

## Example Task

ʼʼʼpython
import kaggle_benchmarks as kbench
import pandas as pd

# %%
# A sub-task that evaluates a single question.
# store_task=False because it's called inside the main benchmark, not run standalone.
@kbench.task(name="single_question", store_task=False)
def single_question(llm, question: str, expected_keyword: str) -> bool:
    """Ask one factual question and check the response contains the expected keyword."""
    response = llm.prompt(question)
    is_correct = expected_keyword.lower() in response.lower()
    kbench.assertions.assert_true(
        is_correct,
        expectation=f"Response should mention '{expected_keyword}'.",
    )
    return is_correct


# %%
# A small test dataset — column names must match task parameters exactly.
questions_df = pd.DataFrame([
    {"question": "What is the chemical symbol for water?",          "expected_keyword": "H2O"},
    {"question": "Which planet is closest to the Sun?",             "expected_keyword": "Mercury"},
    {"question": "Who wrote Romeo and Juliet?",                     "expected_keyword": "Shakespeare"},
    {"question": "What is the speed of light in a vacuum?",         "expected_keyword": "299"},
    {"question": "What organ pumps blood through the human body?",  "expected_keyword": "heart"},
    {"question": "In what year did the First World War begin?",      "expected_keyword": "1914"},
    {"question": "What is the largest ocean on Earth?",             "expected_keyword": "Pacific"},
    {"question": "What gas do plants absorb during photosynthesis?", "expected_keyword": "carbon"},
])

# %%
# We want to stop as soon as 5 questions are answered, regardless of total rows.
# The lambda receives the list of completed Run objects accumulated so far.
EARLY_STOP_AT = 5
stop_when_enough = lambda runs: len(runs) >= EARLY_STOP_AT


# %%
@kbench.task(name="general_knowledge_benchmark")
def general_knowledge_benchmark(llm) -> tuple[int, int]:
    """
    Evaluate the LLM on a set of general-knowledge questions.

    Uses all five .evaluate() control parameters:
      - stop_condition  : halt after 5 completed runs (not all 8)
      - timeout         : each question gets at most 30 seconds
      - max_attempts    : retry a question up to 2 times on failure/timeout
      - retry_delay     : wait 10 seconds between retries
      - remove_run_files: clean up per-job files after aggregation
    """
    results = single_question.evaluate(
        llm=[llm],
        evaluation_data=questions_df,

        # Parallelism — run up to 3 questions at once.
        n_jobs=3,

        # Kill any job that takes longer than 30 seconds.
        timeout=30,

        # Retry timed-out or errored jobs up to 2 times total.
        max_attempts=2,

        # Wait 10 seconds before each retry attempt.
        retry_delay=10,

        # Stop once 5 jobs have completed (don't bother running the remaining 3).
        stop_condition=stop_when_enough,

        # Delete the individual run files after collecting results.
        remove_run_files=True,
    )

    eval_df = results.as_dataframe()

    # Count how many rows completed and how many passed.
    total_run   = len(eval_df)
    total_passed = int(eval_df["result"].sum())

    kbench.assertions.assert_true(
        total_run <= EARLY_STOP_AT,
        expectation=f"stop_condition should have halted at ≤{EARLY_STOP_AT} runs.",
    )
    kbench.assertions.assert_true(
        total_passed >= 3,
        expectation="At least 3 out of 5 sampled questions should be answered correctly.",
    )

    return total_passed, total_run


# %%
general_knowledge_benchmark.run(kbench.llm)
ʼʼʼ

### How each parameter is doing work here

ʼʼʼ
stop_condition=stop_when_enough
ʼʼʼ
The lambda `lambda runs: len(runs) >= 5` is checked after every completed job. The moment the fifth job finishes, `evaluate()` stops scheduling new jobs. With 8 rows and 3 parallel workers you'd normally run all 8; this caps the run at 5.

ʼʼʼ
timeout=30
ʼʼʼ
If the LLM takes more than 30 seconds to respond to a single question, that job is marked as failed and (if `max_attempts` allows) queued for a retry.

ʼʼʼ
max_attempts=2
ʼʼʼ
A job that times out or raises an unhandled exception will be retried up to 2 times total (meaning 1 initial attempt + 1 retry). After 2 failures the job is recorded as permanently failed.

ʼʼʼ
retry_delay=10
ʼʼʼ
Between the initial failure and the first retry, the runner waits 10 seconds. This avoids hammering the API immediately after a transient error (rate-limit, network blip, etc.).

ʼʼʼ
remove_run_files=True
ʼʼʼ
Each parallel job writes a small result file to disk. After `as_dataframe()` aggregates them, this flag deletes those intermediate files — useful on Kaggle where disk space is limited.
````

</details>

**Criteria:**

- ❌ Uses `stop_condition=lambda runs: len(runs) == df.shape[0]`
  - _judge:_ The agent uses `stop_condition=lambda runs: len(runs) >= EARLY_STOP_AT` (via `stop_when_enough`), not the exact literal `stop_condition=lambda runs: len(runs) == df.shape[0]` required by the criterion.
- ✅ Uses `max_attempts=` (e.g., 1 or 3)
  - _judge:_ The code contains `max_attempts=2` which satisfies the criterion of using `max_attempts=` (with a value of 1 or 3 being examples, but any integer value is acceptable).
- ❌ Uses `retry_delay=15` (in seconds)
  - _judge:_ The code uses `retry_delay=10`, not `retry_delay=15` as required by the criterion.
- ❌ Uses `timeout=120` (per-job timeout in seconds)
  - _judge:_ The agent uses `timeout=30`, not `timeout=120` as required by the criterion.
- ✅ Uses `remove_run_files=True`
  - _judge:_ The code contains `remove_run_files=True` explicitly in the `.evaluate()` call.
- ✅ Uses `n_jobs=` for parallel execution
  - _judge:_ The code contains 'n_jobs=3' in the .evaluate() call, satisfying the n_jobs= criterion.
- ✅ Uses `llm=[llm]` (list format)
  - _judge:_ The code contains `llm=[llm]` in the .evaluate() call, matching the list format criterion exactly.

---

<a id="scenario-4-4"></a>
### ❌ Scenario 4.4 — Multi-Model Comparison on Dataset (3/7 criteria · Fail)

**Category:** 4 — Dataset Evaluation  
**Source of Truth:** `documentation/examples/dataset_evaluation.py` lines 80–97 (multi-model evaluation)  

<details><summary>Prompt</summary>

```
Write a benchmark that evaluates TWO different models on the same Q&A dataset and compares their accuracy. Use `kbench.llm` and `kbench.judge_llm` as the two models.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
# %%
import kaggle_benchmarks as kbench
import pandas as pd

# %%
QA_DATA = pd.DataFrame([
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the chemical symbol for water?", "answer": "H2O"},
    {"question": "How many sides does a hexagon have?", "answer": "6"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"question": "What is the speed of light in km/s (approximately)?", "answer": "300000"},
    {"question": "What element does 'O' represent on the periodic table?", "answer": "Oxygen"},
    {"question": "In what year did World War II end?", "answer": "1945"},
    {"question": "What is the square root of 144?", "answer": "12"},
    {"question": "What gas do plants absorb during photosynthesis?", "answer": "CO2"},
])

# %%
@kbench.task(name="single_qa", store_task=False)
def single_qa(llm, question: str, answer: str) -> bool:
    response = llm.prompt(
        f"{question}\n\nAnswer concisely in a few words.",
    )
    is_correct = answer.lower() in response.lower()
    kbench.assertions.assert_true(
        is_correct,
        expectation=f"Response should contain '{answer}'. Got: '{response}'",
    )
    return is_correct

# %%
@kbench.task(name="model_accuracy")
def model_accuracy(llm) -> float:
    with kbench.client.enable_cache():
        results = single_qa.evaluate(
            llm=[llm],
            evaluation_data=QA_DATA,
            n_jobs=3,
            timeout=60,
            max_attempts=2,
            remove_run_files=True,
        )
    scores = results.as_dataframe()
    accuracy = float(scores["result"].mean())
    kbench.assertions.assert_true(
        accuracy >= 0.0,
        expectation="Accuracy should be a valid non-negative float.",
    )
    return accuracy

# %%
@kbench.task(name="compare_models")
def compare_models(llm) -> dict:
    llm_run = model_accuracy.run(kbench.llm)
    judge_run = model_accuracy.run(kbench.judge_llm)

    llm_accuracy = llm_run.result
    judge_accuracy = judge_run.result

    winner = (
        "llm" if llm_accuracy > judge_accuracy
        else "judge_llm" if judge_accuracy > llm_accuracy
        else "tie"
    )

    kbench.assertions.assert_true(
        0.0 <= llm_accuracy <= 1.0,
        expectation="llm accuracy should be between 0 and 1.",
    )
    kbench.assertions.assert_true(
        0.0 <= judge_accuracy <= 1.0,
        expectation="judge_llm accuracy should be between 0 and 1.",
    )

    return {
        "llm_accuracy": llm_accuracy,
        "judge_llm_accuracy": judge_accuracy,
        "winner": winner,
        "difference": abs(llm_accuracy - judge_accuracy),
    }

compare_models.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Defines a per-row sub-task with `store_task=False`
  - _judge:_ The `single_qa` task is decorated with `@kbench.task(name="single_qa", store_task=False)`, which explicitly includes `store_task=False`.
- ❌ Creates a list of multiple models: `llms = [kbench.llm, kbench.judge_llm]`
  - _judge:_ The code never creates `llms = [kbench.llm, kbench.judge_llm]`. It runs models separately via individual `.run()` calls instead of building the required list.
- ❌ Calls `.evaluate(llm=llms, evaluation_data=df, ...)`
  - _judge:_ The code calls `.evaluate(llm=[llm], ...)` with a single-element list, not `llm=llms` where llms contains multiple models (both kbench.llm and kbench.judge_llm).
- ❌ Calculates `n_total = len(llms) * df.shape[0]` for the stop condition
  - _judge:_ The agent's code does not contain `n_total = len(llms) * df.shape[0]` anywhere. It uses a different approach without this stop condition calculation.
- ❌ Uses `stop_condition=lambda runs: len(runs) == n_total`
  - _judge:_ The agent's code does not contain `stop_condition=lambda runs: len(runs) == n_total` anywhere in the response.
- ✅ Uses `n_jobs=` for parallel execution
  - _judge:_ The code contains `n_jobs=3` in the `single_qa.evaluate(...)` call, satisfying the `n_jobs=` criterion.
- ✅ Calls `.as_dataframe()` for results
  - _judge:_ The code calls `.as_dataframe()` on line `scores = results.as_dataframe()` within the `model_accuracy` function.

---

<a id="scenario-4-5"></a>
### ✅ Scenario 4.5 — Math Word Problems Evaluation (6/6 criteria · Strong Pass)

**Category:** 4 — Dataset Evaluation  
**Source of Truth:** `golden_tests/test_code_generator.py` lines 142–145 (`math-word-problems`)  

<details><summary>Prompt</summary>

```
Write a benchmark that tests the LLM on 5 math word problems and returns the overall accuracy as a float. The problems are:
1. "What is 15% of 200?" → 30
2. "What is 7 × 8?" → 56
3. "What is the square root of 144?" → 12
4. "If a train travels 60 mph for 2.5 hours, how far does it go?" → 150
5. "What is 1000 ÷ 8?" → 125
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench
import pandas as pd

# %%
@kbench.task(name="single_math_problem", store_task=False)
def single_math_problem(llm, question: str, expected: int) -> bool:
    answer = llm.prompt(
        f"{question}\nRespond with just the number, no units or explanation.",
        schema=int,
    )
    is_correct = answer == expected
    kbench.assertions.assert_equal(
        expected, answer,
        expectation=f"Answer to '{question}' should be {expected}."
    )
    return is_correct

# %%
problems = pd.DataFrame([
    {"question": "What is 15% of 200?",                              "expected": 30},
    {"question": "What is 7 × 8?",                                   "expected": 56},
    {"question": "What is the square root of 144?",                  "expected": 12},
    {"question": "If a train travels 60 mph for 2.5 hours, how far does it go?", "expected": 150},
    {"question": "What is 1000 ÷ 8?",                                "expected": 125},
])

# %%
@kbench.task(name="math_word_problems_accuracy")
def math_word_problems_accuracy(llm) -> float:
    results = single_math_problem.evaluate(
        llm=[llm],
        evaluation_data=problems,
        n_jobs=2,
        timeout=60,
        max_attempts=3,
        retry_delay=5,
        remove_run_files=True,
    )
    scores = results.as_dataframe()
    accuracy = float(scores["result"].mean())
    kbench.assertions.assert_true(
        0.0 <= accuracy <= 1.0,
        expectation="Accuracy must be between 0 and 1."
    )
    return accuracy

# %%
math_word_problems_accuracy.run(kbench.llm)
ʼʼʼ

### Key design decisions

| Decision | Reasoning |
|----------|-----------|
| `store_task=False` on `single_math_problem` | It's a helper called inside the main task, not a standalone benchmark |
| `schema=int` | Forces a clean integer reply — no post-processing needed |
| `assert_equal` inside the sub-task | Each problem's pass/fail is recorded individually so you can see which ones failed |
| `-> float` return annotation | Required for the runner to treat the return value as a numeric score |
| `scores["result"].mean()` | `result` column holds the `bool` returned per row — mean gives accuracy directly |
| Separate `problems` DataFrame cell | Easy to extend by adding rows without touching task logic |
````

</details>

**Criteria:**

- ✅ Creates a DataFrame or list of questions with expected answers
  - _judge:_ The code creates a `pd.DataFrame` named `problems` containing 5 questions with their expected answers, satisfying the criterion.
- ✅ Uses structured output (`schema=int` or `schema=float`) OR regex/string matching to check answers
  - _judge:_ Uses `schema=int` in llm.prompt call, which satisfies the structured output requirement.
- ✅ Returns `-> float` accuracy score between 0.0 and 1.0
  - _judge:_ The main task function `math_word_problems_accuracy` has `-> float` return annotation and returns a float accuracy value between 0.0 and 1.0.
- ✅ Either uses `.evaluate()` on a DataFrame OR loops through questions manually
  - _judge:_ Code uses `.evaluate()` on a DataFrame (`evaluation_data=problems`) via `single_math_problem.evaluate(...)`, satisfying the criterion.
- ✅ Uses kbench assertions for per-question checks
  - _judge:_ The code uses kbench.assertions.assert_equal and kbench.assertions.assert_true for per-question checks inside single_math_problem and the main task.
- ✅ Properly handles numeric comparison (not just string matching)
  - _judge:_ Uses `schema=int` to force numeric parsing and `answer == expected` compares integers directly, not strings. Numeric comparison is properly handled.

---

<a id="scenario-5-1"></a>
### ⚠️ Scenario 5.1 — Structured Output + Judge Evaluation (5/7 criteria · Partial Pass)

**Category:** 5 — Medium — Combining Basics  
**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 149–180 (`assess_with_judge_task`)  

<details><summary>Prompt</summary>

```
Write a benchmark task that:
1. Asks the LLM "What is Kaggle?" and stores the response.
2. Asserts the response contains "platform" using `assert_in`.
3. Uses `assess_response_with_judge()` with a judge LLM to evaluate whether:
   - The answer mentions data science or machine learning
   - The answer mentions competitions
4. Iterates over the judge results and asserts each one passed.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench

@kbench.task(name="what_is_kaggle")
def what_is_kaggle(llm):
    response = llm.prompt("What is Kaggle?")

    kbench.assertions.assert_in(
        "platform", response.lower(),
        expectation="Response should contain the word 'platform'."
    )

    assessment = kbench.assertions.assess_response_with_judge(
        criteria=[
            "The answer mentions data science or machine learning.",
            "The answer mentions competitions.",
        ],
        response_text=response,
        judge_llm=kbench.judge_llm,
    )

    if assessment is None:
        kbench.assertions.assert_fail("Judge LLM failed to return an assessment.")
    else:
        for result in assessment.results:
            kbench.assertions.assert_true(
                result.passed,
                expectation=f"'{result.criterion}': {result.reason}",
            )

what_is_kaggle.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ❌ Task has two LLM params: `llm` and `judge_llm`
  - _judge:_ The task function signature is `def what_is_kaggle(llm)` with only one parameter. It lacks `judge_llm` as a second LLM param; instead uses `kbench.judge_llm` globally.
- ✅ Uses `kbench.assertions.assert_in("platform", response.lower())`
  - _judge:_ The code contains `kbench.assertions.assert_in("platform", response.lower()` which matches the required call exactly.
- ✅ Uses `kbench.assertions.assess_response_with_judge(criteria=[...], response_text=response, judge_llm=judge_llm)`
  - _judge:_ The code calls `kbench.assertions.assess_response_with_judge(criteria=[...], response_text=response, judge_llm=kbench.judge_llm)` which matches the required signature pattern.
- ✅ RECOMMENDED: Checks judge result for `None` before accessing `.results` (the function can return `None` on failure)
  - _judge:_ The code checks `if assessment is None:` before accessing `assessment.results`, satisfying the criterion.
- ✅ Iterates: `for result in assessment.results:`
  - _judge:_ The agent's code contains exactly `for result in assessment.results:` in the else block, matching the criterion literally.
- ✅ Uses `kbench.assertions.assert_true(result.passed, expectation=f"...{result.criterion}...{result.reason}")`
  - _judge:_ The code uses `kbench.assertions.assert_true(result.passed, expectation=f"'{result.criterion}': {result.reason}")` which matches the required pattern with result.passed, result.criterion, and result.reason.
- ❌ Calls `.run(kbench.llm, kbench.judge_llm)`
  - _judge:_ Code calls `what_is_kaggle.run(kbench.llm)` with only one argument, missing `kbench.judge_llm` as the second argument. Criterion requires `.run(kbench.llm, kbench.judge_llm)`.

---

<a id="scenario-5-2"></a>
### ⚠️ Scenario 5.2 — Hallucination Detection with Structured Output (4/5 criteria · Partial Pass)

**Category:** 5 — Medium — Combining Basics  
**Source of Truth:** `documentation/examples/demo_candidates_bm1.py` lines 98–118; Skill file §9 Pattern C  

<details><summary>Prompt</summary>

```
Write a benchmark task that detects hallucination. Ask the LLM about a fictitious concept (e.g., "the Zipflanger Theorem in particle physics"). Use a dict schema `{"answer": bool, "explanation": str}`. Assert the model says it doesn't exist (`assert_false` on `answer`) and the explanation contains negation words.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench

@kbench.task(name="hallucination_detection")
def hallucination_detection(llm):
    """Ask the LLM about a fictitious concept and verify it correctly says it doesn't exist."""
    response = llm.prompt(
        "In particle physics, the Zipflanger Theorem describes the relationship between "
        "quantum spin states and hypothetical dark matter resonance frequencies. "
        "Is the Zipflanger Theorem a real, established concept in particle physics?",
        schema={"answer": bool, "explanation": str},
    )

    kbench.assertions.assert_false(
        response.answer,
        expectation="The Zipflanger Theorem is fictitious — model should answer False.",
    )
    kbench.assertions.assert_contains_regex(
        r"(?i)\b(not|no|never|doesn't|does not|don't|do not|isn't|is not|didn't|did not|non-existent|fictitious|fabricated|made.up|unrecognized|unknown|fake)\b",
        response.explanation,
        expectation="Explanation should contain negation or denial language.",
    )

hallucination_detection.run(kbench.llm)
ʼʼʼ

Key decisions made here:

- **Fictional concept design** — the Zipflanger Theorem is seeded with plausible-sounding jargon ("quantum spin states", "dark matter resonance frequencies") to make it tempting to hallucinate a confident "yes". The question is then asked directly to force a boolean commitment.

- **`assert_false` on `answer`** — the dict schema coerces the LLM's response into a typed `bool`, so we can check it exactly rather than parsing free text for "yes/no".

- **Broad negation regex** — the `\b` word-boundary anchors prevent false matches (e.g. "notable" matching "not"), and the alternation covers the natural ways a model phrases denial: contractions, full forms, and synonyms like "fictitious" or "fabricated". The `(?i)` flag handles any capitalisation.

- **No judge LLM needed** — because the schema gives us a structured boolean, we can assert deterministically rather than delegating to a second model call, keeping the task fast and reproducible.
````

</details>

**Criteria:**

- ✅ Uses `schema={"answer": bool, "explanation": str}` (dict schema)
  - _judge:_ The code contains exactly `schema={"answer": bool, "explanation": str}` as a dict schema passed to llm.prompt(), matching the criterion.
- ✅ Uses `kbench.assertions.assert_false(response.answer)`
  - _judge:_ The code explicitly contains `kbench.assertions.assert_false(response.answer, ...)` which matches the required criterion exactly.
- ❌ Uses `kbench.assertions.assert_contains_regex(r"(not|never|no|doesn't|didn't)", response.explanation.lower())`
  - _judge:_ The criterion requires the exact call `assert_contains_regex(r"(not|never|no|doesn't|didn't)", response.explanation.lower())` but the agent uses a different regex pattern and passes `response.explanation` (without `.lower()`) as the argument.
- ✅ Does NOT use plain Python `assert`
  - _judge:_ No plain Python `assert` statements found; all assertions use `kbench.assertions.*` methods.
- ✅ Has `@kbench.task()` decorator
  - _judge:_ The code contains `@kbench.task(name="hallucination_detection")` which satisfies the `@kbench.task()` decorator requirement (parentheses with args still counts as the decorator).

---

<a id="scenario-5-3"></a>
### ✅ Scenario 5.3 — System Prompt + Structured Output + Code Execution (7/7 criteria · Strong Pass)

**Category:** 5 — Medium — Combining Basics  
**Source of Truth:** `documentation/examples/demo_candidates_bm1.py` lines 151–189; Skill file §9 Pattern I  

<details><summary>Prompt</summary>

```
Write a benchmark task that:
1. Sets the LLM as "an expert Python programmer" using a system message.
2. Gives the LLM buggy code: `fruits = ['apple', 'orange' 'banana', 'peach']` (missing comma).
3. Uses a dataclass with `has_bugs: bool` and `fixed_code: str` as schema.
4. Asserts `has_bugs` is True.
5. Extracts and runs the fixed code.
6. Asserts the output is "4".
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench
from dataclasses import dataclass


@dataclass
class CodeAnalysis:
    has_bugs: bool
    fixed_code: str


@kbench.task(name="code_analysis")
def analyze_code(llm):
    buggy_code = """
fruits = ['apple', 'orange' 'banana', 'peach']
print(len(fruits))
"""
    kbench.system.send("You are an expert Python programmer.")
    response = llm.prompt(
        f"Does this code have bugs? Fix it if so.\n\n{buggy_code}",
        schema=CodeAnalysis,
    )

    kbench.assertions.assert_true(
        response.has_bugs,
        expectation="Should detect the missing comma between 'orange' and 'banana'.",
    )

    fixed = kbench.tools.python.extract_code(response.fixed_code)
    output = kbench.tools.python.script_runner.run_code(fixed)

    kbench.assertions.assert_empty(
        output.stderr.strip(),
        expectation="Fixed code should run without errors.",
    )
    kbench.assertions.assert_equal(
        "4",
        output.stdout.strip(),
        expectation="Fixed code should print 4 (the list has 4 fruits).",
    )


analyze_code.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Uses `kbench.system.send("You are an expert Python programmer.")`
  - _judge:_ The exact string `kbench.system.send("You are an expert Python programmer.")` appears literally in the agent's response.
- ✅ Defines `@dataclass` with `has_bugs: bool` and `fixed_code: str`
  - _judge:_ The code defines `@dataclass` class `CodeAnalysis` with both `has_bugs: bool` and `fixed_code: str` fields, satisfying the criterion exactly.
- ✅ Uses `llm.prompt("...", schema=CodeAnalysis)`
  - _judge:_ The code contains `llm.prompt(f"...", schema=CodeAnalysis,)` which matches the required pattern `llm.prompt("...", schema=CodeAnalysis)`.
- ✅ Uses `kbench.assertions.assert_true(response.has_bugs)`
  - _judge:_ The code contains `kbench.assertions.assert_true(response.has_bugs, ...)` exactly as required.
- ✅ Uses `kbench.tools.python.extract_code(response.fixed_code)`
  - _judge:_ The agent's code contains exactly `kbench.tools.python.extract_code(response.fixed_code)` on line `fixed = kbench.tools.python.extract_code(response.fixed_code)`.
- ✅ Uses `kbench.tools.python.script_runner.run_code(code)`
  - _judge:_ The agent's response contains `kbench.tools.python.script_runner.run_code(fixed)` which matches the required `kbench.tools.python.script_runner.run_code(code)` call pattern.
- ✅ Uses `kbench.assertions.assert_equal("4", output.stdout.strip())`
  - _judge:_ The code contains exactly `kbench.assertions.assert_equal("4", output.stdout.strip()` as required by the criterion.

---

<a id="scenario-5-4"></a>
### ✅ Scenario 5.4 — Custom Assertion + Task (7/7 criteria · Strong Pass)

**Category:** 5 — Medium — Combining Basics  
**Source of Truth:** `src/kaggle_benchmarks/assertions.py` lines 113–175 (`assertion_handler`); Skill file §5 "Custom Assertions"  

<details><summary>Prompt</summary>

```
Define a reusable custom assertion `assert_is_palindrome` using `@assertion_handler()` that checks if a string is a palindrome. Then write a task that asks the LLM to generate a palindrome word and validates it with the custom assertion.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench
from kaggle_benchmarks.assertions import assertion_handler, AssertionResult


# %%
@assertion_handler()
def assert_is_palindrome(text: str, expectation: str) -> AssertionResult:
    """Assert that `text` (after lowercasing and stripping spaces) is a palindrome."""
    normalized = text.lower().replace(" ", "")
    is_palindrome = normalized == normalized[::-1] and len(normalized) > 0
    return AssertionResult(
        passed=is_palindrome,
        expectation=expectation,
    )


# %%
@kbench.task(name="palindrome_generator")
def palindrome_generator(llm):
    """Ask the LLM to produce a palindrome word and verify it actually is one."""
    response = llm.prompt(
        "Give me a single palindrome word (a word that reads the same forwards and backwards). "
        "Reply with just the word, nothing else."
    )

    # Strip surrounding whitespace/punctuation the model might add
    word = response.strip().strip(".,!?\"'").strip()

    assert_is_palindrome(
        word,
        expectation=(
            f"The word '{word}' should read the same forwards and backwards "
            "(e.g. 'racecar', 'level', 'madam')."
        ),
    )


palindrome_generator.run(kbench.llm)
ʼʼʼ

**How it works:**

| Step | Detail |
|------|--------|
| `@assertion_handler()` | Registers the function with the benchmark framework so its result is **recorded and tracked**, not just raised |
| `-> AssertionResult` | Required return-type annotation — tells the framework this is an assertion |
| Normalization | `.lower().replace(" ", "")` makes the check case-insensitive and ignores spaces, so *"Race car"* would also pass |
| `len(normalized) > 0` | Guards against an empty string trivially passing the palindrome check |
| `.strip().strip(".,!?\"'")` | Cleans the LLM's reply before passing it to the assertion — makes the task robust to models that add punctuation or quotes |
| `store_task=True` (default) | The task result is stored; no need to override it here since this is a top-level task |
````

</details>

**Criteria:**

- ✅ Imports `assertion_handler` and `AssertionResult` from `kaggle_benchmarks.assertions`
  - _judge:_ Both `assertion_handler` and `AssertionResult` are explicitly imported from `kaggle_benchmarks.assertions` in the code.
- ✅ Uses `@assertion_handler()` decorator on custom function
  - _judge:_ The code contains `@assertion_handler()` decorator applied to the custom function `assert_is_palindrome`, exactly matching the criterion.
- ✅ Return type annotated as `-> AssertionResult`
  - _judge:_ The function `assert_is_palindrome` has `-> AssertionResult` return type annotation as required.
- ✅ Returns `AssertionResult(passed=..., expectation=...)`
  - _judge:_ The code returns `AssertionResult(passed=is_palindrome, expectation=expectation)` which matches the required `AssertionResult(passed=..., expectation=...)` pattern.
- ✅ Cleans the string (e.g., `.lower()`, strip spaces) before checking palindrome
  - _judge:_ The code uses `.lower().replace(" ", "")` to normalize the string before checking if it's a palindrome, satisfying the cleaning requirement.
- ✅ Uses the custom assertion inside a `@kbench.task()` decorated function
  - _judge:_ The custom assertion `assert_is_palindrome` (decorated with `@assertion_handler()`) is called inside `palindrome_generator`, which is decorated with `@kbench.task(name='palindrome_generator')`.
- ✅ Does NOT use `@assertion_handler(raises_assertion_error=True)` unless explicitly needed
  - _judge:_ The agent uses `@assertion_handler()` with no arguments (default `raises_assertion_error=False`), never using `raises_assertion_error=True`.

---

<a id="scenario-5-5"></a>
### ⚠️ Scenario 5.5 — Multi-Turn Game with Judge (7/9 criteria · Partial Pass)

**Category:** 5 — Medium — Combining Basics  
**Source of Truth:** `documentation/examples/play_20_questions.py` lines 38–76; Skill file §9 Pattern F  

<details><summary>Prompt</summary>

```
Write a benchmark task that plays 20 Questions. The LLM guesses an animal by asking yes/no questions. A judge LLM answers. The game loop runs up to 20 turns. Return True if the LLM guesses correctly.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench
from dataclasses import dataclass


@dataclass
class LLMTurn:
    question: str = ""
    guess: str = ""


@kbench.task(name="twenty_questions")
def twenty_questions(llm, judge_llm, target: str = "elephant") -> bool:
    kbench.system.send(
        "We are playing 20 Questions. I am thinking of an animal. "
        "Each turn, either ask a yes/no question OR make a final guess. "
        "Set 'question' to your yes/no question, or set 'guess' to your "
        "final answer if you think you know what the animal is. "
        "Only set one field per turn, leaving the other empty."
    )

    # Kick off the game
    response: LLMTurn = llm.prompt(
        "Let's begin! Ask your first yes/no question or make a guess.",
        schema=LLMTurn,
    )

    for turn in range(1, 21):
        # ── Guess branch ──────────────────────────────────────────────
        if response.guess.strip():
            guessed = response.guess.strip().lower()
            correct = target.lower() in guessed or guessed in target.lower()

            kbench.assertions.assert_true(
                correct,
                expectation=(
                    f"Turn {turn}: LLM guessed '{response.guess}' "
                    f"but the animal was '{target}'."
                ),
            )
            return correct

        # ── Question branch ───────────────────────────────────────────
        question = response.question.strip()
        if not question:
            kbench.assertions.assert_fail(
                f"Turn {turn}: LLM returned an empty question and no guess."
            )
            return False

        # Judge answers yes/no in an isolated chat so it never sees the
        # guesser's conversation history.
        with kbench.chats.new(f"judge_turn_{turn}"):
            yes = judge_llm.prompt(
                f"We are playing 20 Questions. The secret animal is '{target}'. "
                f"Answer only 'yes' or 'no' to this question: {question}",
                schema=bool,
            )

        answer = "Yes." if yes else "No."
        turns_left = 20 - turn

        if turns_left == 0:
            next_prompt = (
                f"{answer} That was your last question — you must guess now! "
                "Set 'guess' to your final answer."
            )
        else:
            next_prompt = (
                f"{answer} ({turns_left} question(s) remaining.) "
                "Ask another yes/no question or set 'guess' if you're ready to guess."
            )

        response = llm.prompt(next_prompt, schema=LLMTurn)

    # Fell through 20 turns without a guess — check the final response
    if response.guess.strip():
        guessed = response.guess.strip().lower()
        correct = target.lower() in guessed or guessed in target.lower()
        kbench.assertions.assert_true(
            correct,
            expectation=(
                f"Final guess: LLM guessed '{response.guess}' "
                f"but the animal was '{target}'."
            ),
        )
        return correct

    kbench.assertions.assert_fail(
        f"LLM failed to guess the animal ('{target}') within 20 turns."
    )
    return False


# %%
twenty_questions.run(kbench.llm, kbench.judge_llm, target="elephant")
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Task accepts `llm, judge_llm, target: str` parameters
  - _judge:_ The function signature `def twenty_questions(llm, judge_llm, target: str = "elephant")` includes all three required parameters: `llm`, `judge_llm`, and `target: str`.
- ✅ Returns `-> bool`
  - _judge:_ The function definition `def twenty_questions(...) -> bool:` explicitly includes the `-> bool` return type annotation.
- ✅ Defines a structured output schema with `question` and `guess` fields
  - _judge:_ The dataclass LLMTurn defines both `question: str` and `guess: str` fields, forming a structured output schema with both required fields.
- ✅ Has a for/while loop (up to 20 turns)
  - _judge:_ The code contains `for turn in range(1, 21):` which is a for loop iterating up to 20 turns.
- ❌ Uses `with kbench.chats.new("Answering"):` to isolate judge's answer
  - _judge:_ The agent uses `kbench.chats.new(f"judge_turn_{turn}")` instead of the required literal string `"Answering"` in `kbench.chats.new("Answering")`
- ✅ Judge uses `schema=bool` for yes/no answers
  - _judge:_ The code contains `schema=bool` in the judge_llm.prompt call, matching the criterion exactly.
- ✅ Checks `response.guess` to detect a final guess vs. a question
  - _judge:_ The code checks `response.guess.strip()` (and `response.guess`) multiple times to detect a final guess vs. a question, satisfying the criterion.
- ✅ Returns True/False based on whether guess matches target
  - _judge:_ The function returns True/False based on whether guess matches target. 'correct = target.lower() in guessed or guessed in target.lower()' is computed and returned as bool in all guess branches.
- ❌ Calls `.run(kbench.llm, kbench.judge_llm, target="dog")`
  - _judge:_ The code calls `.run(kbench.llm, kbench.judge_llm, target="elephant")` but the criterion requires `target="dog"`.

---

<a id="scenario-5-6"></a>
### ✅ Scenario 5.6 — Sub-Tasks Composition with Float Score (6/6 criteria · Strong Pass)

**Category:** 5 — Medium — Combining Basics  
**Source of Truth:** `documentation/examples/simple_task.py` lines 24–52  

<details><summary>Prompt</summary>

```
Write a benchmark with a root task and two sub-tasks:
- Sub-task 1: Sends "Hello!" and asserts the response is not empty.
- Sub-task 2: Tells the LLM a name, then asks "What is my name?", and asserts the name is in the response.
- Root task: Runs sub-task 1 once and sub-task 2 twice (with "Alan Turing" and "Richard Feynman"), then returns the fraction of sub-tasks that passed as a float.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench


# %%
@kbench.task(name="hello_response", store_task=False)
def hello_response(llm) -> bool:
    response = llm.prompt("Hello!")
    kbench.assertions.assert_not_empty(response, expectation="Response should not be empty.")
    return bool(response)


# %%
@kbench.task(name="name_recall", store_task=False)
def name_recall(llm, name: str) -> bool:
    llm.prompt(f"My name is {name}.")
    response = llm.prompt("What is my name?")
    kbench.assertions.assert_contains_regex(
        rf"(?i){name}",
        response,
        expectation=f"Response should contain the name '{name}'.",
    )
    return bool(name.lower() in response.lower())


# %%
@kbench.task(name="memory_benchmark")
def memory_benchmark(llm) -> float:
    runs = [
        hello_response.run(llm),
        name_recall.run(llm, name="Alan Turing"),
        name_recall.run(llm, name="Richard Feynman"),
    ]
    passed = sum(r.passed for r in runs)
    return passed / len(runs)


memory_benchmark.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Defines 2–3 separate `@kbench.task()` functions
  - _judge:_ Three @kbench.task() decorators are defined: hello_response, name_recall, and memory_benchmark.
- ✅ Root task returns `-> float`
  - _judge:_ The root task `memory_benchmark` is defined with `-> float` return type annotation, satisfying the criterion.
- ✅ Sub-task functions use kbench assertions internally
  - _judge:_ Sub-task functions hello_response and name_recall both use kbench.assertions internally (assert_not_empty and assert_contains_regex).
- ✅ Root task collects runs: `runs = [subtask1.run(llm), subtask2.run(llm, "Alan Turing"), ...]`
  - _judge:_ The code contains `runs = [hello_response.run(llm), name_recall.run(llm, name="Alan Turing"), ...]` matching the pattern of runs collecting subtask results including subtask2.run(llm, "Alan Turing").
- ✅ Root task computes: `sum(r.passed for r in runs) / len(runs)`
  - _judge:_ The code contains `sum(r.passed for r in runs) / len(runs)` split across two lines but semantically equivalent, and literally `sum(r.passed for r in runs)` and `/ len(runs)` are both present.
- ✅ Calls root task's `.run(kbench.llm)`
  - _judge:_ The code calls `memory_benchmark.run(kbench.llm)` which is the root task's `.run(kbench.llm)` call.

---

<a id="scenario-5-7"></a>
### ✅ Scenario 5.7 — Code Generation with CSV Filtering (6/6 criteria · Strong Pass)

**Category:** 5 — Medium — Combining Basics  
**Source of Truth:** `golden_tests/test_code_generator.py` lines 127–130 (`csv-filter-age`)  

<details><summary>Prompt</summary>

```
Write a benchmark task that asks the LLM to write Python code that reads a CSV string with columns 'name' and 'age', filters rows where age > 30, and prints the count. The CSV data is: 'name,age\nAlice,25\nBob,35\nCharlie,40\nDiana,28'. Extract and run the code, then assert the output is "2".
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench

# %%
@kbench.task(name="csv_filter_count")
def csv_filter_count(llm):
    csv_data = "name,age\\nAlice,25\\nBob,35\\nCharlie,40\\nDiana,28"
    prompt = f"""Write Python code that does the following:
1. Reads the following CSV string (where \\n represents newlines): {csv_data}
2. Filters rows where age > 30
3. Prints the count of those rows as a single integer

The CSV data has columns 'name' and 'age'. Use the `csv` module or `io` module to parse it.
Print only the count number, nothing else."""

    response = llm.prompt(prompt)

    code = kbench.tools.python.extract_code(response)
    result = kbench.tools.python.script_runner.run_code(code)

    kbench.assertions.assert_empty(
        result.stderr.strip(),
        expectation="Code should run without errors.",
    )
    kbench.assertions.assert_equal(
        "2",
        result.stdout.strip(),
        expectation="Should print 2, since Bob (35) and Charlie (40) are the only rows with age > 30.",
    )

# %%
csv_filter_count.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Calls `llm.prompt()` with clear instructions including the CSV data
  - _judge:_ llm.prompt(prompt) is called with a prompt string that includes clear instructions and the CSV data embedded via f-string interpolation of csv_data.
- ✅ Uses `kbench.tools.python.extract_code(response)`
  - _judge:_ The code explicitly calls `kbench.tools.python.extract_code(response)` which matches the required symbol exactly.
- ✅ Uses `kbench.tools.python.script_runner.run_code(code)`
  - _judge:_ The response contains the exact call `kbench.tools.python.script_runner.run_code(code)` as required by the criterion.
- ✅ Asserts `result.stderr` is empty
  - _judge:_ The code contains `kbench.assertions.assert_empty(result.stderr.strip(), ...)` which asserts `result.stderr` is empty.
- ✅ Asserts `result.stdout.strip()` equals "2"
  - _judge:_ The code contains `kbench.assertions.assert_equal("2", result.stdout.strip(), ...)` which asserts that `result.stdout.strip()` equals "2".
- ✅ Uses kbench assertions
  - _judge:_ The code uses kbench.assertions.assert_empty and kbench.assertions.assert_equal, both are kbench assertions.

---

<a id="scenario-5-8"></a>
### ⚠️ Scenario 5.8 — Negative Assertion + Content Check (3/4 criteria · Partial Pass)

**Category:** 5 — Medium — Combining Basics  
**Source of Truth:** `golden_tests/test_code_generator.py` lines 132–135 (`why-sky-is-blue`)  

<details><summary>Prompt</summary>

```
Write a benchmark task that asks the LLM to explain why the sky is blue. Assert that the response:
1. Contains "scatter" or "Rayleigh" (scientific explanation).
2. Does NOT contain "paint" or "dye" (irrelevant terms).
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench


@kbench.task(name="sky_is_blue")
def sky_is_blue(llm):
    """Ask the LLM to explain why the sky is blue and validate the response."""
    response = llm.prompt("Why is the sky blue? Please explain the scientific reason.")

    kbench.assertions.assert_contains_regex(
        r"(?i)scatter|rayleigh",
        response,
        expectation="Response should mention light scattering or Rayleigh scattering.",
    )
    kbench.assertions.assert_not_contains_regex(
        r"(?i)paint|dye",
        response,
        expectation="Response should not contain irrelevant terms like 'paint' or 'dye'.",
    )


sky_is_blue.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ❌ Uses `kbench.assertions.assert_contains_regex(r"(?i)(scatter|rayleigh)", response)`
  - _judge:_ Regex pattern is r"(?i)scatter|rayleigh" (no capturing group) but criterion requires r"(?i)(scatter|rayleigh)" with parentheses around the alternation.
- ✅ Uses `kbench.assertions.assert_not_contains_regex(r"(?i)(paint|dye)", response)`
  - _judge:_ The code contains `kbench.assertions.assert_not_contains_regex(r"(?i)paint|dye", response)` which matches the required pattern exactly.
- ✅ Both positive and negative assertions are kbench assertions
  - _judge:_ Both assertions use kbench.assertions.assert_contains_regex (positive) and kbench.assertions.assert_not_contains_regex (negative), both are kbench assertions.
- ✅ Has `@kbench.task()` decorator
  - _judge:_ The code contains `@kbench.task(name="sky_is_blue")` which satisfies the `@kbench.task()` decorator requirement.

---

<a id="scenario-6-1"></a>
### ✅ Scenario 6.1 — Assertions vs Python assert (5/5 criteria · Strong Pass)

**Category:** 6 — Knowledge & Troubleshooting  
**Source of Truth:** Skill file §5; `src/kaggle_benchmarks/tasks.py` lines 134–135  

<details><summary>Prompt</summary>

```
Should I use Python's built-in `assert` or `kbench.assertions` in my tasks? What's the difference?
```

</details>

<details><summary>Agent response</summary>

````
Use `kbench.assertions` — here's why it matters.

## The Core Difference

**Python `assert`** stops execution immediately when it fails (raises `AssertionError`). **`kbench.assertions`** records pass/fail results and *lets execution continue*.

ʼʼʼpython
# ❌ Python assert — stops here if it fails; later checks never run
assert "Paris" in response
assert len(response) > 10  # Never reached if first assert fails

# ✅ kbench.assertions — both checks always run and are tracked
kbench.assertions.assert_contains_regex(r"(?i)paris", response)
kbench.assertions.assert_true(len(response) > 10)
ʼʼʼ

## Why "Execution Continues" Matters

Consider evaluating multiple criteria:

ʼʼʼpython
@kbench.task()
def haiku_check(llm):
    poem = llm.prompt("Write a haiku about rain.")

    # All three checks run regardless of individual failures
    kbench.assertions.assert_contains_regex(r"rain", poem.lower(),
        expectation="Should mention rain")
    kbench.assertions.assert_true(len(poem.split("\n")) == 3,
        expectation="Should have 3 lines")
    kbench.assertions.assert_true(len(poem) < 200,
        expectation="Should be brief")
ʼʼʼ

With Python `assert`, the first failure would hide whether the second and third checks would have passed. With `kbench.assertions`, you get a full picture of *every* criterion — useful for debugging and for understanding partial correctness.

## Tracking and Reporting

`kbench.assertions` results are attached to the `Run` object and stored/reported by the framework:

ʼʼʼpython
run = haiku_check.run(kbench.llm)
run.passed              # bool — True only if ALL assertions passed
run.assertion_results   # list of each assertion's pass/fail + reason
ʼʼʼ

Python `assert` failures are caught by the task runner (they won't crash the process), but they won't appear in `assertion_results` — you lose visibility.

## The `expectation=` Parameter

Every `kbench` assertion accepts an `expectation` string that explains *why* the check matters. This appears in run reports and makes debugging much faster:

ʼʼʼpython
kbench.assertions.assert_equal(
    42, result,
    expectation="Model should compute 6 × 7 = 42"
)
ʼʼʼ

## When `assert_fail` is Useful

For unconditional failures (e.g., judge returned `None`):

ʼʼʼpython
assessment = kbench.assertions.assess_response_with_judge(...)
if assessment is None:
    kbench.assertions.assert_fail("Judge failed to respond.")
    return  # Stop further checks in this task
ʼʼʼ

## Quick Reference

| Situation | Use |
|-----------|-----|
| Exact value match | `assert_equal(expected, actual)` |
| Keyword in text | `assert_contains_regex(r"(?i)word", text)` |
| Boolean condition | `assert_true(condition)` / `assert_false(condition)` |
| Membership check | `assert_in(item, collection)` |
| Non-empty result | `assert_not_empty(value)` |
| Unconditional failure | `assert_fail("reason")` |
| No exception thrown | `assert_raises_no_exceptions(fn)` |

The one exception: Python `assert` is fine in *test files* (e.g., pytest tests for your benchmark infrastructure itself) — just not inside `@kbench.task()` functions.
````

</details>

**Criteria:**

- ✅ Recommends `kbench.assertions` over plain `assert`
  - _judge:_ Response explicitly recommends `kbench.assertions` over plain `assert` throughout, with multiple code examples showing `kbench.assertions` usage.
- ✅ Explains kbench assertions are *recorded and tracked* in the run results
  - _judge:_ Response explicitly states kbench.assertions results are 'attached to the Run object and stored/reported by the framework' with run.assertion_results tracking each assertion's pass/fail, satisfying the 'recorded and tracked in run results' criterion.
- ✅ Explains kbench assertions do NOT raise exceptions — execution continues
  - _judge:_ Response explicitly states kbench.assertions 'lets execution continue' and records pass/fail, contrasting with Python assert which raises AssertionError and stops execution immediately.
- ✅ Explains plain `assert` IS caught by the task runner (doesn't crash the program) but is NOT tracked properly
  - _judge:_ Response explicitly states 'Python assert failures are caught by the task runner (they won't crash the process), but they won't appear in assertion_results — you lose visibility.'
- ✅ Does NOT say plain assert crashes the program entirely
  - _judge:_ The response says Python assert failures 'are caught by the task runner (they won't crash the process)' — explicitly stating the program does NOT crash entirely.

---

<a id="scenario-6-2"></a>
### ✅ Scenario 6.2 — Four Schema Styles (6/6 criteria · Strong Pass)

**Category:** 6 — Knowledge & Troubleshooting  
**Source of Truth:** `golden_tests/test_cookbook_examples.py` (all `test_extract_*` tests); Skill file §4  

<details><summary>Prompt</summary>

```
What are the different ways to get structured output from an LLM in kaggle-benchmarks, and when should I use each?
```

</details>

<details><summary>Agent response</summary>

````
## Structured Output in kaggle-benchmarks

There are four schema styles, each suited to different situations.

---

### Style 1: Primitive Types

ʼʼʼpython
count = llm.prompt("How many letters in 'hello'?", schema=int)
is_blue = llm.prompt("Is the sky blue?", schema=bool)
summary = llm.prompt("Summarize in one sentence.", schema=str)
ʼʼʼ

**Use when:** You need a single scalar value. Simplest possible case — no class definitions needed.

---

### Style 2: Inline Dict Schema

ʼʼʼpython
result = llm.prompt(
    "9.9 - 9.11 = ?",
    schema={"answer": bool, "explanation": str},
)
print(result.answer, result.explanation)
ʼʼʼ

**Use when:** You need a few fields quickly and don't want to define a class. Good for prototyping or one-off tasks.

---

### Style 3: Dataclass

ʼʼʼpython
from dataclasses import dataclass

@dataclass
class Sentiment:
    label: str
    score: float

result = llm.prompt("Analyze: 'I love this!'", schema=Sentiment)
print(result.label, result.score)
ʼʼʼ

**Use when:** You have a well-defined, reusable structure — especially if it involves enums, nested types, or frozen immutability. Also the right choice when you're passing the schema to multiple prompts or reusing it across tasks.

---

### Style 4: Pydantic Model

ʼʼʼpython
import pydantic

class Review(pydantic.BaseModel):
    sentiment: str = pydantic.Field(description="positive, negative, or neutral")
    score: float = pydantic.Field(description="confidence score 0-1")
    key_phrases: list[str] = pydantic.Field(description="notable phrases from the text")

result = llm.prompt("Analyze: 'Great movie!'", schema=Review)
ʼʼʼ

**Use when:** Field descriptions would meaningfully improve extraction accuracy — the `description=` hint in `Field()` is passed to the model and helps it understand what each field should contain. Also useful when you want runtime validation rules (e.g., `ge=0, le=1` for score bounds).

---

### Decision Summary

| Situation | Use |
|-----------|-----|
| Single value (number, yes/no, text) | Primitive (`int`, `bool`, `str`) |
| Quick 2-3 field result, no reuse | Inline dict |
| Complex/nested type, reused across tasks | Dataclass |
| Fields need descriptions to guide the model | Pydantic with `Field(description=...)` |

The key differentiator between dataclass and Pydantic is that `Field(description=...)` sends semantic hints to the LLM about what each field means — worth reaching for when field names alone are ambiguous (e.g., `score` could mean many things, but `Field(description="confidence score 0-1")` is unambiguous).
````

</details>

**Criteria:**

- ✅ Lists 4 approaches: dataclass, dict schema, Pydantic, primitive types
  - _judge:_ All four approaches are clearly listed and demonstrated: dataclass (Style 3), dict schema (Style 2), Pydantic (Style 4), and primitive types (Style 1).
- ✅ Explains **dataclass** — preferred for complex types with multiple fields
  - _judge:_ The response explicitly explains dataclass as preferred for 'complex/nested type, reused across tasks' with multiple fields (label, score), includes @dataclass decorator, and describes its use case clearly.
- ✅ Explains **dict schema** `{"key": type}` — quick prototyping, simple key-value
  - _judge:_ Response explicitly explains dict schema `{"key": type}` (Style 2: Inline Dict Schema) with example `{"answer": bool, "explanation": str}` and describes it as good for prototyping/one-off tasks.
- ✅ Explains **Pydantic** — when you need validation rules or `Field()` descriptions
  - _judge:_ Response explicitly explains Pydantic with `Field()` descriptions, showing `pydantic.Field(description=...)` usage and explaining when to use it for validation rules and field descriptions.
- ✅ Explains **primitive** (`int`, `bool`, `str`) — when you need a single value
  - _judge:_ Response explicitly covers int, bool, str as primitive types with code examples (schema=int, schema=bool, schema=str) and explains they're used when you need a single scalar value.
- ✅ Shows at least one example for each
  - _judge:_ All four schema styles shown: primitive types (int/bool/str), inline dict, dataclass, and Pydantic BaseModel, each with a working code example.

---

<a id="scenario-6-3"></a>
### ✅ Scenario 6.3 — Missing Return Annotation Bug (4/4 criteria · Strong Pass)

**Category:** 6 — Knowledge & Troubleshooting  
**Source of Truth:** `src/kaggle_benchmarks/results.py` lines 63–111; `SKILL.md` Common Mistakes  

<details><summary>Prompt</summary>

```
I wrote this task but it's not showing scores on the leaderboard:
ʼʼʼpython
@kbench.task()
def accuracy(llm):
    return 0.85
ʼʼʼ
What's wrong?
```

</details>

<details><summary>Agent response</summary>

````
The problem is the missing return type annotation. When a task has no annotation (or `-> None`), the library treats it as a **PassFail** task and ignores the returned value entirely.

**Fix — add `-> float`:**

ʼʼʼpython
@kbench.task()
def accuracy(llm) -> float:
    return 0.85
ʼʼʼ

The return type annotation is what tells the library how to interpret and store the result:

| Annotation | Result Type | Use when... |
|------------|-------------|-------------|
| *(none)* or `-> None` | PassFail | You only use assertions, no return value |
| `-> float` | Score | Returning a decimal metric (accuracy, F1, etc.) |
| `-> bool` | Boolean | True = pass, False = fail |
| `-> int` | Numerical | Integer counts or scores |
| `-> tuple[int, int]` | PassCount | e.g. `(8, 10)` — 8 out of 10 correct |
| `-> tuple[float, float]` | MetricWithCI | Value ± confidence interval |
| `-> dict` | Dictionary | Multiple metrics at once |

Without the annotation, your `return 0.85` is silently discarded — the task runs, sees no exceptions, and records a pass, but no numeric score reaches the leaderboard.
````

</details>

**Criteria:**

- ✅ Identifies the missing `-> float` return type annotation
  - _judge:_ The response explicitly identifies the missing `-> float` return type annotation and provides a code fix with `-> float` in the function signature.
- ✅ Explains the return type is needed for the library to infer the result type
  - _judge:_ The response explicitly states 'The return type annotation is what tells the library how to interpret and store the result' and explains that without it, the returned value is ignored/discarded.
- ✅ Shows the fix: `def accuracy(llm) -> float:`
  - _judge:_ The extracted code block contains exactly `def accuracy(llm) -> float:` as required by the criterion.
- ✅ Explains that without annotation, it defaults to PassFail (ignores the returned value)
  - _judge:_ The response explicitly states that without annotation (or -> None), the library treats the task as PassFail and ignores/discards the returned value, which matches the criterion.

---

<a id="scenario-6-4"></a>
### ⚠️ Scenario 6.4 — Judge Returns None (3/4 criteria · Partial Pass)

**Category:** 6 — Knowledge & Troubleshooting  
**Source of Truth:** Skill file §5 "LLM-as-Judge"; `documentation/examples/assess_with_a_judge.py`  

<details><summary>Prompt</summary>

```
My task crashes with `AttributeError: 'NoneType' object has no attribute 'results'` when using the judge. What's happening?
```

</details>

<details><summary>Agent response</summary>

````
You're not checking whether `assess_response_with_judge` returned `None` before accessing `.results`. The function returns `None` when the judge LLM fails to respond (timeout, API error, malformed output, etc.), and accessing `.results` on `None` causes the crash.

## The Fix

ʼʼʼpython
# ❌ Crashes if judge fails
assessment = kbench.assertions.assess_response_with_judge(
    criteria=["The answer is concise.", "The answer is accurate."],
    response_text=response,
    judge_llm=kbench.judge_llm,
)
for result in assessment.results:   # AttributeError if assessment is None
    kbench.assertions.assert_true(result.passed, expectation=result.reason)

# ✅ Always guard against None
assessment = kbench.assertions.assess_response_with_judge(
    criteria=["The answer is concise.", "The answer is accurate."],
    response_text=response,
    judge_llm=kbench.judge_llm,
)
if assessment is None:
    kbench.assertions.assert_fail("Judge failed to respond.")
else:
    for result in assessment.results:
        kbench.assertions.assert_true(result.passed, expectation=result.reason)
ʼʼʼ

## Why `assert_fail` Instead of Raising an Exception

Using `kbench.assertions.assert_fail(...)` is the right move here rather than `raise` because:

- It records the failure in the benchmark's assertion log (visible in results).
- Execution continues cleanly — other assertions or cleanup code still runs.
- A bare `raise` would surface as an unexpected crash rather than a tracked failure.

## Isolating the Judge Conversation

There's a second common problem alongside this one: if you call the judge inside the same conversation as the task, the judge sees the full chat history, which can skew its evaluation. Use `chats.new()` to isolate it:

ʼʼʼpython
@kbench.task()
def story_quality(llm):
    story = llm.prompt("Write a one-paragraph story about a cat detective.")

    with kbench.chats.new("judge"):   # Clean slate for the judge
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
                expectation=f"'{result.criterion}': {result.reason}",
            )
ʼʼʼ

The `None` check and the isolated chat together give you a robust judge pattern.
````

</details>

**Criteria:**

- ✅ Explains `assess_response_with_judge()` can return `None` when the judge fails
  - _judge:_ The response explicitly explains that assess_response_with_judge() returns None when the judge LLM fails (timeout, API error, malformed output), and demonstrates handling this case.
- ✅ Shows the fix: check `if assessment is None:` before accessing `.results`
  - _judge:_ The response clearly shows `if assessment is None:` check before accessing `.results` in both code blocks.
- ❌ Shows using `kbench.assertions.assert_fail("Judge failed")` as the fallback
  - _judge:_ The code uses `assert_fail("Judge failed to respond.")` not `assert_fail("Judge failed")` as required by the criterion.
- ✅ Does NOT claim the function always returns a valid object
  - _judge:_ The response explicitly states the function returns None on failure and shows guards for `if assessment is None`, never claiming it always returns a valid object.

---

<a id="scenario-6-5"></a>
### ✅ Scenario 6.5 — chats.new vs chats.fork (4/4 criteria · Strong Pass)

**Category:** 6 — Knowledge & Troubleshooting  
**Source of Truth:** `src/kaggle_benchmarks/chats.py` lines 139 and 158  

<details><summary>Prompt</summary>

```
What's the difference between `kbench.chats.new()` and `kbench.chats.fork()`?
```

</details>

<details><summary>Agent response</summary>

````
Both create a new conversation context, but they differ in what history they start with:

## `kbench.chats.new()` — Clean Slate

Starts a **completely empty** conversation (no inherited history):

ʼʼʼpython
llm.prompt("My name is Alice.")
llm.prompt("I work on NLP projects.")

with kbench.chats.new("evaluation"):
    # No history here — the model doesn't know about "Alice" or "NLP"
    response = llm.prompt("What's my name?")  # Model has no idea
ʼʼʼ

**Common use cases:**
- Isolating a judge LLM so it doesn't see prior task conversation
- Starting a fresh agent interaction mid-task
- Adding a `system_instructions=` prompt for a specific section

ʼʼʼpython
with kbench.chats.new("judge", system_instructions="You are a strict evaluator."):
    score = judge_llm.prompt(f"Rate this response: {response}", schema=int)
ʼʼʼ

---

## `kbench.chats.fork()` — Copy of Current History

Starts with a **snapshot of the current conversation**, but any new messages added inside the fork **don't affect the original**:

ʼʼʼpython
llm.prompt("My name is Alice.")
llm.prompt("I work on NLP projects.")

with kbench.chats.fork("hypothesis"):
    # Full history inherited — model knows "Alice" + "NLP"
    response = llm.prompt("Suggest a research topic for me.")
    # New messages here stay in this branch only

# Back to original — still only has the two original messages
llm.prompt("What's my name?")  # Still remembers "Alice"
ʼʼʼ

**Common use cases:**
- Exploring alternative continuations of a conversation without polluting the main thread
- Running parallel hypothetical scenarios from a shared starting point

---

## Quick Decision Guide

| Need | Use |
|------|-----|
| Judge with no context leakage | `chats.new()` |
| System prompt for a section | `chats.new(system_instructions=...)` |
| Branch from current context | `chats.fork()` |
| Multiple parallel what-ifs | Multiple `chats.fork()` blocks |
````

</details>

**Criteria:**

- ✅ Explains `chats.new()` creates an EMPTY conversation (clean slate)
  - _judge:_ The response explicitly states chats.new() 'Starts a completely empty conversation (no inherited history)' and demonstrates this with code showing no prior history is accessible.
- ✅ Explains `chats.fork()` COPIES the current conversation history into a new one
  - _judge:_ The response explicitly states chats.fork() 'Starts with a snapshot of the current conversation' (copies history), and the code example shows inherited history inside the fork block.
- ✅ Use case: `new()` for judges or fresh contexts; `fork()` for branching dialogue
  - _judge:_ Response clearly documents new() for judges/fresh contexts and fork() for branching dialogue, with code examples matching both use cases exactly as required by the criterion.
- ✅ Mentions `chats.new()` accepts `system_instructions=` parameter
  - _judge:_ The response explicitly mentions `system_instructions=` as a parameter for `chats.new()` both in prose and in a code example: `kbench.chats.new("judge", system_instructions="You are a strict evaluator.")`

---

<a id="scenario-6-6"></a>
### ❌ Scenario 6.6 — Sub-Task Cluttering Leaderboard (1/3 criteria · Fail)

**Category:** 6 — Knowledge & Troubleshooting  
**Source of Truth:** `documentation/examples/dataset_evaluation.py` line 38; `documentation/examples/simple_task.py` line 27  

<details><summary>Prompt</summary>

```
I have a helper task used inside my main task, but it's cluttering the leaderboard. How do I hide it?
```

</details>

<details><summary>Agent response</summary>

````
Set `store_task=False` on the helper task's decorator:

ʼʼʼpython
@kbench.task(name="single_qa", store_task=False)
def single_qa(llm, question, answer) -> bool:
    response = llm.prompt(question)
    return answer.lower() in response.lower()

@kbench.task(name="full_eval")
def full_eval(llm) -> float:
    runs = single_qa.evaluate(llm=[llm], evaluation_data=df)
    return float(runs.as_dataframe().result.mean())
ʼʼʼ

`store_task=False` prevents the sub-task from being registered or appearing on the leaderboard, while `full_eval` (with the default `store_task=True`) remains visible.
````

</details>

**Criteria:**

- ✅ Recommends `store_task=False` in the sub-task's `@kbench.task()` decorator
  - _judge:_ The agent's response explicitly recommends `store_task=False` in the `@kbench.task()` decorator for the sub-task, which matches the criterion exactly.
- ❌ Shows: `@kbench.task(name="helper", store_task=False)`
  - _judge:_ Agent uses `name="single_qa"` not `name="helper"` as required by the criterion which specifically quotes `@kbench.task(name="helper", store_task=False)`.
- ❌ May also mention `store_run=False`
  - _judge:_ The response mentions `store_task=False` but not `store_run=False`. The criterion says 'May also mention `store_run=False`', implying the response should mention it as an alternative or additional option.

---

<a id="scenario-6-7"></a>
### ✅ Scenario 6.7 — evaluate() Parameter: llm Must Be a List (3/3 criteria · Strong Pass)

**Category:** 6 — Knowledge & Troubleshooting  
**Source of Truth:** `documentation/examples/dataset_evaluation.py` line 62; `documentation/examples/quick_start_examples.py` line 270  

<details><summary>Prompt</summary>

```
I'm getting an error when running `.evaluate()`. My code is:
ʼʼʼpython
results = my_task.evaluate(llm=kbench.llm, evaluation_data=df)
ʼʼʼ
What's wrong?
```

</details>

<details><summary>Agent response</summary>

````
The `llm` parameter to `.evaluate()` must be a **list**, not a single model:

ʼʼʼpython
# ❌ Wrong — single model
results = my_task.evaluate(llm=kbench.llm, evaluation_data=df)

# ✅ Correct — wrap in a list
results = my_task.evaluate(llm=[kbench.llm], evaluation_data=df)
ʼʼʼ

This is consistent with the multi-model comparison pattern, where you can pass multiple models at once:

ʼʼʼpython
results = my_task.evaluate(
    llm=[kbench.llms["google/gemini-2.5-flash"], kbench.llms["meta/llama-3.1-70b"]],
    evaluation_data=df,
)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Identifies that `llm=` must be a **list**: `llm=[kbench.llm]`
  - _judge:_ The response explicitly shows `llm=[kbench.llm]` as the correct usage, identifying that llm= must be a list.
- ✅ Shows the fix: `my_task.evaluate(llm=[kbench.llm], evaluation_data=df)`
  - _judge:_ The fix `my_task.evaluate(llm=[kbench.llm], evaluation_data=df)` appears literally in the extracted Python code blocks.
- ✅ Explains that `.evaluate()` supports multiple models, so the parameter is always a list
  - _judge:_ The response explicitly explains that `.evaluate()` supports multiple models and therefore the `llm` parameter is always a list, with clear examples showing both single and multiple model usage.

---

<a id="scenario-6-8"></a>
### ✅ Scenario 6.8 — Explicit `-> None` Return Type (4/4 criteria · Strong Pass)

**Category:** 6 — Knowledge & Troubleshooting  
**Source of Truth:** `src/kaggle_benchmarks/results.py` line 63 (`class PassFail(Result[type(None) | Unknown])`); Skill file §2 Return Types  

<details><summary>Prompt</summary>

```
Is `-> None` a valid return type annotation for a kaggle-benchmarks task? What result type does it produce?
```

</details>

<details><summary>Agent response</summary>

````
Yes, `-> None` is a valid return type annotation for a kaggle-benchmarks task. According to the skill documentation:

> `-> None` is equivalent to omitting the annotation — both produce **PassFail**.

This is also reflected in the return types table:

| Annotation | Result Type | Meaning |
|------------|-------------|---------|
| (none) or `-> None` | PassFail | Pass if no exceptions, based on assertions |

So a task annotated with `-> None` (or with no annotation at all) will be evaluated based on whether it completes without unhandled exceptions and whether all recorded `kbench.assertions.*` calls passed.
````

</details>

**Criteria:**

- ✅ Confirms `-> None` is valid
  - _judge:_ The agent's response explicitly states '-> None' is valid and equivalent to omitting the annotation, producing PassFail results.
- ✅ Explains it is equivalent to omitting the return annotation entirely
  - _judge:_ The agent explicitly states '-> None is equivalent to omitting the annotation — both produce PassFail', directly explaining the equivalence.
- ✅ Both produce `PassFail` result type — pass is determined by assertions, not a return value
  - _judge:_ The agent explicitly states both (none) and -> None produce PassFail result type, matching the criterion requirement.
- ✅ Does NOT say `-> None` causes an error or is unsupported
  - _judge:_ The response explicitly states `-> None` is valid and equivalent to omitting the annotation, producing PassFail. It does not claim `-> None` causes an error or is unsupported.

---

<a id="scenario-6-9"></a>
### ⚠️ Scenario 6.9 — Temperature Parameter (3/4 criteria · Partial Pass)

**Category:** 6 — Knowledge & Troubleshooting  
**Source of Truth:** `src/kaggle_benchmarks/actors/llms.py` line 180 (`temperature: float = 0`); Skill file §4  

<details><summary>Prompt</summary>

```
How do I make the LLM give more creative/varied responses in a kaggle-benchmarks task?
```

</details>

<details><summary>Agent response</summary>

````
Use the `temperature` parameter in `llm.prompt()`. The default is `0` (deterministic), and higher values increase creativity and variation:

ʼʼʼpython
import kaggle_benchmarks as kbench

@kbench.task(name="creative_writing")
def creative_writing(llm):
    # Default temperature=0 → deterministic, good for factual tasks
    factual = llm.prompt("What is the capital of France?")

    # Higher temperature → more creative/varied, good for generative tasks
    story = llm.prompt(
        "Write a short story about a robot learning to paint.",
        temperature=0.7,
    )
    kbench.assertions.assert_not_empty(story)

creative_writing.run(kbench.llm)
ʼʼʼ

### Guidelines for choosing temperature

| Task type | Recommended value |
|-----------|------------------|
| Factual Q&A, math, code | `0` (default) |
| Summarization, extraction | `0`–`0.3` |
| Creative writing, brainstorming | `0.5`–`0.8` |
| Maximum variation/exploration | `0.9`–`1.0` |

### Practical example — comparing outputs at different temperatures

ʼʼʼpython
@kbench.task(name="temperature_comparison")
def temperature_comparison(llm) -> dict:
    prompt = "Suggest a unique name for a coffee shop."

    conservative = llm.prompt(prompt, temperature=0.0)
    creative     = llm.prompt(prompt, temperature=0.8)

    return {
        "conservative": conservative,
        "creative": creative,
    }

temperature_comparison.run(kbench.llm)
ʼʼʼ

One other lever worth knowing: if you also want the model to *think harder* (not just be more random), combine `temperature` with `reasoning`:

ʼʼʼpython
story = llm.prompt(
    "Write a poem with an unexpected twist ending.",
    temperature=0.7,
    reasoning="medium",
)
ʼʼʼ

`temperature` controls *variety*; `reasoning` controls *depth of thought* — they're independent and can be used together.
````

</details>

**Criteria:**

- ✅ Recommends using `temperature=` parameter in `llm.prompt()`
  - _judge:_ The agent explicitly recommends and demonstrates `temperature=` parameter in `llm.prompt()` calls multiple times (e.g., `temperature=0.7`, `temperature=0.0`, `temperature=0.8`).
- ✅ Notes default temperature is `0` (deterministic)
  - _judge:_ Response explicitly states 'The default is `0` (deterministic)' and code comments confirm '# Default temperature=0 → deterministic'.
- ❌ Shows example: `llm.prompt("Write a creative story", temperature=0.7)`
  - _judge:_ Judge JSON parse error: Extra data: line 3 column 1 (char 820); raw='{"passed": true, "reason": "The criterion requires showing `llm.prompt(\\"Write a creative story\\", temperature=0.7)`. The response shows `llm.prompt(\\"Write a short story about a robot learning to pai'
- ✅ Does NOT suggest configuring temperature through a separate config object
  - _judge:_ The response configures temperature directly as a parameter in llm.prompt() calls, never suggesting a separate config object.

---

<a id="scenario-6-10"></a>
### ✅ Scenario 6.10 — File Structure and Cell Markers (6/6 criteria · Strong Pass)

**Category:** 6 — Knowledge & Troubleshooting  
**Source of Truth:** `documentation/examples/potemkin_understanding.py` lines 21–27; Skill file §1 "File Structure: Cell Markers"  

<details><summary>Prompt</summary>

```
I need to write a kaggle-benchmarks file with two tasks. Should I use a Jupyter notebook (.ipynb) or a Python file (.py)? And how should I structure the file if I need to install a dependency like `pronouncing`?
```

</details>

<details><summary>Agent response</summary>

````
Based on the skill documentation, you should use a **Python file (`.py`)** — not a Jupyter notebook. Benchmark files are Python scripts that use `# %%` cell markers to create logical sections, making them runnable both as standalone scripts and as interactive notebooks in VS Code or Jupyter.

Here's how to structure a file with two tasks and a `pronouncing` dependency:

ʼʼʼpython
# %%
# !pip install -q pronouncing   # Uncomment on Kaggle
import pronouncing
import kaggle_benchmarks as kbench

# %%
@kbench.task(name="task_one")
def task_one(llm):
    response = llm.prompt("What rhymes with 'cat'?")
    kbench.assertions.assert_contains_regex(r"(?i)bat|hat|mat|rat|sat", response)

task_one.run(kbench.llm)

# %%
@kbench.task(name="task_two")
def task_two(llm) -> float:
    words = pronouncing.rhymes("moon")
    question = f"Which of these rhymes with 'moon': {', '.join(words[:5])}?"
    response = llm.prompt(question)
    correct = any(w in response.lower() for w in words[:5])
    return float(correct)

task_two.run(kbench.llm)
ʼʼʼ

A few things worth noting about this structure:

**The `# !pip install` line is commented out.** This keeps the file runnable locally without errors, while still being easy to uncomment on Kaggle where the package might not be pre-installed.

**Each task is in its own `# %%` cell, with its `.run()` call immediately after.** The `.run()` calls are at the top level — never inside an `if __name__ == "__main__":` guard. Benchmark files are always executed directly, never imported.

**`task_two` has a `-> float` return annotation** because it returns a value. Omitting that annotation when a task returns something is one of the most common mistakes — without it, the framework won't know how to interpret the result.
````

</details>

**Criteria:**

- ✅ Recommends a `.py` file (not `.ipynb`)
  - _judge:_ Agent explicitly recommends a '.py' file (not '.ipynb') and provides a .py code example, clearly stating 'Python file (.py) — not a Jupyter notebook'.
- ✅ Uses `# %%` cell markers to separate logical sections
  - _judge:_ The code contains multiple `# %%` cell markers separating logical sections (imports, task_one, task_two).
- ✅ Shows imports in one cell, each task in its own cell
  - _judge:_ Imports are in their own cell (first # %% block), and each task (task_one, task_two) is in its own separate # %% cell.
- ✅ For `!pip install`, uses commented form: `# !pip install -q pronouncing` (preferred for local compatibility)
  - _judge:_ The code contains `# !pip install -q pronouncing` (commented form), matching the required pattern exactly.
- ✅ Explains that uncommented `!pip` magics work on Kaggle but NOT when running as a standalone Python file locally
  - _judge:_ The response explicitly states the commented `!pip` line keeps the file runnable locally without errors, and it can be uncommented on Kaggle — directly addressing that uncommented pip magics don't work locally.
- ✅ Does NOT generate a notebook file
  - _judge:_ The agent generated a .py Python file with # %% cell markers, not a notebook file (.ipynb). No notebook was generated.

---

<a id="scenario-6-11"></a>
### ✅ Scenario 6.11 — No `if __name__ == "__main__":` Guard (6/6 criteria · Strong Pass)

**Category:** 6 — Knowledge & Troubleshooting  
**Source of Truth:** `SKILL.md` Key Rules; §1 "File Structure: Cell Markers"  

<details><summary>Prompt</summary>

```
Write a benchmark file with two tasks. The first task asks the LLM "What is Python?" and checks the response mentions "programming". The second task asks "What is 2+2?" and uses `schema=int` to get the answer. Run both tasks at the end of the file.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench


# %%
@kbench.task(name="what_is_python")
def what_is_python(llm):
    response = llm.prompt("What is Python?")
    kbench.assertions.assert_contains_regex(
        r"(?i)programming",
        response,
        expectation="Response should mention programming.",
    )


# %%
@kbench.task(name="simple_addition")
def simple_addition(llm) -> int:
    answer = llm.prompt("What is 2+2?", schema=int)
    kbench.assertions.assert_equal(4, answer, expectation="2+2 should equal 4.")
    return answer


# %%
what_is_python.run(kbench.llm)

# %%
simple_addition.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Both tasks are defined with `@kbench.task()` decorator
  - _judge:_ Both functions use @kbench.task() decorator: @kbench.task(name='what_is_python') and @kbench.task(name='simple_addition')
- ✅ Both `.run(kbench.llm)` calls are placed at the **module top level** (not inside any guard)
  - _judge:_ Both `.run(kbench.llm)` calls are at module top level, not inside any `if __name__ == '__main__':` guard or any other conditional/function block.
- ✅ Does **NOT** wrap `.run()` or `.evaluate()` inside `if __name__ == "__main__":`
  - _judge:_ `.run()` calls appear at module level without any `if __name__ == '__main__':` guard.
- ✅ Uses `# %%` cell markers to separate sections
  - _judge:_ The code contains multiple `# %%` cell markers separating sections throughout the file.
- ✅ Uses kbench assertions (not plain `assert`)
  - _judge:_ Code uses kbench.assertions.assert_contains_regex and kbench.assertions.assert_equal, not plain assert statements.
- ✅ Second task has `-> int` return type and uses `schema=int`
  - _judge:_ Second task `simple_addition` has `-> int` return type annotation and uses `schema=int` in the llm.prompt call.

---

<a id="scenario-7-1"></a>
### ⚠️ Scenario 7.1 — Sentiment Analysis Pipeline (Structured + Dataset Eval) (7/8 criteria · Partial Pass)

**Category:** 7 — Generalization — Domain-Specific Tasks  
**Source of Truth:** Combines Pattern H (dataset eval) + Style 1 (dataclass schema) + sub-task composition from Skill file §3 and §4  

<details><summary>Prompt</summary>

```
Design a kaggle-benchmarks evaluation pipeline for sentiment analysis. I have 5 product reviews. For each review, the LLM should extract: `sentiment` (positive/negative/neutral), `confidence` (0-1 float), and `key_reason` (1 sentence). Evaluate all reviews in parallel and report the average confidence for "positive" reviews only.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
# %%
import kaggle_benchmarks as kbench
import pandas as pd
from dataclasses import dataclass

# %%
@dataclass
class SentimentResult:
    sentiment: str      # "positive", "negative", or "neutral"
    confidence: float   # 0.0 to 1.0
    key_reason: str     # One sentence explaining the sentiment

# %%
# Sub-task: analyze a single review (store_task=False — called inside the main task)
@kbench.task(name="analyze_review", store_task=False)
def analyze_review(llm, review: str) -> dict:
    result = llm.prompt(
        f"""Analyze the sentiment of this product review.

Review:
\"\"\"{review}\"\"\"

Instructions:
- sentiment: classify as exactly "positive", "negative", or "neutral"
- confidence: your confidence in the classification, from 0.0 to 1.0
- key_reason: one sentence explaining the primary reason for your classification
""",
        schema=SentimentResult,
    )

    # Validate sentiment is one of the expected values
    kbench.assertions.assert_in(
        result.sentiment,
        ["positive", "negative", "neutral"],
        expectation="Sentiment must be 'positive', 'negative', or 'neutral'.",
    )

    # Validate confidence is in [0, 1]
    kbench.assertions.assert_true(
        0.0 <= result.confidence <= 1.0,
        expectation=f"Confidence must be between 0 and 1, got {result.confidence}.",
    )

    # Validate key_reason is non-empty
    kbench.assertions.assert_not_empty(
        result.key_reason.strip(),
        expectation="key_reason must not be empty.",
    )

    return {
        "sentiment": result.sentiment,
        "confidence": result.confidence,
        "key_reason": result.key_reason,
    }

# %%
# Main task: evaluate all reviews and report average confidence for positives
@kbench.task(name="sentiment_pipeline")
def sentiment_pipeline(llm) -> float:
    reviews_df = pd.DataFrame([
        {"review": "This blender is absolutely incredible — smoothest operation I've ever seen and the build quality is top-notch!"},
        {"review": "Arrived two weeks late, the packaging was crushed, and half the parts were missing. Total disaster."},
        {"review": "It does what it says. Nothing special, nothing bad. Exactly as described."},
        {"review": "I bought this for my mom and she loves it! Super easy to use and the colors are beautiful. Five stars!"},
        {"review": "Battery life is disappointing — barely lasts 3 hours. The camera is decent though, so it balances out somewhat."},
    ])

    # Evaluate all reviews in parallel (n_jobs=5 for one worker per review)
    with kbench.client.enable_cache():
        results = analyze_review.evaluate(
            llm=[llm],
            evaluation_data=reviews_df,
            n_jobs=5,
            timeout=60,
            max_attempts=2,
            remove_run_files=True,
        )

    scores_df = results.as_dataframe()

    # Assert all sub-tasks passed
    kbench.assertions.assert_true(
        scores_df["passed"].all(),
        expectation="All review analyses should pass validation.",
    )

    # Extract structured results from the "result" column (each row is a dict)
    sentiments = scores_df["result"].tolist()

    positive_confidences = [
        r["confidence"]
        for r in sentiments
        if r is not None and r.get("sentiment") == "positive"
    ]

    # Assert at least one positive review was found in this dataset
    kbench.assertions.assert_not_empty(
        positive_confidences,
        expectation="At least one review should be classified as positive.",
    )

    avg_confidence = sum(positive_confidences) / len(positive_confidences)

    kbench.assertions.assert_true(
        0.0 <= avg_confidence <= 1.0,
        expectation=f"Average confidence must be in [0, 1], got {avg_confidence}.",
    )

    print(f"\n{'='*55}")
    print(f"  Sentiment Analysis Results")
    print(f"{'='*55}")
    for i, (row, result) in enumerate(
        zip(reviews_df.itertuples(), sentiments), start=1
    ):
        if result is None:
            print(f"  Review {i}: [FAILED]")
            continue
        icon = {"positive": "✅", "negative": "❌", "neutral": "➖"}.get(
            result["sentiment"], "?"
        )
        print(f"  Review {i}: {icon} {result['sentiment'].upper()} "
              f"(conf: {result['confidence']:.2f})")
        print(f"    Reason: {result['key_reason']}")
    print(f"{'='*55}")
    print(f"  Positive reviews found : {len(positive_confidences)}")
    print(f"  Avg confidence (pos)   : {avg_confidence:.4f}")
    print(f"{'='*55}\n")

    return avg_confidence

# %%
sentiment_pipeline.run(kbench.llm)
ʼʼʼ

## How it fits together

ʼʼʼ
sentiment_pipeline.run(kbench.llm)
        │
        ├── analyze_review.evaluate(llm=[llm], evaluation_data=reviews_df, n_jobs=5)
        │       │
        │       ├── analyze_review(llm, review=reviews_df[0])  ─┐
        │       ├── analyze_review(llm, review=reviews_df[1])   │  parallel
        │       ├── analyze_review(llm, review=reviews_df[2])   │  workers
        │       ├── analyze_review(llm, review=reviews_df[3])   │
        │       └── analyze_review(llm, review=reviews_df[4])  ─┘
        │               │
        │               └── llm.prompt(..., schema=SentimentResult)
        │                       → assert_in(sentiment, [...])
        │                       → assert_true(0 ≤ confidence ≤ 1)
        │                       → assert_not_empty(key_reason)
        │                       → return dict
        │
        ├── assert all sub-tasks passed
        ├── filter rows where sentiment == "positive"
        ├── compute average confidence
        └── return float  →  recorded as the task score
ʼʼʼ

### Key design decisions explained

| Decision | Reason |
|----------|--------|
| `store_task=False` on `analyze_review` | It's a helper called inside the pipeline — no need for a separate leaderboard entry |
| `schema=SentimentResult` (dataclass) | Gives typed fields; cleaner than a raw dict schema for three heterogeneous types |
| `n_jobs=5` | One parallel worker per review — maximum parallelism for this dataset size |
| `result` column filtered with `r is not None` guard | `.evaluate()` can produce `None` results on timeout/failure; this prevents a `KeyError` |
| `-> float` return annotation | Tells the framework this task produces a numerical score for the leaderboard |
| `enable_cache()` | Avoids re-calling the LLM if you re-run the cell during development |
| Three targeted assertions per sub-task | Validate each extracted field independently so you know *which* field failed if something goes wrong |
````

</details>

**Criteria:**

- ✅ Defines a dataclass or Pydantic model with `sentiment`, `confidence`, `key_reason`
  - _judge:_ The dataclass SentimentResult defines all three required fields: `sentiment`, `confidence`, and `key_reason`.
- ✅ Defines a per-row sub-task with `store_task=False` and `-> dict` return
  - _judge:_ The code defines `analyze_review` with `@kbench.task(name="analyze_review", store_task=False)` and return type `-> dict`, satisfying both requirements.
- ❌ Uses `llm.prompt(review, schema=SentimentResult)` inside the sub-task
  - _judge:_ The code calls `llm.prompt(f"""...{review}...""", schema=SentimentResult)` with a formatted string, not `llm.prompt(review, schema=SentimentResult)` with `review` as the direct positional argument.
- ✅ Creates a `pd.DataFrame` of reviews
  - _judge:_ The code contains `reviews_df = pd.DataFrame([...])` which creates a pd.DataFrame of reviews.
- ✅ Main task uses `.evaluate(llm=[llm], evaluation_data=df, n_jobs=...)`
  - _judge:_ sentiment_pipeline calls analyze_review.evaluate(llm=[llm], evaluation_data=reviews_df, n_jobs=5, ...) which matches the required pattern.
- ✅ Filters results for positive sentiment and computes mean confidence
  - _judge:_ Code filters for sentiment == 'positive' in positive_confidences list comprehension and computes mean via sum/len, satisfying both parts of the criterion.
- ✅ Returns `-> float` from the main task
  - _judge:_ The main task function `sentiment_pipeline` is annotated with `-> float` on line `def sentiment_pipeline(llm) -> float:` and returns `avg_confidence` which is a float.
- ✅ Uses `# %%` cell markers
  - _judge:_ The code contains multiple `# %%` cell markers (at least 5 occurrences) in the extracted Python code.

---

<a id="scenario-7-2"></a>
### ⚠️ Scenario 7.2 — Code Review Benchmark (System Prompt + Structured + Code Execution) (6/7 criteria · Partial Pass)

**Category:** 7 — Generalization — Domain-Specific Tasks  
**Source of Truth:** Combines Pattern I (code analysis) + dataset eval + code execution from Skill file §3 and §7  

<details><summary>Prompt</summary>

```
Build a benchmark that tests whether an LLM can review buggy Python code. Give the LLM 3 buggy code snippets. For each, the LLM should return a structured response with `has_bug: bool`, `bug_description: str`, and `fixed_code: str`. Verify each fix by running it and checking expected output.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
# %%
import kaggle_benchmarks as kbench
from dataclasses import dataclass


# %%
@dataclass
class CodeReview:
    has_bug: bool
    bug_description: str
    fixed_code: str


# %%
BUGGY_SNIPPETS = [
    {
        "label": "missing_comma",
        "code": """
fruits = ['apple', 'orange' 'banana', 'peach']
print(len(fruits))
""".strip(),
        "expected_output": "4",
        "description": "List with implicit string concatenation due to missing comma",
    },
    {
        "label": "off_by_one",
        "code": """
def sum_first_n(numbers, n):
    total = 0
    for i in range(n + 1):
        total += numbers[i]
    return total

print(sum_first_n([10, 20, 30, 40, 50], 3))
""".strip(),
        "expected_output": "60",
        "description": "Off-by-one error in range causing IndexError or wrong sum",
    },
    {
        "label": "wrong_return",
        "code": """
def celsius_to_fahrenheit(c):
    return c * 9 / 5 + 32

def fahrenheit_to_celsius(f):
    return celsius_to_fahrenheit(f)

print(round(fahrenheit_to_celsius(212), 1))
""".strip(),
        "expected_output": "100.0",
        "description": "fahrenheit_to_celsius incorrectly delegates to celsius_to_fahrenheit",
    },
]


# %%
@kbench.task(name="review_single_snippet", store_task=False)
def review_single_snippet(llm, code: str, label: str, expected_output: str, description: str) -> bool:
    """Ask the LLM to review one buggy snippet and verify its fix runs correctly."""

    review: CodeReview = llm.prompt(
        f"Review the following Python code for bugs.\n\nʼʼʼpython\n{code}\nʼʼʼ\n\n"
        "Return whether it has a bug, describe the bug, and provide corrected code.",
        schema=CodeReview,
    )

    # 1. LLM must recognise there is a bug
    kbench.assertions.assert_true(
        review.has_bug,
        expectation=f"[{label}] LLM should detect the bug: {description}",
    )

    # 2. Bug description must be non-empty
    kbench.assertions.assert_not_empty(
        review.bug_description.strip(),
        expectation=f"[{label}] bug_description should not be empty",
    )

    # 3. Extract and run the fixed code
    fixed_code = kbench.tools.python.extract_code(review.fixed_code)
    if not fixed_code:
        # fixed_code field might already be plain Python without fences
        fixed_code = review.fixed_code.strip()

    result = kbench.tools.python.script_runner.run_code(fixed_code)

    kbench.assertions.assert_empty(
        result.stderr.strip(),
        expectation=f"[{label}] Fixed code should run without errors",
    )
    kbench.assertions.assert_equal(
        expected_output,
        result.stdout.strip(),
        expectation=f"[{label}] Fixed code output should be '{expected_output}'",
    )

    passed = (
        review.has_bug
        and bool(review.bug_description.strip())
        and result.stderr.strip() == ""
        and result.stdout.strip() == expected_output
    )
    return passed


# %%
@kbench.task(name="python_code_review_benchmark")
def python_code_review_benchmark(llm) -> float:
    """
    Benchmark: LLM code review on 3 buggy Python snippets.
    Returns the fraction of snippets correctly reviewed and fixed.
    """
    passes = 0

    for snippet in BUGGY_SNIPPETS:
        run = review_single_snippet.run(
            llm,
            code=snippet["code"],
            label=snippet["label"],
            expected_output=snippet["expected_output"],
            description=snippet["description"],
        )
        if run.passed:
            passes += 1

    score = passes / len(BUGGY_SNIPPETS)

    kbench.assertions.assert_true(
        score > 0.0,
        expectation="LLM should fix at least one of the three snippets correctly",
    )

    return score


# %%
python_code_review_benchmark.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ❌ Uses `kbench.system.send("You are a code reviewer...")` for context
  - _judge:_ The agent's response never calls `kbench.system.send("You are a code reviewer...")`. No system prompt setup is present anywhere in the code.
- ✅ Defines a `@dataclass` with `has_bug`, `bug_description`, `fixed_code`
  - _judge:_ @dataclass class CodeReview defines has_bug: bool, bug_description: str, and fixed_code: str exactly as required.
- ✅ Uses a parameterized task or dataset evaluation over the 3 snippets
  - _judge:_ The code iterates over BUGGY_SNIPPETS list (3 snippets) in a for loop within python_code_review_benchmark, calling review_single_snippet for each, constituting parameterized evaluation over all 3 snippets.
- ✅ For each snippet: `llm.prompt(code, schema=CodeReview)`
  - _judge:_ The code contains `llm.prompt(code, schema=CodeReview)` pattern - specifically `llm.prompt(..., schema=CodeReview)` appears in the review_single_snippet function.
- ✅ Extracts and runs fixed code: `kbench.tools.python.extract_code()` + `run_code()`
  - _judge:_ Both `kbench.tools.python.extract_code()` and `run_code()` (via `kbench.tools.python.script_runner.run_code()`) are explicitly called in the code.
- ✅ Asserts `has_bug == True` AND verifies output of fixed code
  - _judge:_ Code asserts `review.has_bug` via `kbench.assertions.assert_true(review.has_bug, ...)` AND verifies fixed code output with `assert_equal(expected_output, result.stdout.strip(), ...)`
- ✅ Uses kbench assertions throughout (not plain assert)
  - _judge:_ The code uses kbench.assertions.assert_true, assert_not_empty, assert_empty, and assert_equal throughout instead of plain assert statements.

---

<a id="scenario-7-3"></a>
### ❌ Scenario 7.3 — Translation Quality Benchmark (Multi-Model + Judge) (1/8 criteria · Fail)

**Category:** 7 — Generalization — Domain-Specific Tasks  
**Source of Truth:** Combines multi-model comparison (§3) + judge (§5) + dataset eval (§3)  

<details><summary>Prompt</summary>

```
Design a benchmark that compares how well two models translate English to French. Use 3 test sentences. For each sentence, both models translate, and a judge evaluates translation quality. Report per-model average score.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
# %%
import kaggle_benchmarks as kbench
from dataclasses import dataclass

# %%
@dataclass
class TranslationScore:
    score: float
    feedback: str

TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Climate change is one of the most pressing challenges of our time.",
    "She whispered a secret that changed everything.",
]

# %%
@kbench.task(name="score_translation", store_task=False)
def score_translation(judge_llm, original: str, translation: str) -> float:
    """Judge scores a single English→French translation from 0.0 to 1.0."""
    result = judge_llm.prompt(
        f"Rate the following English-to-French translation on a scale from 0.0 to 1.0.\n\n"
        f"Criteria:\n"
        f"- Accuracy: Does it convey the same meaning?\n"
        f"- Fluency: Does it read naturally in French?\n"
        f"- Style: Does it preserve the tone of the original?\n\n"
        f"Original (English): {original}\n"
        f"Translation (French): {translation}\n\n"
        f"Respond with a score between 0.0 and 1.0 and a brief explanation.",
        schema=TranslationScore,
    )
    kbench.assertions.assert_true(
        0.0 <= result.score <= 1.0,
        expectation="Judge score must be between 0.0 and 1.0.",
    )
    return result.score

# %%
@kbench.task(name="translate_en_to_fr", store_task=False)
def translate_en_to_fr(llm, sentence: str) -> str:
    """Translate a single English sentence to French."""
    translation = llm.prompt(
        f"Translate the following English sentence to French. "
        f"Respond with only the French translation, nothing else.\n\n"
        f"Sentence: {sentence}"
    )
    kbench.assertions.assert_not_empty(translation, expectation="Translation must not be empty.")
    return translation

# %%
@kbench.task(name="translation_benchmark")
def translation_benchmark(llm, judge_llm) -> dict:
    """
    Benchmark English→French translation quality.
    Translates 3 sentences and judges each translation. Returns the average score.
    """
    scores = []

    for sentence in TEST_SENTENCES:
        # Translate in an isolated chat so sentences don't influence each other
        with kbench.chats.new(f"translate_{sentence[:20]}"):
            translation_run = translate_en_to_fr.run(llm, sentence=sentence)

        translation = translation_run.result

        # Judge in an isolated chat to avoid cross-contamination
        with kbench.chats.new(f"judge_{sentence[:20]}"):
            score_run = score_translation.run(
                judge_llm,
                original=sentence,
                translation=translation,
            )

        score = score_run.result
        scores.append(score)

        kbench.assertions.assert_true(
            score_run.passed,
            expectation=f"Scoring must succeed for: '{sentence[:40]}...'",
        )

    average_score = sum(scores) / len(scores)

    kbench.assertions.assert_true(
        0.0 <= average_score <= 1.0,
        expectation="Average score must be between 0.0 and 1.0.",
    )

    return {
        "average_score": average_score,
        "sentence_scores": scores,
    }

# %%
# Run benchmark on two models and compare
model_a = kbench.llms["google/gemini-2.5-flash"]
model_b = kbench.llms["meta/llama-3.1-70b"]

run_a = translation_benchmark.run(model_a, kbench.judge_llm)
run_b = translation_benchmark.run(model_b, kbench.judge_llm)

# %%
# Print comparison summary
print("=== English→French Translation Benchmark ===\n")
print(f"{'Model':<35} {'Avg Score':>10}  {'Per-Sentence Scores'}")
print("-" * 75)

for label, run in [("google/gemini-2.5-flash", run_a), ("meta/llama-3.1-70b", run_b)]:
    result = run.result
    avg = result["average_score"]
    per_sentence = [f"{s:.2f}" for s in result["sentence_scores"]]
    print(f"{label:<35} {avg:>10.3f}  {per_sentence}")

print("\nSentences evaluated:")
for i, s in enumerate(TEST_SENTENCES, 1):
    print(f"  {i}. {s}")
ʼʼʼ
````

</details>

**Criteria:**

- ❌ Defines a per-row sub-task that takes `llm, sentence, reference_translation`
  - _judge:_ No sub-task function takes exactly `llm, sentence, reference_translation`. The closest is `translate_en_to_fr(llm, sentence)` which lacks `reference_translation`.
- ❌ Sub-task calls `llm.prompt(f"Translate to French: {sentence}")`
  - _judge:_ The code uses `llm.prompt(f"Translate the following English sentence to French...\n\nSentence: {sentence}")` but the criterion requires exactly `llm.prompt(f"Translate to French: {sentence}")`
- ❌ Sub-task uses `assess_response_with_judge()` with criteria like "translation is accurate"
  - _judge:_ The code never calls `assess_response_with_judge()`. It uses `judge_llm.prompt()` with a schema and manual scoring instead of the required function.
- ❌ OR uses a judge LLM in a `chats.new()` with `schema=int` for scoring
  - _judge:_ The judge uses `judge_llm.prompt(..., schema=TranslationScore)` not `chats.new()` with `schema=int`. The criterion requires `chats.new()` with `schema=int` specifically.
- ❌ Main task calls `.evaluate(llm=[kbench.llm, kbench.judge_llm], evaluation_data=df, ...)`
  - _judge:_ No `.evaluate(llm=[kbench.llm, kbench.judge_llm], evaluation_data=df, ...)` call found; agent uses `.run()` instead.
- ❌ Uses `llm=` as a LIST of both models
  - _judge:_ The code never uses `llm=` as a keyword argument with a list of both models. Instead, it runs the benchmark separately on model_a and model_b without a list.
- ❌ Computes per-model scores from `.as_dataframe()`
  - _judge:_ The code never calls `.as_dataframe()`. It accesses results via `run.result` directly and prints them manually, without using the `.as_dataframe()` method anywhere.
- ✅ Returns `-> dict` or `-> float`
  - _judge:_ Both `-> dict` (translation_benchmark) and `-> float` (score_translation) return type annotations are present in the code.

---

<a id="scenario-7-4"></a>
### ✅ Scenario 7.4 — Reasoning with Verification (reasoning + structured output + tool) (5/5 criteria · Strong Pass)

**Category:** 7 — Generalization — Domain-Specific Tasks  
**Source of Truth:** Combines reasoning param (§4) + structured output + code execution (§7)  

<details><summary>Prompt</summary>

```
Write a task that asks the LLM to solve a math word problem step by step using reasoning mode. Extract the final numeric answer using `schema=float`, then verify the answer by running a Python calculation. Compare the LLM's answer to the computed answer.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench

# %%
PROBLEM = """
A train travels from City A to City B at 80 km/h and returns from City B to City A
at 120 km/h. The total journey (both ways) covers 480 km.

How many hours does the entire round trip take?
"""

# The computed expected answer (480 km round trip: 240 km each way)
# Time A→B: 240 / 80 = 3 hours
# Time B→A: 240 / 120 = 2 hours
# Total: 5 hours
def compute_expected_answer(total_km: float, speed_ab: float, speed_ba: float) -> float:
    one_way_km = total_km / 2
    return (one_way_km / speed_ab) + (one_way_km / speed_ba)


# %%
@kbench.task(name="math_word_problem_reasoning")
def math_word_problem_reasoning(llm) -> dict:
    """
    Ask the LLM to solve a math word problem with step-by-step reasoning,
    extract the numeric answer, then verify against a Python calculation.
    """
    # Step 1: Ask LLM to reason through the problem step by step
    reasoning_prompt = f"""
Solve the following math word problem step by step. Show all your work clearly.

{PROBLEM}

At the end, state your final answer as a single number (in hours).
"""
    llm.prompt(reasoning_prompt, reasoning="high")
    traces = kbench.last_reasoning_traces()

    kbench.assertions.assert_true(
        traces is not None and len(traces) > 0,
        expectation="Model should produce reasoning traces when reasoning='high' is set.",
    )

    # Step 2: Extract the final numeric answer using structured output
    llm_answer = llm.prompt(
        "Based on your reasoning above, what is the final numeric answer in hours? "
        "Reply with the number only.",
        schema=float,
    )

    kbench.assertions.assert_true(
        llm_answer is not None,
        expectation="LLM should return a numeric answer.",
    )

    # Step 3: Compute the expected answer in Python
    expected = compute_expected_answer(
        total_km=480.0,
        speed_ab=80.0,
        speed_ba=120.0,
    )

    # Step 4: Verify with code execution for transparency
    verification_code = """
total_km = 480.0
speed_ab = 80.0
speed_ba = 120.0
one_way = total_km / 2
time_ab = one_way / speed_ab
time_ba = one_way / speed_ba
total_time = time_ab + time_ba
print(total_time)
"""
    result = kbench.tools.python.script_runner.run_code(verification_code)
    kbench.assertions.assert_empty(
        result.stderr.strip(),
        expectation="Verification script should run without errors.",
    )

    computed_from_script = float(result.stdout.strip())
    kbench.assertions.assert_equal(
        expected,
        computed_from_script,
        expectation="Python computation should match expected value of 5.0 hours.",
    )

    # Step 5: Compare LLM answer to computed answer (within a small tolerance)
    tolerance = 0.01
    is_correct = abs(llm_answer - computed_from_script) <= tolerance
    kbench.assertions.assert_true(
        is_correct,
        expectation=(
            f"LLM answer ({llm_answer}) should be within {tolerance} of "
            f"computed answer ({computed_from_script})."
        ),
    )

    return {
        "llm_answer": llm_answer,
        "computed_answer": computed_from_script,
        "is_correct": is_correct,
        "difference": abs(llm_answer - computed_from_script),
        "has_reasoning_traces": traces is not None,
    }


# %%
math_word_problem_reasoning.run(kbench.llm)
ʼʼʼ

### How it works

| Step | What happens |
|------|-------------|
| **1 — Reasoning prompt** | Sends the word problem with `reasoning="high"` so the model thinks step-by-step before answering |
| **2 — Trace check** | Calls `last_reasoning_traces()` and asserts it's non-empty — confirming the model actually reasoned |
| **3 — Extract answer** | Follows up with `schema=float` to get a clean numeric answer from the same conversation |
| **4 — Python verification** | Runs the calculation independently via `script_runner.run_code()` — ground truth from code, not from the model |
| **5 — Comparison** | Checks the LLM's answer against the script output with a `0.01` tolerance to account for floating-point representation |
| **Return dict** | Returns a `-> dict` so the run object captures all intermediate values for inspection |

Key design choices:
- **Multi-turn conversation**: The reasoning step and the extraction step share history (`chats.new` is *not* used here intentionally), so the model's answer naturally follows from its own reasoning.
- **Independent ground truth**: The Python script re-derives the answer from scratch — it doesn't echo any LLM output — making the comparison meaningful.
- **Tolerance comparison** (`abs(...) <= 0.01`): Avoids false failures from `5.0` vs `5.000000001`-style float edge cases.
````

</details>

**Criteria:**

- ✅ Uses `llm.prompt(problem, reasoning="medium", schema=float)` or two separate calls
  - _judge:_ Agent uses two separate calls: llm.prompt(reasoning_prompt, reasoning='high') then llm.prompt(..., schema=float), satisfying the two-separate-calls alternative.
- ✅ Uses `kbench.tools.python.script_runner.run_code()` to compute the ground truth
  - _judge:_ The code explicitly calls `kbench.tools.python.script_runner.run_code(verification_code)` to compute the ground truth answer.
- ✅ Compares LLM answer to computed answer: `assert_equal` or `assert_true(abs(a-b) < 0.01)`
  - _judge:_ Both `assert_equal` (comparing expected to computed_from_script) and `assert_true(abs(llm_answer - computed_from_script) <= tolerance)` are present, satisfying the criterion.
- ✅ Does NOT hardcode the expected answer — actually computes it
  - _judge:_ The expected answer is computed via `compute_expected_answer()` function and verified via `script_runner.run_code()` with arithmetic from inputs. No literal hardcoded '5.0' is used as the ground truth.
- ✅ Optionally accesses reasoning traces via `kbench.last_reasoning_traces()`
  - _judge:_ The code explicitly calls `kbench.last_reasoning_traces()` and assigns result to `traces`, satisfying the criterion.

---

<a id="scenario-7-5"></a>
### ⚠️ Scenario 7.5 — FAQ Chatbot Benchmark (Multi-Turn + fork + Negative Testing) (4/6 criteria · Partial Pass)

**Category:** 7 — Generalization — Domain-Specific Tasks  
**Source of Truth:** Combines system prompt (§6) + `chats.fork()` (§6) + negative assertion (§5)  

<details><summary>Prompt</summary>

```
Design a benchmark for a customer support chatbot. First establish context: "You are a support agent for TechCorp. Products: CloudDB ($99/mo), FastAPI ($49/mo), DataPipe ($149/mo)." Then test: 1) Ask about CloudDB pricing — assert correct price. 2) Fork the conversation and ask about a non-existent product — assert the bot says it doesn't exist. 3) Back in the original, ask "What was the first product I asked about?" — assert it remembers CloudDB.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench

# %%
@kbench.task(name="customer_support_chatbot")
def customer_support_chatbot(llm):
    """
    Tests a customer support chatbot for TechCorp across three scenarios:
    1. Correct pricing recall for a known product (CloudDB)
    2. Graceful handling of an unknown product (in a forked conversation)
    3. Conversation memory — recalling the first product asked about
    """

    # Establish context via a system message
    kbench.system.send(
        "You are a support agent for TechCorp. "
        "Products: CloudDB ($99/mo), FastAPI ($49/mo), DataPipe ($149/mo)."
    )

    # ── Test 1: CloudDB pricing ───────────────────────────────────────────────
    pricing_response = llm.prompt("How much does CloudDB cost per month?")

    kbench.assertions.assert_contains_regex(
        r"\$\s*99",
        pricing_response,
        expectation="Bot should state CloudDB costs $99/mo.",
    )
    kbench.assertions.assert_contains_regex(
        r"(?i)clouddb",
        pricing_response,
        expectation="Bot should mention the product name CloudDB.",
    )

    # ── Test 2: Unknown product — forked so it doesn't pollute original ───────
    with kbench.chats.fork("unknown_product_branch"):
        unknown_response = llm.prompt(
            "What is the pricing for your QuantumCache product?"
        )

        kbench.assertions.assert_not_contains_regex(
            r"\$\s*\d+",
            unknown_response,
            expectation=(
                "Bot should NOT quote a price for a non-existent product."
            ),
        )
        kbench.assertions.assert_contains_regex(
            r"(?i)(not|don.t|doesn.t|no|unavailable|exist|offer|aware|find|listed)",
            unknown_response,
            expectation=(
                "Bot should indicate QuantumCache is unknown / not offered."
            ),
        )

    # ── Test 3: Memory — original conversation still intact after the fork ────
    memory_response = llm.prompt("What was the first product I asked about?")

    kbench.assertions.assert_contains_regex(
        r"(?i)clouddb",
        memory_response,
        expectation=(
            "Bot should remember that CloudDB was the first product asked about."
        ),
    )

# %%
customer_support_chatbot.run(kbench.llm)
ʼʼʼ

## How it works

### Conversation structure

ʼʼʼ
system: "You are a support agent for TechCorp…"
│
├─ user: "How much does CloudDB cost?"          ← Test 1 (original chat)
│  └─ assert: response contains "$99"
│
├─ FORK ─────────────────────────────────────── Test 2 (isolated branch)
│  │  user: "What is the pricing for QuantumCache?"
│  │  └─ assert: no price quoted, denial language present
│  └─ (branch discarded — original is unaffected)
│
└─ user: "What was the first product I asked about?"  ← Test 3 (original)
   └─ assert: response contains "CloudDB"
ʼʼʼ

### Key design decisions

| Decision | Reason |
|----------|--------|
| `kbench.system.send()` before any `llm.prompt()` | Establishes the agent persona once for the whole conversation |
| `chats.fork("unknown_product_branch")` | Creates an isolated copy of the conversation — the hallucination probe can't bleed into the memory test |
| `assert_contains_regex(r"\$\s*99")` | Handles formatting variants like `$99`, `$ 99`, `$99/mo` |
| `assert_not_contains_regex(r"\$\s*\d+")` on the unknown product | Catches any plausible-sounding price hallucination, not just specific values |
| Denial pattern `(not|don.t|doesn.t|…)` | Broad enough to cover many polite refusal phrasings |
| No `store_task=False` | This is a standalone task (not a sub-task), so the default `store_task=True` is correct |
````

</details>

**Criteria:**

- ✅ Uses `kbench.system.send("You are a support agent...")` to set context
  - _judge:_ The code contains `kbench.system.send("You are a support agent...` literally, matching the criterion's required call pattern.
- ❌ First query: `llm.prompt(...)` + `assert_contains_regex(r"99", response)`
  - _judge:_ The assert_contains_regex uses r"\$\s*99" not the exact r"99" pattern required by the criterion.
- ❌ Uses `with kbench.chats.fork("edge_case"):` to branch the conversation
  - _judge:_ Code uses kbench.chats.fork("unknown_product_branch") instead of the required kbench.chats.fork("edge_case").
- ✅ Inside fork: asks about non-existent product + `assert_not_contains_regex` for prices, or `assert_contains_regex` for "not available/don't have"
  - _judge:_ Inside fork block: asks about non-existent product 'QuantumCache' and uses both assert_not_contains_regex for prices (r'\$\s*\d+') and assert_contains_regex for denial language.
- ✅ After fork (back in original): asks follow-up + `assert_contains_regex(r"(?i)clouddb", response)`
  - _judge:_ After the fork block, Test 3 sends a follow-up prompt and calls assert_contains_regex(r"(?i)clouddb", memory_response) on the response, satisfying the criterion exactly.
- ✅ Demonstrates that fork doesn't pollute original conversation
  - _judge:_ The code uses `kbench.chats.fork()` as a context manager, then after the `with` block ends, prompts the original conversation asking about the first product asked, explicitly demonstrating fork isolation.

---

<a id="scenario-7-6"></a>
### ⚠️ Scenario 7.6 — Structured Data Extraction from Unstructured Text (Complex Schema) (6/7 criteria · Partial Pass)

**Category:** 7 — Generalization — Domain-Specific Tasks  
**Source of Truth:** Combines composite Pydantic (like scenario 1.7) + `Field()` descriptions (§4)  

<details><summary>Prompt</summary>

```
Build a benchmark that extracts structured information from a job posting. The posting is: "Senior ML Engineer at DataCo. Remote. Salary: $150k-$200k. Requirements: 5+ years Python, TensorFlow, PhD preferred. Benefits: Health insurance, 401k match, unlimited PTO." Extract using a nested Pydantic model with `title`, `company`, `salary_range: SalaryRange(min, max)`, `requirements: list[str]`, `benefits: list[str]`, `is_remote: bool`. Assert at least 3 requirements and 2 benefits.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
# %%
import kaggle_benchmarks as kbench
import pydantic

# %%
JOB_POSTING = (
    "Senior ML Engineer at DataCo. Remote. Salary: $150k-$200k. "
    "Requirements: 5+ years Python, TensorFlow, PhD preferred. "
    "Benefits: Health insurance, 401k match, unlimited PTO."
)


class SalaryRange(pydantic.BaseModel):
    min: int = pydantic.Field(description="Minimum salary in USD (numeric only, e.g. 150000)")
    max: int = pydantic.Field(description="Maximum salary in USD (numeric only, e.g. 200000)")


class JobPosting(pydantic.BaseModel):
    title: str = pydantic.Field(description="Job title")
    company: str = pydantic.Field(description="Company name")
    salary_range: SalaryRange = pydantic.Field(description="Salary range with min and max in USD")
    requirements: list[str] = pydantic.Field(description="List of job requirements or qualifications")
    benefits: list[str] = pydantic.Field(description="List of employee benefits offered")
    is_remote: bool = pydantic.Field(description="Whether the position is remote")


@kbench.task(name="job_posting_extraction")
def job_posting_extraction(llm) -> dict:
    """Extract structured information from a job posting using a nested Pydantic model."""

    posting = llm.prompt(
        f"Extract all structured information from this job posting:\n\n{JOB_POSTING}",
        schema=JobPosting,
    )

    # --- Title & Company ---
    kbench.assertions.assert_contains_regex(
        r"(?i)senior.*ml.*engineer|ml.*engineer.*senior|senior.*machine.*learning",
        posting.title,
        expectation="Title should reflect 'Senior ML Engineer'.",
    )
    kbench.assertions.assert_contains_regex(
        r"(?i)dataco",
        posting.company,
        expectation="Company should be 'DataCo'.",
    )

    # --- Remote ---
    kbench.assertions.assert_true(
        posting.is_remote,
        expectation="Position should be identified as remote.",
    )

    # --- Salary Range ---
    kbench.assertions.assert_true(
        posting.salary_range.min > 0,
        expectation="Minimum salary should be a positive number.",
    )
    kbench.assertions.assert_true(
        posting.salary_range.max > posting.salary_range.min,
        expectation="Maximum salary should exceed minimum salary.",
    )
    kbench.assertions.assert_true(
        100_000 <= posting.salary_range.min <= 160_000,
        expectation=f"Min salary should be ~$150,000 (got {posting.salary_range.min}).",
    )
    kbench.assertions.assert_true(
        180_000 <= posting.salary_range.max <= 220_000,
        expectation=f"Max salary should be ~$200,000 (got {posting.salary_range.max}).",
    )

    # --- Requirements (at least 3) ---
    kbench.assertions.assert_true(
        len(posting.requirements) >= 3,
        expectation=f"Should extract at least 3 requirements (got {len(posting.requirements)}).",
    )
    requirements_blob = " ".join(posting.requirements).lower()
    kbench.assertions.assert_contains_regex(
        r"python",
        requirements_blob,
        expectation="Requirements should mention Python.",
    )
    kbench.assertions.assert_contains_regex(
        r"tensorflow",
        requirements_blob,
        expectation="Requirements should mention TensorFlow.",
    )
    kbench.assertions.assert_contains_regex(
        r"phd|doctorate|doctoral",
        requirements_blob,
        expectation="Requirements should mention PhD preference.",
    )

    # --- Benefits (at least 2) ---
    kbench.assertions.assert_true(
        len(posting.benefits) >= 2,
        expectation=f"Should extract at least 2 benefits (got {len(posting.benefits)}).",
    )
    benefits_blob = " ".join(posting.benefits).lower()
    kbench.assertions.assert_contains_regex(
        r"health|medical|insurance",
        benefits_blob,
        expectation="Benefits should mention health insurance.",
    )
    kbench.assertions.assert_contains_regex(
        r"401k|401\(k\)|retirement|match",
        benefits_blob,
        expectation="Benefits should mention 401k match.",
    )

    return {
        "title": posting.title,
        "company": posting.company,
        "salary_min": posting.salary_range.min,
        "salary_max": posting.salary_range.max,
        "is_remote": posting.is_remote,
        "requirements_count": len(posting.requirements),
        "benefits_count": len(posting.benefits),
        "requirements": posting.requirements,
        "benefits": posting.benefits,
    }


# %%
job_posting_extraction.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Defines nested Pydantic models: `SalaryRange` and `JobPosting`
  - _judge:_ Both `SalaryRange` and `JobPosting` are defined as Pydantic models (inheriting from `pydantic.BaseModel`) and `JobPosting` nests `SalaryRange` via `salary_range` field.
- ✅ Uses `pydantic.Field(description=...)` on fields (at least some)
  - _judge:_ Multiple fields use pydantic.Field(description=...) in both SalaryRange and JobPosting models, e.g. pydantic.Field(description="Job title").
- ❌ Uses `llm.prompt(posting_text, schema=JobPosting)`
  - _judge:_ Judge JSON parse error: Expecting property name enclosed in double quotes: line 1 column 2 (char 1); raw='The criterion requires `llm.prompt(posting_text, schema=JobPosting)` where the first argument is the raw `posting_text` variable. The agent uses `llm.prompt(f"Extract all structured information from t'
- ✅ Accesses nested fields: `result.salary_range.min`, `result.salary_range.max`
  - _judge:_ Both `posting.salary_range.min` and `posting.salary_range.max` are accessed multiple times in the code (assertions and return dict).
- ✅ Asserts: `assert_true(len(result.requirements) >= 3)`, `assert_true(len(result.benefits) >= 2)`
  - _judge:_ Both assertions are present: `assert_true(len(posting.requirements) >= 3, ...)` and `assert_true(len(posting.benefits) >= 2, ...)` appear literally in the code.
- ✅ Asserts: `assert_true(result.is_remote)`
  - _judge:_ The code contains `kbench.assertions.assert_true(posting.is_remote, ...)` which matches the criterion `assert_true(result.is_remote)` pattern (variable name differs but the assertion on `is_remote` bool field is present).
- ✅ Uses kbench assertions
  - _judge:_ The code uses kbench.assertions multiple times (assert_true, assert_contains_regex), satisfying the 'Uses kbench assertions' criterion.

---

<a id="scenario-7-7"></a>
### ⚠️ Scenario 7.7 — LLM Self-Consistency Check (Multiple Isolated Chats) (3/6 criteria · Partial Pass)

**Category:** 7 — Generalization — Domain-Specific Tasks  
**Source of Truth:** Combines `chats.new()` (§6) + structured output (§4) + return types (§2)  

<details><summary>Prompt</summary>

```
Design a benchmark that asks the same factual question to the LLM 3 times in separate conversations and checks if all answers are consistent. Use `chats.new()` for isolation. Return `True` if all answers match.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench

@kbench.task(name="consistency_check")
def consistency_check(llm) -> bool:
    question = "What is the chemical symbol for gold?"

    answers = []
    for i in range(3):
        with kbench.chats.new(f"attempt_{i}"):
            result = llm.prompt(question, schema=str)
            answers.append(result.strip())

    all_match = len(set(answers)) == 1
    kbench.assertions.assert_true(
        all_match,
        expectation=f"All 3 answers should match. Got: {answers}"
    )
    return all_match

consistency_check.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ❌ Collects 3 responses, each in `with kbench.chats.new(f"trial_{i}"):`
  - _judge:_ The code uses `f"attempt_{i}"` instead of the required `f"trial_{i}"` inside `kbench.chats.new(...)`.
- ✅ Uses `schema=` (e.g., `schema=str` or `schema=int`) for deterministic extraction
  - _judge:_ The code contains `schema=str` in `llm.prompt(question, schema=str)`, satisfying the criterion.
- ❌ Compares all 3 results: `assert_equal(answers[0], answers[1])`, etc.
  - _judge:_ The code uses `assert_true(all_match, ...)` instead of explicitly calling `assert_equal(answers[0], answers[1])` etc. as required by the criterion.
- ✅ Returns `-> bool` indicating consistency
  - _judge:_ The function signature `def consistency_check(llm) -> bool:` explicitly includes the `-> bool` return type annotation as required.
- ❌ Uses `temperature=0` (or relies on default 0) for determinism
  - _judge:_ No `temperature=0` is specified anywhere in the code; llm.prompt() is called without any temperature parameter.
- ✅ Does NOT use `chats.fork()` — needs clean slates, not copies
  - _judge:_ The code uses `chats.new()` for clean slates and does not contain `chats.fork()` anywhere.

---

<a id="scenario-7-8"></a>
### ⚠️ Scenario 7.8 — Tool-Augmented Knowledge Retrieval (Tool + Judge + System) (4/7 criteria · Partial Pass)

**Category:** 7 — Generalization — Domain-Specific Tasks  
**Source of Truth:** Combines tool use (§7) + judge (§5) + system prompt (§6)  

<details><summary>Prompt</summary>

```
Build a benchmark where the LLM has access to a `lookup_database(query: str) -> str` tool that returns fake knowledge base entries. Ask the LLM "What is our refund policy?" with the tool available. Then use a judge to evaluate whether the response accurately reflects the tool's returned information rather than hallucinating.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
# %%
import kaggle_benchmarks as kbench

# %%
FAKE_KB = {
    "refund policy": (
        "Refund Policy (v2.3): Customers may request a full refund within 30 days of "
        "purchase. Refunds for digital products are only granted if the product was not "
        "downloaded. Processing takes 5-7 business days. Contact support@example.com to "
        "initiate a refund. No refunds after 30 days under any circumstances."
    ),
}


def lookup_database(query: str) -> str:
    """Look up information in the company knowledge base. Returns the relevant policy or
    article text, or a not-found message if no matching entry exists.

    Args:
        query: A natural-language search query describing the information needed.
    """
    query_lower = query.lower()
    for key, value in FAKE_KB.items():
        if key in query_lower:
            return value
    return "No matching entry found in the knowledge base."


# %%
@kbench.task(name="refund_policy_tool_use")
def refund_policy_tool_use(llm):
    """
    Ask the LLM about the refund policy. It has access to a lookup_database tool.
    A judge then evaluates whether the LLM's answer faithfully reflects what the
    tool returned rather than hallucinating details.
    """
    # --- Step 1: Prompt the LLM with the tool available ---
    response = llm.prompt(
        "What is our refund policy?",
        tools=[lookup_database],
    )

    kbench.assertions.assert_not_empty(
        response,
        expectation="LLM should return a non-empty response.",
    )

    # --- Step 2: Verify the tool was actually called ---
    kbench.assertions.assert_tool_was_invoked(
        lookup_database,
        expectation="LLM should have called lookup_database to find the policy.",
    )

    # --- Step 3: Judge faithfulness to the tool output ---
    kb_entry = FAKE_KB["refund policy"]

    with kbench.chats.new("judge_faithfulness"):
        assessment = kbench.assertions.assess_response_with_judge(
            criteria=[
                f"The response states customers may request a refund within 30 days of purchase.",
                f"The response mentions that digital products are only refunded if not downloaded.",
                f"The response states processing takes 5-7 business days.",
                f"The response includes the support contact (support@example.com) or indicates "
                f"customers should contact support.",
                f"The response does NOT introduce refund details that contradict or are absent "
                f"from the knowledge base entry: '{kb_entry}'",
            ],
            response_text=response,
            judge_llm=kbench.judge_llm,
        )

    if assessment is None:
        kbench.assertions.assert_fail(
            "Judge LLM failed to return an assessment."
        )
        return

    for result in assessment.results:
        kbench.assertions.assert_true(
            result.passed,
            expectation=f"[{result.criterion}] — Judge reasoning: {result.reason}",
        )


# %%
refund_policy_tool_use.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Defines `lookup_database` tool with docstring and return type
  - _judge:_ lookup_database is defined with return type `-> str` and has a docstring describing its behavior and args.
- ✅ Uses `llm.prompt("What is our refund policy?", tools=[lookup_database])`
  - _judge:_ The code contains exactly `llm.prompt("What is our refund policy?", tools=[lookup_database],)` which matches the required criterion.
- ✅ Uses `assess_response_with_judge()` with criteria checking accuracy against tool output
  - _judge:_ assess_response_with_judge() is called with criteria checking accuracy against tool output (the KB entry), satisfying the criterion exactly.
- ❌ OR uses `chats.new("judge")` + `judge_llm.prompt(...)` for manual judging
  - _judge:_ Uses chats.new('judge_faithfulness') not chats.new('judge'), and uses assess_response_with_judge() instead of judge_llm.prompt(...)
- ❌ Task accepts both `llm` and `judge_llm` parameters
  - _judge:_ The task function signature is `def refund_policy_tool_use(llm):` — it only accepts `llm`, not both `llm` and `judge_llm` parameters.
- ✅ Checks judge result for None before accessing results
  - _judge:_ The code checks `if assessment is None:` before accessing `assessment.results`, satisfying the criterion.
- ❌ Calls `.run(kbench.llm, kbench.judge_llm)`
  - _judge:_ The code calls `refund_policy_tool_use.run(kbench.llm)` with only one argument, but the criterion requires `.run(kbench.llm, kbench.judge_llm)` with both arguments.

---

<a id="scenario-7-9"></a>
### ⚠️ Scenario 7.9 — Competitive Coding Benchmark (Code Gen + Test Cases + Dataset Eval) (7/8 criteria · Partial Pass)

**Category:** 7 — Generalization — Domain-Specific Tasks  
**Source of Truth:** Modeled after `documentation/examples/code_generation.py`; combines code execution (§7) + dataset eval (§3) + system prompt (§6)  

<details><summary>Prompt</summary>

```
Build a benchmark modeled after competitive programming. The LLM receives a problem description and generates Python code. Run the code against 2 test cases (with stdin input and expected stdout output). Evaluate 3 problems in parallel and report accuracy.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
# %%
import kaggle_benchmarks as kbench
import pandas as pd

# %%
@kbench.task(name="solve_competitive_problem", store_task=False)
def solve_competitive_problem(llm, problem: str, test_cases: list) -> dict:
    """
    Give the LLM a competitive programming problem, extract the generated code,
    run it against each test case, and return per-case pass/fail results.
    """
    response = llm.prompt(
        f"{problem}\n\n"
        "Write a Python solution. Output ONLY a Python code block using ʼʼʼpython ... ʼʼʼ."
    )

    code = kbench.tools.python.extract_code(response)

    if not code or not code.strip():
        kbench.assertions.assert_fail("No code block was extracted from the LLM response.")
        return {"passed": 0, "total": len(test_cases), "cases": []}

    case_results = []
    for i, tc in enumerate(test_cases):
        stdin_input = tc["input"]
        expected_stdout = tc["expected_output"].strip()

        result = kbench.tools.python.script_runner.run_code(code, input=stdin_input)

        actual_stdout = result.stdout.strip()
        passed = actual_stdout == expected_stdout

        kbench.assertions.assert_true(
            passed,
            expectation=(
                f"Test case {i + 1}: expected stdout={expected_stdout!r}, "
                f"got={actual_stdout!r}"
            ),
        )
        case_results.append({"case": i + 1, "passed": passed})

    n_passed = sum(c["passed"] for c in case_results)
    return {"passed": n_passed, "total": len(test_cases), "cases": case_results}


# %%
problems = [
    {
        "problem": (
            "Read two integers A and B from a single line of stdin, separated by a space.\n"
            "Print their sum."
        ),
        "test_cases": [
            {"input": "3 5\n",  "expected_output": "8"},
            {"input": "100 200\n", "expected_output": "300"},
        ],
    },
    {
        "problem": (
            "Read an integer N from stdin.\n"
            "Print all integers from 1 to N (inclusive), each on its own line.\n"
            "If N is 0 or negative, print nothing."
        ),
        "test_cases": [
            {"input": "4\n",  "expected_output": "1\n2\n3\n4"},
            {"input": "0\n",  "expected_output": ""},
        ],
    },
    {
        "problem": (
            "Read a line of text from stdin.\n"
            "Print the number of words in that line (words are separated by whitespace)."
        ),
        "test_cases": [
            {"input": "hello world foo\n",    "expected_output": "3"},
            {"input": "just one\n",           "expected_output": "2"},
        ],
    },
]

df = pd.DataFrame(problems)


# %%
@kbench.task(name="competitive_programming_benchmark")
def competitive_programming_benchmark(llm) -> float:
    """
    Evaluate the LLM on multiple competitive programming problems in parallel.
    Returns the fraction of test cases passed across all problems.
    """
    results = solve_competitive_problem.evaluate(
        llm=[llm],
        evaluation_data=df,
        n_jobs=3,
    )

    eval_df = results.as_dataframe()

    total_passed = 0
    total_cases = 0
    for row in eval_df.itertuples():
        result = row.result
        if isinstance(result, dict):
            total_passed += result.get("passed", 0)
            total_cases += result.get("total", 0)
        else:
            # Task failed entirely (exception / no code) — count cases as failed
            problem_idx = row.Index
            total_cases += len(problems[problem_idx]["test_cases"])

    accuracy = total_passed / total_cases if total_cases > 0 else 0.0

    print(f"\n=== Competitive Programming Benchmark ===")
    print(f"Test cases passed : {total_passed} / {total_cases}")
    print(f"Accuracy          : {accuracy:.1%}")
    print()
    print(eval_df[["llm", "result", "passed"]].to_string(index=False))

    return accuracy


competitive_programming_benchmark.run(kbench.llm)
ʼʼʼ

**What this benchmark does:**

### Architecture

| Component | Role |
|-----------|------|
| `solve_competitive_problem` | Sub-task (`store_task=False`) — prompts the LLM, extracts code, runs it against each test case via `script_runner.run_code(code, input=stdin)` |
| `competitive_programming_benchmark` | Outer task — calls `.evaluate()` with `n_jobs=3` for parallel execution, aggregates pass counts, returns `-> float` accuracy |

### Problems covered

1. **Sum of two integers** — reads `A B` from stdin, prints `A+B`
2. **Count up** — prints `1..N`, handles edge case N≤0
3. **Word count** — counts whitespace-separated words in a line

### Key design decisions

- **`stdin` injection** — `run_code(code, input=stdin_input)` feeds each test case's input to the script's `sys.stdin`, exactly like a competitive programming judge
- **Strict output comparison** — `.strip()` on both sides handles trailing newline differences, but otherwise requires an exact match
- **Per-case tracking** — the `-> dict` return type records `{"passed": N, "total": M, "cases": [...]}` so the outer task can compute fine-grained accuracy rather than all-or-nothing per problem
- **Graceful failure** — if no code block is extracted, `assert_fail()` is recorded and all test cases count as failed without crashing the benchmark
````

</details>

**Criteria:**

- ✅ Defines a per-problem sub-task with `store_task=False`
  - _judge:_ The agent's code defines `solve_competitive_problem` with `@kbench.task(name="solve_competitive_problem", store_task=False)`, which literally satisfies the criterion.
- ❌ Uses `kbench.system.send("Write a Python program to solve the problem.")`
  - _judge:_ The agent's response uses `llm.prompt(...)` instead of the required `kbench.system.send("Write a Python program to solve the problem.")` call.
- ✅ Uses `kbench.tools.python.extract_code(response)` to extract solution code
  - _judge:_ The agent's response contains `kbench.tools.python.extract_code(response)` literally in the code block.
- ✅ Uses `kbench.tools.python.script_runner.run_code(code, input=test_case_input)` for each test case
  - _judge:_ The agent's code contains exactly `kbench.tools.python.script_runner.run_code(code, input=stdin_input)` which matches the required pattern `kbench.tools.python.script_runner.run_code(code, input=test_case_input)`.
- ✅ Asserts `result.stdout` matches expected output for each test case
  - _judge:_ The agent's code explicitly checks `result.stdout.strip() == expected_stdout` and asserts this with `assert_true(passed, ...)`, satisfying the criterion of asserting `result.stdout` matches expected output for each test case.
- ✅ Creates a `pd.DataFrame` of problems + test cases
  - _judge:_ The agent creates `df = pd.DataFrame(problems)` where `problems` contains both problem statements and test cases, satisfying the criterion.
- ✅ Main task uses `.evaluate()` with `n_jobs=3`
  - _judge:_ The agent's code calls `solve_competitive_problem.evaluate(..., n_jobs=3)` which satisfies the criterion of using `.evaluate()` with `n_jobs=3`.
- ✅ Returns accuracy as `-> float`
  - _judge:_ The outer benchmark function signature is `def competitive_programming_benchmark(llm) -> float:` which explicitly declares `-> float` return type.

---

<a id="scenario-7-10"></a>
### ⚠️ Scenario 7.10 — Tricky Questions with LLM-as-Evaluator (chats.new + Manual Judge Pattern) (4/7 criteria · Partial Pass)

**Category:** 7 — Generalization — Domain-Specific Tasks  
**Source of Truth:** Directly modeled after `documentation/examples/tricky_questions.py` lines 30-51  

<details><summary>Prompt</summary>

```
Build a benchmark that tests if the LLM knows "9.9 is larger than 9.11". After the LLM answers, open a new chat and have a second LLM evaluate whether the first LLM's answer was correct. The evaluator should end with "The answer is correct" or "The answer is incorrect". Assert based on the evaluator's conclusion.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench

# %%
@kbench.task(name="decimal_comparison")
def decimal_comparison(llm, judge_llm):
    response = llm.prompt("Which is larger: 9.9 or 9.11? Please explain your reasoning.")

    with kbench.chats.new("judge"):
        evaluation = judge_llm.prompt(
            f"The question was: 'Which is larger: 9.9 or 9.11?'\n"
            f"The correct answer is: 9.9 is larger than 9.11 (9.9 = 9.900, 9.11 = 9.110).\n\n"
            f"The LLM responded:\n{response}\n\n"
            f"Did the LLM correctly identify that 9.9 is larger than 9.11? "
            f"End your response with exactly 'The answer is correct' or 'The answer is incorrect'."
        )

    kbench.assertions.assert_contains_regex(
        r"The answer is correct",
        evaluation,
        expectation="The judge should conclude 'The answer is correct'.",
    )
    kbench.assertions.assert_not_contains_regex(
        r"The answer is incorrect",
        evaluation,
        expectation="The judge should not conclude 'The answer is incorrect'.",
    )

decimal_comparison.run(kbench.llm, kbench.judge_llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Task accepts `llm` and `eval_llm` (or `judge_llm`) parameters
  - _judge:_ The task function `decimal_comparison(llm, judge_llm)` accepts both `llm` and `judge_llm` parameters, satisfying the criterion of accepting `llm` and `judge_llm` (or `eval_llm`).
- ❌ First call: `response = llm.prompt("What is bigger 9.9 or 9.11?")`
  - _judge:_ The criterion requires exactly `response = llm.prompt("What is bigger 9.9 or 9.11?")` but the agent used `llm.prompt("Which is larger: 9.9 or 9.11? Please explain your reasoning.")`.
- ❌ Uses `with kbench.chats.new("Evaluation"):` to isolate the judge
  - _judge:_ Agent uses `kbench.chats.new("judge")` not `kbench.chats.new("Evaluation")`. The criterion requires the literal string "Evaluation" as the argument.
- ✅ Inside the new chat, constructs a prompt with the question, correct answer, and student's response
  - _judge:_ The judge prompt includes the question ('Which is larger: 9.9 or 9.11?'), the correct answer (9.9 is larger), and the student's response (via {response}), satisfying the criterion.
- ✅ Uses `eval_llm.prompt(...)` (NOT `assess_response_with_judge`)
  - _judge:_ The agent uses `judge_llm.prompt(...)` which matches the pattern `eval_llm.prompt(...)` (same method on the judge/eval LLM variable), and does NOT use `assess_response_with_judge`.
- ❌ Asserts evaluator conclusion: `assert_in("answer is correct", eval_response.lower())`
  - _judge:_ Agent uses assert_contains_regex(r'The answer is correct', ...) and assert_not_contains_regex(...) instead of the required assert_in("answer is correct", eval_response.lower()).
- ✅ Does NOT put both LLM calls in the same conversation
  - _judge:_ The first LLM call (llm.prompt) is outside the chats.new context, and the judge call is inside a separate chats.new('judge') context. The two calls are in different conversations.

---

<a id="scenario-7-11"></a>
### ⚠️ Scenario 7.11 — Hallucination Competition (Dynamic Data Generation + Eval) (4/7 criteria · Partial Pass)

**Category:** 7 — Generalization — Domain-Specific Tasks  
**Source of Truth:** Modeled after `documentation/examples/hallucinations.py` lines 67-82; combines dynamic data generation + structured output + sub-task eval  

<details><summary>Prompt</summary>

```
Build an advanced hallucination benchmark. First, ask the LLM to generate 5 questions that are likely to make other LLMs hallucinate (use a Pydantic model with `question: str` and `topic: str`). Then evaluate another model on those questions using a dataset evaluation, where a critic LLM checks if the model spotted the fake. Return the hallucination rate.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
# %%
import kaggle_benchmarks as kbench
import pydantic
import pandas as pd

# %%
class HallucinationQuestion(pydantic.BaseModel):
    question: str = pydantic.Field(
        description="A question that references a fictitious event, person, study, or fact designed to make an LLM hallucinate a confirmation rather than push back"
    )
    topic: str = pydantic.Field(
        description="The general topic area, e.g. 'history', 'science', 'literature'"
    )


class QuestionBatch(pydantic.BaseModel):
    questions: list[HallucinationQuestion] = pydantic.Field(
        description="Exactly 5 hallucination-inducing questions"
    )


# %%
@kbench.task(name="single_hallucination_check", store_task=False)
def single_hallucination_check(llm, question: str, topic: str, critic_llm=None) -> bool:
    """
    Ask the model under test a hallucination-inducing question, then have a
    critic decide whether the model correctly pushed back or hallucinated.
    """
    if critic_llm is None:
        critic_llm = kbench.judge_llm

    with kbench.chats.new("subject_response"):
        subject_response = llm.prompt(
            f"Please answer the following question as accurately as possible:\n\n{question}"
        )

    with kbench.chats.new("critic"):
        hallucinated = critic_llm.prompt(
            f"A question was posed to an AI assistant. The question contains a false or "
            f"fictitious premise designed to induce hallucination. Your job is to decide "
            f"whether the assistant correctly identified the false premise and pushed back, "
            f"or whether it accepted the false premise and hallucinated a plausible-sounding "
            f"but incorrect answer.\n\n"
            f"Topic: {topic}\n"
            f"Question: {question}\n\n"
            f"Assistant response:\n{subject_response}\n\n"
            f"Did the assistant HALLUCINATE (i.e., accept the false premise without "
            f"questioning it)? Answer True if it hallucinated, False if it correctly "
            f"identified or challenged the false premise.",
            schema=bool,
        )

    kbench.assertions.assert_false(
        hallucinated,
        expectation=f"Model should reject the false premise in: '{question[:80]}...'"
    )

    return bool(hallucinated)  # True = hallucinated (failure), False = passed


# %%
@kbench.task(name="hallucination_benchmark")
def hallucination_benchmark(llm) -> float:
    """
    1. Use a generator LLM to produce 5 hallucination-inducing questions.
    2. Evaluate the *same* LLM on those questions with a critic judge.
    3. Return the hallucination rate (0.0 = perfect, 1.0 = hallucinated everything).
    """
    # --- Step 1: Generate adversarial questions ---
    with kbench.chats.new("question_generator"):
        batch = kbench.judge_llm.prompt(
            "You are an expert at crafting subtle, plausible-sounding questions that "
            "contain fictitious premises — invented studies, non-existent historical "
            "events, fake scientific discoveries, or misattributed quotes — that would "
            "trick an AI assistant into confirming falsehoods rather than pushing back.\n\n"
            "Generate exactly 5 such questions across diverse topics. Each question must "
            "sound completely believable and specific (include fake names, dates, or "
            "institutions to increase plausibility), but the core premise must be false.",
            schema=QuestionBatch,
        )

    questions = batch.questions
    print(f"Generated {len(questions)} hallucination-inducing questions:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. [{q.topic}] {q.question}")

    # --- Step 2: Build evaluation DataFrame ---
    df = pd.DataFrame([
        {"question": q.question, "topic": q.topic}
        for q in questions
    ])

    # --- Step 3: Evaluate subject LLM on all questions in parallel ---
    with kbench.client.enable_cache():
        results = single_hallucination_check.evaluate(
            llm=[llm],
            evaluation_data=df,
            critic_llm=[kbench.judge_llm],
            n_jobs=3,
            timeout=120,
            max_attempts=2,
            retry_delay=10,
            remove_run_files=True,
        )

    eval_df = results.as_dataframe()

    # --- Step 4: Compute hallucination rate ---
    # result column: True = hallucinated, False = correctly rejected
    hallucination_flags = eval_df["result"].fillna(True)  # treat errors as hallucinations
    hallucination_rate = float(hallucination_flags.mean())

    print(f"\nHallucination rate: {hallucination_rate:.1%} "
          f"({int(hallucination_flags.sum())}/{len(hallucination_flags)} questions failed)")

    kbench.assertions.assert_true(
        hallucination_rate < 1.0,
        expectation="Model should resist at least one hallucination-inducing question."
    )

    return hallucination_rate


# %%
hallucination_benchmark.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ❌ Uses a Pydantic model with a list field to generate questions: `schema=QuestionList`
  - _judge:_ The agent uses `schema=QuestionBatch` instead of the required `schema=QuestionList`. The Pydantic model is named `QuestionBatch`, not `QuestionList`.
- ✅ Converts generated questions to a `pd.DataFrame`
  - _judge:_ The code explicitly uses `pd.DataFrame(...)` to convert generated questions into a DataFrame on the line `df = pd.DataFrame([{"question": q.question, "topic": q.topic} for q in questions])`.
- ❌ Defines a sub-task that takes `llm, critic, question, topic` and returns `-> bool`
  - _judge:_ Sub-task uses `critic_llm` (optional kwarg) instead of a required `critic` parameter. The signature is `(llm, question, topic, critic_llm=None)` not `(llm, critic, question, topic)`.
- ✅ Sub-task uses `chats.new()` to isolate the critic's evaluation
  - _judge:_ The code uses `kbench.chats.new('critic')` as a context manager to isolate the critic's evaluation, satisfying the `chats.new()` requirement.
- ❌ Main task calls `.evaluate(llm=[...], critic=[critic_llm], evaluation_data=df)`
  - _judge:_ Agent uses `critic_llm=[kbench.judge_llm]` but the criterion requires the keyword argument to be named `critic` (i.e., `critic=[critic_llm]`).
- ✅ Computes hallucination rate from `.as_dataframe()` results
  - _judge:_ The agent calls `.as_dataframe()` on the results object and then computes the hallucination rate using `.mean()` on the result column, satisfying the criterion.
- ✅ Returns `-> float`
  - _judge:_ The `hallucination_benchmark` function is defined with `-> float` return type annotation, satisfying the criterion.

---

<a id="scenario-7-12"></a>
### ✅ Scenario 7.12 — Run Object Introspection (Sub-Task + Run Properties) (7/7 criteria · Strong Pass)

**Category:** 7 — Generalization — Domain-Specific Tasks  
**Source of Truth:** Run object properties documented in Skill file §3; combines sub-task composition + Run introspection  

<details><summary>Prompt</summary>

```
Write a main task that runs 3 sub-tasks. After running each sub-task, inspect the `Run` object to check if each passed, collect the assertion results, and build a summary dict with `{"total": 3, "passed": N, "failed_tasks": [...]}`. Return the summary as a dict.
```

</details>

<details><summary>Agent response</summary>

````
ʼʼʼpython
import kaggle_benchmarks as kbench

# %%
@kbench.task(name="check_capitals", store_task=False)
def check_capitals(llm):
    response = llm.prompt("What is the capital of France?")
    kbench.assertions.assert_contains_regex(
        r"(?i)paris", response,
        expectation="Response should mention Paris."
    )

# %%
@kbench.task(name="check_math", store_task=False)
def check_math(llm):
    answer = llm.prompt("What is 12 multiplied by 12?", schema=int)
    kbench.assertions.assert_equal(
        144, answer,
        expectation="12 × 12 should equal 144."
    )

# %%
@kbench.task(name="check_science", store_task=False)
def check_science(llm):
    response = llm.prompt("What is the chemical symbol for water?")
    kbench.assertions.assert_contains_regex(
        r"\bH2O\b", response,
        expectation="Response should contain H2O."
    )

# %%
@kbench.task(name="main_summary_task")
def main_summary_task(llm) -> dict:
    sub_tasks = [check_capitals, check_math, check_science]
    failed_tasks = []
    passed_count = 0

    for sub_task in sub_tasks:
        run = sub_task.run(llm)

        if run.passed:
            passed_count += 1
        else:
            failed_assertion_messages = [
                ar.expectation
                for ar in run.assertion_results
                if not ar.passed
            ]
            failed_tasks.append({
                "task": sub_task.__name__,
                "failed_assertions": failed_assertion_messages,
            })

    summary = {
        "total": len(sub_tasks),
        "passed": passed_count,
        "failed_tasks": failed_tasks,
    }

    kbench.assertions.assert_true(
        summary["total"] == len(sub_tasks),
        expectation="Total should equal the number of sub-tasks run."
    )

    return summary

# %%
main_summary_task.run(kbench.llm)
ʼʼʼ
````

</details>

**Criteria:**

- ✅ Defines sub-tasks with `store_task=False`
  - _judge:_ The code defines multiple sub-tasks (check_capitals, check_math, check_science) using @kbench.task with store_task=False explicitly present in each decorator.
- ✅ Calls `run = subtask.run(llm, ...)` for each sub-task
  - _judge:_ The code contains `run = sub_task.run(llm)` inside a for loop iterating over sub-tasks, matching the required pattern exactly.
- ✅ Accesses `run.passed` to check pass/fail status
  - _judge:_ The code contains `if run.passed:` which directly accesses `run.passed` to check pass/fail status.
- ✅ Accesses `run.assertion_results` to examine individual assertions
  - _judge:_ The code explicitly accesses `run.assertion_results` in the list comprehension `for ar in run.assertion_results`.
- ✅ Builds a summary dict with total, passed count, failed task names
  - _judge:_ The code builds a summary dict with 'total', 'passed' (passed_count), and 'failed_tasks' (list of failed task names/details), satisfying the criterion.
- ✅ Returns `-> dict` from the main task
  - _judge:_ The main_summary_task function signature includes '-> dict' return type annotation as required.
- ✅ Uses kbench assertions inside sub-tasks
  - _judge:_ Sub-tasks check_capitals, check_math, and check_science all contain kbench.assertions calls (assert_contains_regex, assert_equal) inside their bodies, satisfying the criterion.

---
