# Test Scenarios: kaggle-benchmarks Skill Files

This document defines test scenarios for validating the skill files in [`skills/kaggle-benchmarks/`](../skills/kaggle-benchmarks/). Each scenario provides a **prompt** to give an agent that has the skill files as context, **expected answer criteria** the agent's response must satisfy, and a **source of truth** reference for verification.

## How to Use This Test Script

1. Start a new agent conversation
2. Provide the skill file `skills/kaggle-benchmarks/SKILL.md` as context
3. For each scenario, send the **Prompt** to the agent
4. Verify the agent's response against the **Expected Answer** criteria
5. Mark each criterion as ✅ PASS or ❌ FAIL

---

## Category 1: Basic — Simple Task + Assertion (Cookbook Recipes)

These map directly to golden test patterns in `golden_tests/test_cookbook_examples.py`.

### Scenario 1.1 — Simple Q&A with Regex Check

**Prompt:**
> Write a kaggle-benchmarks task that asks the LLM "What is Kaggle?" and asserts the response contains the word "platform" (case-insensitive). Run it with `kbench.llm`.

**Expected Answer:**
- [ ] Uses `import kaggle_benchmarks as kbench`
- [ ] Uses `@kbench.task()` decorator
- [ ] First parameter of task function is `llm`
- [ ] Calls `llm.prompt("What is Kaggle?")`
- [ ] Uses `kbench.assertions.assert_contains_regex(r"(?i)platform", response)` or `kbench.assertions.assert_in("platform", response.lower())`
- [ ] Does NOT use plain Python `assert`
- [ ] Calls `.run(kbench.llm)` at the end

**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 149–180 (`assess_with_judge_task`); `documentation/examples/quick_start_examples.py` lines 29–38

---

### Scenario 1.2 — Extract Integer with Structured Output

**Prompt:**
> Write a kaggle-benchmarks task that gives the LLM the text "The Apollo 11 mission landed on the Moon in 1969." and extracts the year as an integer using `schema=int`. Assert the year equals 1969.

**Expected Answer:**
- [ ] Uses `@kbench.task()` decorator with `llm` as first param
- [ ] Uses `llm.prompt("...", schema=int)`
- [ ] Uses `kbench.assertions.assert_equal(1969, year)`
- [ ] Does NOT manually parse or cast the response to int
- [ ] Calls `.run(kbench.llm)`

**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 187–199 (`test_extract_int`)

---

### Scenario 1.3 — Extract Bool with Structured Output

**Prompt:**
> Write a benchmark task that asks the LLM whether "I absolutely loved this movie! It was fantastic." is a positive review. Use `schema=bool` to get a boolean answer and assert it's True.

**Expected Answer:**
- [ ] Uses `schema=bool` in `llm.prompt()`
- [ ] Uses `kbench.assertions.assert_true(is_positive)`
- [ ] Does NOT parse the response string manually
- [ ] Has `@kbench.task()` decorator

**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 206–218 (`test_extract_bool`)

---

### Scenario 1.4 — Extract Dict with Schema

**Prompt:**
> Write a task that extracts person details (name, age, occupation) from the text "Contact info: John Doe, age 42, works as a Software Engineer." Use an inline dict schema `{"name": str, "age": int, "occupation": str}`. Assert each field.

**Expected Answer:**
- [ ] Uses `schema={"name": str, "age": int, "occupation": str}`
- [ ] Accesses fields via `person.name`, `person.age`, `person.occupation` (dot notation, not dict indexing)
- [ ] Uses multiple kbench assertions (e.g., `assert_equal`, `assert_contains_regex`)
- [ ] Does NOT define a dataclass or pydantic model (uses inline dict)

**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 225–246 (`test_extract_dict`)

---

### Scenario 1.5 — Extract Dataclass with Structured Output

**Prompt:**
> Write a task that asks the LLM to generate a level 5 wizard character for a fantasy game. Define a `@dataclass` with fields: name (str), class_type (str), level (int), inventory (str). Assert the class_type contains "wizard" and level equals 5.

**Expected Answer:**
- [ ] Defines a `@dataclass` with `name`, `class_type`, `level`, `inventory`
- [ ] Uses `llm.prompt("...", schema=RPGCharacter)`
- [ ] Uses `kbench.assertions.assert_contains_regex(r"(?i)wizard", character.class_type)`
- [ ] Uses `kbench.assertions.assert_equal(5, character.level)`
- [ ] Has `@kbench.task()` decorator

**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 253–281 (`test_extract_dataclass`)

---

### Scenario 1.6 — Extract Pydantic Model

**Prompt:**
> Write a task that asks the LLM about planet Jupiter. Define a Pydantic `BaseModel` with fields: name (str), mass_earth_masses (float), has_life (bool), moons (list[str]). Assert mass > 300 and moons list is non-empty.

**Expected Answer:**
- [ ] Defines a Pydantic `BaseModel` (imports from `pydantic`)
- [ ] Uses `Field()` with descriptions (optional but preferred)
- [ ] Uses `llm.prompt("...", schema=Planet)`
- [ ] Uses `kbench.assertions.assert_true(planet.mass_earth_masses > 300)`
- [ ] Uses `kbench.assertions.assert_true(len(planet.moons) > 0)` or `assert_not_empty`

**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 288–313 (`test_extract_pydantic`)

---

### Scenario 1.7 — Composite Pydantic with Nested List

**Prompt:**
> Write a task that asks the LLM to list the 6 main characters of Friends. Define TWO Pydantic models: `Actor` with fields `actor_name` and `role_name`, and `Casting` with a field `actors: list[Actor]`. Assert there are exactly 6 actors and that "Jennifer" is among the actor names.

**Expected Answer:**
- [ ] Defines two Pydantic models (nested structure)
- [ ] Uses `schema=Casting` in the prompt
- [ ] Uses `kbench.assertions.assert_equal(len(casting.actors), 6)` or similar
- [ ] Joins actor names and uses `kbench.assertions.assert_in("Jennifer", ...)`
- [ ] Does NOT return a flat list — uses nested Pydantic models

**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 320–344 (`test_extract_composite_pydantic`)

---

### Scenario 1.8 — Multi-Turn Memory

**Prompt:**
> Write a benchmark task that tests multi-turn conversation memory. First tell the LLM "My name is Alice", then ask "What is my name?". Assert the response contains "Alice".

**Expected Answer:**
- [ ] Calls `llm.prompt()` twice in sequence (multi-turn)
- [ ] First call: `llm.prompt("My name is Alice.")`
- [ ] Second call: `response = llm.prompt("What is my name?")`
- [ ] Uses `kbench.assertions.assert_contains_regex(r"(?i)alice", response)` or `assert_in`
- [ ] Does NOT use `chats.new()` (relies on automatic history)

**Source of Truth:** `golden_tests/test_code_generator.py` lines 122–125 (`memory-alice`); `documentation/examples/simple_task.py` lines 33–36 (`subtask2`)

---

### Scenario 1.9 — Simple Greeting Assertion

**Prompt:**
> Write the simplest possible benchmark task: send "Hello!" to the LLM and assert the response is not empty.

**Expected Answer:**
- [ ] Uses `@kbench.task()` decorator
- [ ] Calls `llm.prompt("Hello!")`
- [ ] Uses `kbench.assertions.assert_not_empty(response)`
- [ ] Calls `.run(kbench.llm)`
- [ ] No return type annotation (pass/fail task)

**Source of Truth:** `documentation/examples/simple_task.py` lines 27–30 (`subtask1`); `documentation/examples/simple_multiple_tasks.py` lines 37–40 (`task1`)

---

### Scenario 1.10 — Reasoning Parameter

**Prompt:**
> Write a task that asks "What is 2 + 2?" using reasoning mode set to "low". Assert the response contains "4".

**Expected Answer:**
- [ ] Uses `llm.prompt("...", reasoning="low")`
- [ ] Uses `kbench.assertions.assert_contains_regex(r"4", response)`
- [ ] Has `@kbench.task()` decorator
- [ ] Does NOT set reasoning via a separate configuration

**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 601–619 (`test_reasoning_param`)

---

### Scenario 1.11 — Image Input (URL)

**Prompt:**
> Write a task that sends the Kaggle logo image (URL: `https://www.kaggle.com/static/images/site-logo.png`) to the LLM and asks "What does this logo say?". Assert the response contains "kaggle" (case-insensitive).

**Expected Answer:**
- [ ] Imports `images` from `kaggle_benchmarks.content_types`
- [ ] Uses `images.from_url("...")` to create image object
- [ ] Uses `llm.prompt("...", image=image)` (preferred approach)
- [ ] Uses `kbench.assertions.assert_contains_regex(r"(?i)kaggle", response)`

**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 404–427 (`test_image_url`)

---

### Scenario 1.12 — Audio Input

**Prompt:**
> Write a benchmark task that sends an MP3 audio file at path "speech.mp3" to the LLM and asks it to transcribe the audio. Assert the response is not empty.

**Expected Answer:**
- [ ] Imports `audios` from `kaggle_benchmarks.content_types`
- [ ] Uses `audios.from_path("speech.mp3")` to create audio object
- [ ] Uses `llm.prompt("Transcribe this audio.", audio=audio)`
- [ ] Uses `kbench.assertions.assert_not_empty(response)`
- [ ] Has `@kbench.task()` decorator
- [ ] Does NOT try to use `user.send()` for the audio (uses `llm.prompt(audio=)` instead)

**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 490–516 (`test_audio_local_file`, `test_audio_url`, `test_audio_base64`); Skill file §4 Audio

---

### Scenario 1.13 — Video Input (URL)

**Prompt:**
> Write a benchmark task that sends a YouTube video URL to the LLM and asks "What is happening in this video?". Assert the response is not empty.

**Expected Answer:**
- [ ] Imports `videos` from `kaggle_benchmarks.content_types`
- [ ] Uses `videos.from_url("https://www.youtube.com/watch?v=...")` to create video object
- [ ] Uses `llm.prompt("...", video=video)`
- [ ] Uses `kbench.assertions.assert_not_empty(response)`
- [ ] Has `@kbench.task()` decorator

**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 430–465 (`test_video_url`); Skill file §4 Videos

---

## Category 2: Basic — Function Tool Use

### Scenario 2.1 — Simple Function Tool

**Prompt:**
> Write a benchmark task that provides a calculator tool to the LLM. The tool `run_simple_calculator(a: float, b: float, operator: str) -> float` supports +, -, *, /. Ask the LLM "What is 50 plus 25?" and pass the calculator as a tool. Assert the final answer contains "75".

**Expected Answer:**
- [ ] Defines the `run_simple_calculator` function with proper type hints and docstring
- [ ] Uses `llm.prompt("...", tools=[run_simple_calculator])`
- [ ] Uses `kbench.assertions` to check the response contains "75"
- [ ] Does NOT manually parse tool call JSON (uses automatic tool calling)
- [ ] Has `@kbench.task()` decorator

**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 660–685 (`test_simple_tool_use`); Skill file §7 "Custom Function Tools"

---

### Scenario 2.2 — Multiple Tools Selection

**Prompt:**
> Write a benchmark task where the LLM is given TWO tools: `add_tool(a, b)` and `multiply_tool(a, b)`. Ask it "What is 12 multiplied by 34? Use the multiply_tool." Verify the correct tool was called.

**Expected Answer:**
- [ ] Defines two separate tool functions with docstrings
- [ ] Uses `llm.prompt("...", tools=[add_tool, multiply_tool])`
- [ ] Uses kbench assertion to verify the answer contains "408" (12*34)
- [ ] Has `@kbench.task()` decorator

**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 708–736 (`test_multiple_tool_selection`)

---

### Scenario 2.3 — Tool Error Handling

**Prompt:**
> Write a benchmark task where the tool function raises a ValueError. The LLM should be asked to call the tool and then report the error. Verify the LLM mentions "error" or "failed" in its response.

**Expected Answer:**
- [ ] Defines a tool function that raises `ValueError`
- [ ] Uses `llm.prompt("...", tools=[flaky_tool])`
- [ ] Uses `kbench.assertions.assert_contains_regex(r"(?i)error|failed", response)`
- [ ] Has `@kbench.task()` decorator
- [ ] Does NOT try-catch around the `llm.prompt()` call (the library handles tool errors)

**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 741–762 (`test_tool_error_handling`); Skill file §7

---

## Category 3: Basic — Code Execution

### Scenario 3.1 — Extract and Run Code

**Prompt:**
> Write a benchmark task that asks the LLM to write Python code to compute the factorial of 10 and print it. Extract the code, run it, and assert the output is "3628800" with no errors.

**Expected Answer:**
- [ ] Calls `llm.prompt()` with a coding prompt
- [ ] Uses `kbench.tools.python.extract_code(response)` to extract
- [ ] Uses `kbench.tools.python.script_runner.run_code(code)` to execute
- [ ] Asserts `result.stderr` is empty: `kbench.assertions.assert_empty(result.stderr.strip())`
- [ ] Asserts `result.stdout` contains "3628800": `kbench.assertions.assert_equal("3628800", result.stdout.strip())`
- [ ] Uses kbench assertions, NOT plain assert

**Source of Truth:** `golden_tests/test_code_generator.py` lines 112–116 (`factorial-10`); `documentation/examples/simple_multiple_tasks.py` lines 58–70 (`task4`)

---

### Scenario 3.2 — Code Extraction + Subchat in One Task

**Prompt:**
> Write a benchmark task that does three things in sequence:
> 1. Sends "Hello!" and asserts the response is not empty.
> 2. Opens a subchat, sends "Hello in subchat!" and asserts that response is not empty.
> 3. Asks the LLM to write Python code to print 'hello world!', extracts the code, runs it, and asserts the output matches.

**Expected Answer:**
- [ ] Has three sequential steps within one task
- [ ] Uses `with kbench.chats.new("subchat1"):` for step 2
- [ ] Uses `kbench.tools.python.extract_code()` and `script_runner.run_code()` for step 3
- [ ] Uses `kbench.assertions.assert_not_empty()` for steps 1 and 2
- [ ] Uses `kbench.assertions.assert_equal("hello world!", result.stdout.strip())` for step 3
- [ ] All assertions are kbench assertions

**Source of Truth:** `documentation/examples/simple_multiple_tasks.py` lines 58–71 (`task4`)

---

## Category 4: Dataset Evaluation

### Scenario 4.1 — Basic DataFrame Evaluation with Bool Return

**Prompt:**
> Write a benchmark task that evaluates the LLM on a set of riddles. The task should:
> 1. Accept `llm`, `riddle`, and `answer_keyword` parameters.
> 2. Prompt the LLM with the riddle.
> 3. Return `True` if the answer_keyword is found in the response (case-insensitive), `False` otherwise.
> 4. Create a DataFrame with 3 riddles and their answer keywords.
> 5. Run `.evaluate()` on the DataFrame with `n_jobs=3`.
> 6. Print the results table.
>
> Use these riddles:
> - "I have cities but no houses. What am I?" → "map"
> - "What has an eye but cannot see?" → "needle"
> - "What has to be broken before you can use it?" → "egg"

**Expected Answer:**
- [ ] Task has `-> bool` return type annotation
- [ ] Task parameters are `llm, riddle, answer_keyword` (matching DataFrame columns)
- [ ] Returns `answer_keyword.lower() in response.lower()`
- [ ] Creates `pd.DataFrame` with the 3 riddles
- [ ] Calls `.evaluate(llm=[kbench.llm], evaluation_data=df, n_jobs=3)`
- [ ] `llm` parameter is passed as a LIST `[kbench.llm]`
- [ ] Calls `.as_dataframe()` on the result

**Source of Truth:** `documentation/examples/quick_start_examples.py` lines 244–278 (`solve_and_check_riddle`)

---

### Scenario 4.2 — Sub-Task + Main Task with Accuracy Reporting

**Prompt:**
> Write a two-level benchmark evaluation:
> 1. A sub-task (`store_task=False`) that takes `llm, question, answer` and returns a dict with "question", "gold_target", "predicted_answer", and "is_correct" fields.
> 2. A main task that receives `llm` and a `df` parameter, calls the sub-task's `.evaluate()` inside `kbench.client.enable_cache()`, and returns `tuple[float, float]` of (accuracy, std).
> 3. Use this data: "Capital of Singapore?" → "Singapore", "Capital of France?" → "Paris"

**Expected Answer:**
- [ ] Sub-task has `store_task=False` in decorator
- [ ] Sub-task returns `-> dict` with all four fields
- [ ] Sub-task checks `answer.lower() in response.lower()`
- [ ] Main task wraps evaluate in `with kbench.client.enable_cache():`
- [ ] Main task calls `sub_task.evaluate(llm=[llm], evaluation_data=df, ...)`
- [ ] Main task uses `runs.as_dataframe()` to get results
- [ ] Main task computes `eval_df.result.str.get("is_correct").mean()` and `.std()`
- [ ] Main task returns `-> tuple[float, float]`
- [ ] Uses `float()` to convert from numpy types

**Source of Truth:** `documentation/examples/dataset_evaluation.py` lines 36–77; `golden_tests/test_cookbook_examples.py` lines 350–397

---

### Scenario 4.3 — Evaluate with Stop Condition and Retry

**Prompt:**
> Show the full set of parameters available when calling `.evaluate()` on a dataset. Write a task that uses `stop_condition`, `max_attempts`, `retry_delay`, `timeout`, and `remove_run_files`.

**Expected Answer:**
- [ ] Uses `stop_condition=lambda runs: len(runs) == df.shape[0]`
- [ ] Uses `max_attempts=` (e.g., 1 or 3)
- [ ] Uses `retry_delay=15` (in seconds)
- [ ] Uses `timeout=120` (per-job timeout in seconds)
- [ ] Uses `remove_run_files=True`
- [ ] Uses `n_jobs=` for parallel execution
- [ ] Uses `llm=[llm]` (list format)

**Source of Truth:** `documentation/examples/dataset_evaluation.py` lines 57–67; `src/kaggle_benchmarks/tasks.py` `evaluate()` lines 154–204

---

### Scenario 4.4 — Multi-Model Comparison on Dataset

**Prompt:**
> Write a benchmark that evaluates TWO different models on the same Q&A dataset and compares their accuracy. Use `kbench.llm` and `kbench.judge_llm` as the two models.

**Expected Answer:**
- [ ] Defines a per-row sub-task with `store_task=False`
- [ ] Creates a list of multiple models: `llms = [kbench.llm, kbench.judge_llm]`
- [ ] Calls `.evaluate(llm=llms, evaluation_data=df, ...)`
- [ ] Calculates `n_total = len(llms) * df.shape[0]` for the stop condition
- [ ] Uses `stop_condition=lambda runs: len(runs) == n_total`
- [ ] Uses `n_jobs=` for parallel execution
- [ ] Calls `.as_dataframe()` for results

**Source of Truth:** `documentation/examples/dataset_evaluation.py` lines 80–97 (multi-model evaluation)

---

### Scenario 4.5 — Math Word Problems Evaluation

**Prompt:**
> Write a benchmark that tests the LLM on 5 math word problems and returns the overall accuracy as a float. The problems are:
> 1. "What is 15% of 200?" → 30
> 2. "What is 7 × 8?" → 56
> 3. "What is the square root of 144?" → 12
> 4. "If a train travels 60 mph for 2.5 hours, how far does it go?" → 150
> 5. "What is 1000 ÷ 8?" → 125

**Expected Answer:**
- [ ] Creates a DataFrame or list of questions with expected answers
- [ ] Uses structured output (`schema=int` or `schema=float`) OR regex/string matching to check answers
- [ ] Returns `-> float` accuracy score between 0.0 and 1.0
- [ ] Either uses `.evaluate()` on a DataFrame OR loops through questions manually
- [ ] Uses kbench assertions for per-question checks
- [ ] Properly handles numeric comparison (not just string matching)

**Source of Truth:** `golden_tests/test_code_generator.py` lines 142–145 (`math-word-problems`)

---

### Scenario 4.6 — Failure-Tolerant Evaluation (`on_failure="continue"`)

**Prompt:**
> I'm running a 500-sample benchmark against a third-party API. A few samples fail every run because of API timeouts and 5xx errors. By default `.evaluate()` raises on the first failure and I lose all the work. How do I make it collect failures into the results instead of raising?

**Expected Answer:**
- [ ] Recommends `on_failure="continue"` parameter on `.evaluate()`
- [ ] Explains the default is `"raise"` (which is why the eval aborts today)
- [ ] Explains failed runs are returned in the `Runs` object with `status=FAILED`
- [ ] Shows splitting via `results.completed_runs` and `results.errored_runs`
- [ ] Warns that aggregating over the full `Runs` (e.g., `.as_dataframe().result.mean()`) will break because failed runs carry the `results.FAILED` sentinel — must aggregate over `results.completed_runs` only
- [ ] Mentions inspecting `run.error_message` on errored runs for debugging

**Source of Truth:** `SKILL.md` §3 "Failure Handling: `on_failure='raise'` vs `'continue'`"; `golden_tests/test_cookbook_examples.py` `test_dataset_eval_resilient`; `tests/test_benchmarks.py` `test_runs_completed_and_errored_partition`

---

### Scenario 4.7 — Resilient Production Pattern (Cache + Retry + Continue)

**Prompt:**
> Write a complete benchmark task for a 500-sample evaluation against a flaky API. The eval should: (1) collect failures rather than raising, (2) automatically retry failed samples up to 3 times (without re-running samples that already succeeded), (3) merge results from all attempts, (4) return accuracy as a float computed over only the samples that completed.

**Expected Answer:**
- [ ] Defines a per-sample sub-task with `store_task=False`
- [ ] Main task wraps `.evaluate()` in `with kbench.client.enable_cache():`
- [ ] Passes `on_failure="continue"` to `.evaluate()`
- [ ] Passes `max_attempts=3` (or similar > 1) to `.evaluate()`
- [ ] May also pass `retry_delay=` for backoff between attempts
- [ ] Uses `results.completed_runs.as_dataframe()` (NOT `results.as_dataframe()`) for accuracy aggregation
- [ ] Optionally reports `len(results.errored_runs)` for visibility
- [ ] Returns `-> float` (or `-> dict` with accuracy + error stats)
- [ ] Does NOT manually loop and call `.run()` per sample — uses `.evaluate()`
- [ ] Does NOT try to catch exceptions inside the task body (that would poison the cache with bogus successes)

**Source of Truth:** `SKILL.md` §3 "Resilient Pattern for Large Datasets" + §9 Pattern H.5; `cookbook.md` "Recipe: Best Practices for Large Datasets"; `documentation/examples/dataset_evaluation.py`

---

### Scenario 4.8 — Default `on_failure` Behavior

**Prompt:**
> What's the default value of `on_failure` on `.evaluate()`? What happens when I don't pass it and one of my samples raises?

**Expected Answer:**
- [ ] Says the default is `"raise"`
- [ ] In dev mode: the first per-sample exception aborts the eval immediately (joblib worker raises, propagates out)
- [ ] In Kaggle batch mode: all parallel workers finish, then `evaluate()` raises a `RuntimeError` summarizing the failures (NOT silent partial results)
- [ ] The `RuntimeError` message includes a hint pointing at `on_failure="continue"` and the `max_attempts` + `enable_cache()` pattern
- [ ] Says `"raise"` is the right default because silent failures are worse than loud ones
- [ ] Does NOT claim the default is `"continue"` or that batch mode silently drops failures (that was old behavior, fixed in v0.7.0)

**Source of Truth:** `SKILL.md` §3 "Failure Handling" table; `src/kaggle_benchmarks/tasks.py` `evaluate()` `on_failure` parameter default; `docs/large_eval_rerun/detailed_design.md` §1

---

## Category 5: Medium — Combining Basics

### Scenario 5.1 — Structured Output + Judge Evaluation

**Prompt:**
> Write a benchmark task that:
> 1. Asks the LLM "What is Kaggle?" and stores the response.
> 2. Asserts the response contains "platform" using `assert_in`.
> 3. Uses `assess_response_with_judge()` with a judge LLM to evaluate whether:
>    - The answer mentions data science or machine learning
>    - The answer mentions competitions
> 4. Iterates over the judge results and asserts each one passed.

**Expected Answer:**
- [ ] Task has two LLM params: `llm` and `judge_llm`
- [ ] Uses `kbench.assertions.assert_in("platform", response.lower())`
- [ ] Uses `kbench.assertions.assess_response_with_judge(criteria=[...], response_text=response, judge_llm=judge_llm)`
- [ ] RECOMMENDED: Checks judge result for `None` before accessing `.results` (the function can return `None` on failure)
- [ ] Iterates: `for result in assessment.results:`
- [ ] Uses `kbench.assertions.assert_true(result.passed, expectation=f"...{result.criterion}...{result.reason}")`
- [ ] Calls `.run(kbench.llm, kbench.judge_llm)`

> **Note:** The golden test (`test_cookbook_examples.py` lines 157–170) does NOT check for None because it runs
> with `continue_with_exceptions = False` globally. In user code, always check for None as shown in Skill file §5.

**Source of Truth:** `golden_tests/test_cookbook_examples.py` lines 149–180 (`assess_with_judge_task`)

---

### Scenario 5.2 — Hallucination Detection with Structured Output

**Prompt:**
> Write a benchmark task that detects hallucination. Ask the LLM about a fictitious concept (e.g., "the Zipflanger Theorem in particle physics"). Use a dict schema `{"answer": bool, "explanation": str}`. Assert the model says it doesn't exist (`assert_false` on `answer`) and the explanation contains negation words.

**Expected Answer:**
- [ ] Uses `schema={"answer": bool, "explanation": str}` (dict schema)
- [ ] Uses `kbench.assertions.assert_false(response.answer)`
- [ ] Uses `kbench.assertions.assert_contains_regex(r"(not|never|no|doesn't|didn't)", response.explanation.lower())`
- [ ] Does NOT use plain Python `assert`
- [ ] Has `@kbench.task()` decorator

**Source of Truth:** `documentation/examples/demo_candidates_bm1.py` lines 98–118; Skill file §9 Pattern C

---

### Scenario 5.3 — System Prompt + Structured Output + Code Execution

**Prompt:**
> Write a benchmark task that:
> 1. Sets the LLM as "an expert Python programmer" using a system message.
> 2. Gives the LLM buggy code: `fruits = ['apple', 'orange' 'banana', 'peach']` (missing comma).
> 3. Uses a dataclass with `has_bugs: bool` and `fixed_code: str` as schema.
> 4. Asserts `has_bugs` is True.
> 5. Extracts and runs the fixed code.
> 6. Asserts the output is "4".

**Expected Answer:**
- [ ] Uses `kbench.system.send("You are an expert Python programmer.")`
- [ ] Defines `@dataclass` with `has_bugs: bool` and `fixed_code: str`
- [ ] Uses `llm.prompt("...", schema=CodeAnalysis)`
- [ ] Uses `kbench.assertions.assert_true(response.has_bugs)`
- [ ] Uses `kbench.tools.python.extract_code(response.fixed_code)`
- [ ] Uses `kbench.tools.python.script_runner.run_code(code)`
- [ ] Uses `kbench.assertions.assert_equal("4", output.stdout.strip())`

**Source of Truth:** `documentation/examples/demo_candidates_bm1.py` lines 151–189; Skill file §9 Pattern I

---

### Scenario 5.4 — Custom Assertion + Task

**Prompt:**
> Define a reusable custom assertion `assert_is_palindrome` using `@assertion_handler()` that checks if a string is a palindrome. Then write a task that asks the LLM to generate a palindrome word and validates it with the custom assertion.

**Expected Answer:**
- [ ] Imports `assertion_handler` and `AssertionResult` from `kaggle_benchmarks.assertions`
- [ ] Uses `@assertion_handler()` decorator on custom function
- [ ] Return type annotated as `-> AssertionResult`
- [ ] Returns `AssertionResult(passed=..., expectation=...)`
- [ ] Cleans the string (e.g., `.lower()`, strip spaces) before checking palindrome
- [ ] Uses the custom assertion inside a `@kbench.task()` decorated function
- [ ] Does NOT use `@assertion_handler(raises_assertion_error=True)` unless explicitly needed

**Source of Truth:** `src/kaggle_benchmarks/assertions.py` lines 113–175 (`assertion_handler`); Skill file §5 "Custom Assertions"

---

### Scenario 5.5 — Multi-Turn Game with Judge

**Prompt:**
> Write a benchmark task that plays 20 Questions. The LLM guesses an animal by asking yes/no questions. A judge LLM answers. The game loop runs up to 20 turns. Return True if the LLM guesses correctly.

**Expected Answer:**
- [ ] Task accepts `llm, judge_llm, target: str` parameters
- [ ] Returns `-> bool`
- [ ] Defines a structured output schema with `question` and `guess` fields
- [ ] Has a for/while loop (up to 20 turns)
- [ ] Uses `with kbench.chats.new("Answering"):` to isolate judge's answer
- [ ] Judge uses `schema=bool` for yes/no answers
- [ ] Checks `response.guess` to detect a final guess vs. a question
- [ ] Returns True/False based on whether guess matches target
- [ ] Calls `.run(kbench.llm, kbench.judge_llm, target="dog")`

**Source of Truth:** `documentation/examples/play_20_questions.py` lines 38–76; Skill file §9 Pattern F

---

### Scenario 5.6 — Sub-Tasks Composition with Float Score

**Prompt:**
> Write a benchmark with a root task and two sub-tasks:
> - Sub-task 1: Sends "Hello!" and asserts the response is not empty.
> - Sub-task 2: Tells the LLM a name, then asks "What is my name?", and asserts the name is in the response.
> - Root task: Runs sub-task 1 once and sub-task 2 twice (with "Alan Turing" and "Richard Feynman"), then returns the fraction of sub-tasks that passed as a float.

**Expected Answer:**
- [ ] Defines 2–3 separate `@kbench.task()` functions
- [ ] Root task returns `-> float`
- [ ] Sub-task functions use kbench assertions internally
- [ ] Root task collects runs: `runs = [subtask1.run(llm), subtask2.run(llm, "Alan Turing"), ...]`
- [ ] Root task computes: `sum(r.passed for r in runs) / len(runs)`
- [ ] Calls root task's `.run(kbench.llm)`

**Source of Truth:** `documentation/examples/simple_task.py` lines 24–52

---

### Scenario 5.7 — Code Generation with CSV Filtering

**Prompt:**
> Write a benchmark task that asks the LLM to write Python code that reads a CSV string with columns 'name' and 'age', filters rows where age > 30, and prints the count. The CSV data is: 'name,age\nAlice,25\nBob,35\nCharlie,40\nDiana,28'. Extract and run the code, then assert the output is "2".

**Expected Answer:**
- [ ] Calls `llm.prompt()` with clear instructions including the CSV data
- [ ] Uses `kbench.tools.python.extract_code(response)`
- [ ] Uses `kbench.tools.python.script_runner.run_code(code)`
- [ ] Asserts `result.stderr` is empty
- [ ] Asserts `result.stdout.strip()` equals "2"
- [ ] Uses kbench assertions

**Source of Truth:** `golden_tests/test_code_generator.py` lines 127–130 (`csv-filter-age`)

---

### Scenario 5.8 — Negative Assertion + Content Check

**Prompt:**
> Write a benchmark task that asks the LLM to explain why the sky is blue. Assert that the response:
> 1. Contains "scatter" or "Rayleigh" (scientific explanation).
> 2. Does NOT contain "paint" or "dye" (irrelevant terms).

**Expected Answer:**
- [ ] Uses `kbench.assertions.assert_contains_regex(r"(?i)(scatter|rayleigh)", response)`
- [ ] Uses `kbench.assertions.assert_not_contains_regex(r"(?i)(paint|dye)", response)`
- [ ] Both positive and negative assertions are kbench assertions
- [ ] Has `@kbench.task()` decorator

**Source of Truth:** `golden_tests/test_code_generator.py` lines 132–135 (`why-sky-is-blue`)

---

## Category 6: Knowledge & Troubleshooting

### Scenario 6.1 — Assertions vs Python assert

**Prompt:**
> Should I use Python's built-in `assert` or `kbench.assertions` in my tasks? What's the difference?

**Expected Answer:**
- [ ] Recommends `kbench.assertions` over plain `assert`
- [ ] Explains kbench assertions are *recorded and tracked* in the run results
- [ ] Explains kbench assertions do NOT raise exceptions — execution continues
- [ ] Explains plain `assert` IS caught by the task runner (doesn't crash the program) but is NOT tracked properly
- [ ] Does NOT say plain assert crashes the program entirely

**Source of Truth:** Skill file §5; `src/kaggle_benchmarks/tasks.py` lines 134–135

---

### Scenario 6.2 — Four Schema Styles

**Prompt:**
> What are the different ways to get structured output from an LLM in kaggle-benchmarks, and when should I use each?

**Expected Answer:**
- [ ] Lists 4 approaches: dataclass, dict schema, Pydantic, primitive types
- [ ] Explains **dataclass** — preferred for complex types with multiple fields
- [ ] Explains **dict schema** `{"key": type}` — quick prototyping, simple key-value
- [ ] Explains **Pydantic** — when you need validation rules or `Field()` descriptions
- [ ] Explains **primitive** (`int`, `bool`, `str`) — when you need a single value
- [ ] Shows at least one example for each

**Source of Truth:** `golden_tests/test_cookbook_examples.py` (all `test_extract_*` tests); Skill file §4

---

### Scenario 6.3 — Missing Return Annotation Bug

**Prompt:**
> I wrote this task but it's not showing scores on the leaderboard:
> ```python
> @kbench.task()
> def accuracy(llm):
>     return 0.85
> ```
> What's wrong?

**Expected Answer:**
- [ ] Identifies the missing `-> float` return type annotation
- [ ] Explains the return type is needed for the library to infer the result type
- [ ] Shows the fix: `def accuracy(llm) -> float:`
- [ ] Explains that without annotation, it defaults to PassFail (ignores the returned value)

**Source of Truth:** `src/kaggle_benchmarks/results.py` lines 63–111; `SKILL.md` Common Mistakes

---

### Scenario 6.4 — Judge Returns None

**Prompt:**
> My task crashes with `AttributeError: 'NoneType' object has no attribute 'results'` when using the judge. What's happening?

**Expected Answer:**
- [ ] Explains `assess_response_with_judge()` can return `None` when the judge fails
- [ ] Shows the fix: check `if assessment is None:` before accessing `.results`
- [ ] Shows using `kbench.assertions.assert_fail("Judge failed")` as the fallback
- [ ] Does NOT claim the function always returns a valid object

**Source of Truth:** Skill file §5 "LLM-as-Judge"; `documentation/examples/assess_with_a_judge.py`

---

### Scenario 6.5 — chats.new vs chats.fork

**Prompt:**
> What's the difference between `kbench.chats.new()` and `kbench.chats.fork()`?

**Expected Answer:**
- [ ] Explains `chats.new()` creates an EMPTY conversation (clean slate)
- [ ] Explains `chats.fork()` COPIES the current conversation history into a new one
- [ ] Use case: `new()` for judges or fresh contexts; `fork()` for branching dialogue
- [ ] Mentions `chats.new()` accepts `system_instructions=` parameter

**Source of Truth:** `src/kaggle_benchmarks/chats.py` lines 139 and 158

---

### Scenario 6.6 — Sub-Task Cluttering Leaderboard

**Prompt:**
> I have a helper task used inside my main task, but it's cluttering the leaderboard. How do I hide it?

**Expected Answer:**
- [ ] Recommends `store_task=False` in the sub-task's `@kbench.task()` decorator
- [ ] Shows: `@kbench.task(name="helper", store_task=False)`
- [ ] May also mention `store_run=False`

**Source of Truth:** `documentation/examples/dataset_evaluation.py` line 38; `documentation/examples/simple_task.py` line 27

---

### Scenario 6.7 — evaluate() Parameter: llm Must Be a List

**Prompt:**
> I'm getting an error when running `.evaluate()`. My code is:
> ```python
> results = my_task.evaluate(llm=kbench.llm, evaluation_data=df)
> ```
> What's wrong?

**Expected Answer:**
- [ ] Identifies that `llm=` must be a **list**: `llm=[kbench.llm]`
- [ ] Shows the fix: `my_task.evaluate(llm=[kbench.llm], evaluation_data=df)`
- [ ] Explains that `.evaluate()` supports multiple models, so the parameter is always a list

**Source of Truth:** `documentation/examples/dataset_evaluation.py` line 62; `documentation/examples/quick_start_examples.py` line 270

---

### Scenario 6.8 — Explicit `-> None` Return Type

**Prompt:**
> Is `-> None` a valid return type annotation for a kaggle-benchmarks task? What result type does it produce?

**Expected Answer:**
- [ ] Confirms `-> None` is valid
- [ ] Explains it is equivalent to omitting the return annotation entirely
- [ ] Both produce `PassFail` result type — pass is determined by assertions, not a return value
- [ ] Does NOT say `-> None` causes an error or is unsupported

**Source of Truth:** `src/kaggle_benchmarks/results.py` line 63 (`class PassFail(Result[type(None) | Unknown])`); Skill file §2 Return Types

---

### Scenario 6.9 — Temperature Parameter

**Prompt:**
> How do I make the LLM give more creative/varied responses in a kaggle-benchmarks task?

**Expected Answer:**
- [ ] Recommends using `temperature=` parameter in `llm.prompt()`
- [ ] Notes default temperature is `0` (deterministic)
- [ ] Shows example: `llm.prompt("Write a creative story", temperature=0.7)`
- [ ] Does NOT suggest configuring temperature through a separate config object

**Source of Truth:** `src/kaggle_benchmarks/actors/llms.py` line 180 (`temperature: float = 0`); Skill file §4

---

### Scenario 6.10 — File Structure and Cell Markers

**Prompt:**
> I need to write a kaggle-benchmarks file with two tasks. Should I use a Jupyter notebook (.ipynb) or a Python file (.py)? And how should I structure the file if I need to install a dependency like `pronouncing`?

**Expected Answer:**
- [ ] Recommends a `.py` file (not `.ipynb`)
- [ ] Uses `# %%` cell markers to separate logical sections
- [ ] Shows imports in one cell, each task in its own cell
- [ ] For `!pip install`, uses commented form: `# !pip install -q pronouncing` (preferred for local compatibility)
- [ ] Explains that uncommented `!pip` magics work on Kaggle but NOT when running as a standalone Python file locally
- [ ] Does NOT generate a notebook file

**Source of Truth:** `documentation/examples/potemkin_understanding.py` lines 21–27; Skill file §1 "File Structure: Cell Markers"

---

### Scenario 6.11 — No `if __name__ == "__main__":` Guard

**Prompt:**
> Write a benchmark file with two tasks. The first task asks the LLM "What is Python?" and checks the response mentions "programming". The second task asks "What is 2+2?" and uses `schema=int` to get the answer. Run both tasks at the end of the file.

**Expected Answer:**
- [ ] Both tasks are defined with `@kbench.task()` decorator
- [ ] Both `.run(kbench.llm)` calls are placed at the **module top level** (not inside any guard)
- [ ] Does **NOT** wrap `.run()` or `.evaluate()` inside `if __name__ == "__main__":`
- [ ] Uses `# %%` cell markers to separate sections
- [ ] Uses kbench assertions (not plain `assert`)
- [ ] Second task has `-> int` return type and uses `schema=int`

> **CRITICAL:** The agent must NOT produce `if __name__ == "__main__":` blocks. Benchmark files are notebook-style scripts — all code runs at the top level. This is explicitly documented in `SKILL.md` Key Rules and §1.

**Source of Truth:** `SKILL.md` Key Rules; §1 "File Structure: Cell Markers"

---

## Category 7: Generalization — Domain-Specific Tasks

These test the agent's ability to **combine** basic patterns from the skill files and apply **domain knowledge** to design new benchmarks. No scenario here maps directly to a single skill file example — the agent must generalize.

### Scenario 7.1 — Sentiment Analysis Pipeline (Structured + Dataset Eval)

**Prompt:**
> Design a kaggle-benchmarks evaluation pipeline for sentiment analysis. I have 5 product reviews. For each review, the LLM should extract: `sentiment` (positive/negative/neutral), `confidence` (0-1 float), and `key_reason` (1 sentence). Evaluate all reviews in parallel and report the average confidence for "positive" reviews only.

**Expected Answer:**
- [ ] Defines a dataclass or Pydantic model with `sentiment`, `confidence`, `key_reason`
- [ ] Defines a per-row sub-task with `store_task=False` and `-> dict` return
- [ ] Uses `llm.prompt(review, schema=SentimentResult)` inside the sub-task
- [ ] Creates a `pd.DataFrame` of reviews
- [ ] Main task uses `.evaluate(llm=[llm], evaluation_data=df, n_jobs=...)` 
- [ ] Filters results for positive sentiment and computes mean confidence
- [ ] Returns `-> float` from the main task
- [ ] Uses `# %%` cell markers

**Source of Truth:** Combines Pattern H (dataset eval) + Style 1 (dataclass schema) + sub-task composition from Skill file §3 and §4

---

### Scenario 7.2 — Code Review Benchmark (System Prompt + Structured + Code Execution)

**Prompt:**
> Build a benchmark that tests whether an LLM can review buggy Python code. Give the LLM 3 buggy code snippets. For each, the LLM should return a structured response with `has_bug: bool`, `bug_description: str`, and `fixed_code: str`. Verify each fix by running it and checking expected output.

**Expected Answer:**
- [ ] Uses `kbench.system.send("You are a code reviewer...")` for context
- [ ] Defines a `@dataclass` with `has_bug`, `bug_description`, `fixed_code`
- [ ] Uses a parameterized task or dataset evaluation over the 3 snippets
- [ ] For each snippet: `llm.prompt(code, schema=CodeReview)`
- [ ] Extracts and runs fixed code: `kbench.tools.python.extract_code()` + `run_code()`
- [ ] Asserts `has_bug == True` AND verifies output of fixed code
- [ ] Uses kbench assertions throughout (not plain assert)

**Source of Truth:** Combines Pattern I (code analysis) + dataset eval + code execution from Skill file §3 and §7

---

### Scenario 7.3 — Translation Quality Benchmark (Multi-Model + Judge)

**Prompt:**
> Design a benchmark that compares how well two models translate English to French. Use 3 test sentences. For each sentence, both models translate, and a judge evaluates translation quality. Report per-model average score.

**Expected Answer:**
- [ ] Defines a per-row sub-task that takes `llm, sentence, reference_translation` 
- [ ] Sub-task calls `llm.prompt(f"Translate to French: {sentence}")`
- [ ] Sub-task uses `assess_response_with_judge()` with criteria like "translation is accurate"
- [ ] OR uses a judge LLM in a `chats.new()` with `schema=int` for scoring
- [ ] Main task calls `.evaluate(llm=[kbench.llm, kbench.judge_llm], evaluation_data=df, ...)`
- [ ] Uses `llm=` as a LIST of both models
- [ ] Computes per-model scores from `.as_dataframe()`
- [ ] Returns `-> dict` or `-> float`

**Source of Truth:** Combines multi-model comparison (§3) + judge (§5) + dataset eval (§3)

---

### Scenario 7.4 — Reasoning with Verification (reasoning + structured output + tool)

**Prompt:**
> Write a task that asks the LLM to solve a math word problem step by step using reasoning mode. Extract the final numeric answer using `schema=float`, then verify the answer by running a Python calculation. Compare the LLM's answer to the computed answer.

**Expected Answer:**
- [ ] Uses `llm.prompt(problem, reasoning="medium", schema=float)` or two separate calls
- [ ] Uses `kbench.tools.python.script_runner.run_code()` to compute the ground truth
- [ ] Compares LLM answer to computed answer: `assert_equal` or `assert_true(abs(a-b) < 0.01)`
- [ ] Does NOT hardcode the expected answer — actually computes it
- [ ] Optionally accesses reasoning traces via `kbench.last_reasoning_traces()`

**Source of Truth:** Combines reasoning param (§4) + structured output + code execution (§7)

---

### Scenario 7.5 — FAQ Chatbot Benchmark (Multi-Turn + fork + Negative Testing)

**Prompt:**
> Design a benchmark for a customer support chatbot. First establish context: "You are a support agent for TechCorp. Products: CloudDB ($99/mo), FastAPI ($49/mo), DataPipe ($149/mo)." Then test: 1) Ask about CloudDB pricing — assert correct price. 2) Fork the conversation and ask about a non-existent product — assert the bot says it doesn't exist. 3) Back in the original, ask "What was the first product I asked about?" — assert it remembers CloudDB.

**Expected Answer:**
- [ ] Uses `kbench.system.send("You are a support agent...")` to set context
- [ ] First query: `llm.prompt(...)` + `assert_contains_regex(r"99", response)`
- [ ] Uses `with kbench.chats.fork("edge_case"):` to branch the conversation
- [ ] Inside fork: asks about non-existent product + `assert_not_contains_regex` for prices, or `assert_contains_regex` for "not available/don't have"
- [ ] After fork (back in original): asks follow-up + `assert_contains_regex(r"(?i)clouddb", response)`
- [ ] Demonstrates that fork doesn't pollute original conversation

**Source of Truth:** Combines system prompt (§6) + `chats.fork()` (§6) + negative assertion (§5)

---

### Scenario 7.6 — Structured Data Extraction from Unstructured Text (Complex Schema)

**Prompt:**
> Build a benchmark that extracts structured information from a job posting. The posting is: "Senior ML Engineer at DataCo. Remote. Salary: $150k-$200k. Requirements: 5+ years Python, TensorFlow, PhD preferred. Benefits: Health insurance, 401k match, unlimited PTO." Extract using a nested Pydantic model with `title`, `company`, `salary_range: SalaryRange(min, max)`, `requirements: list[str]`, `benefits: list[str]`, `is_remote: bool`. Assert at least 3 requirements and 2 benefits.

**Expected Answer:**
- [ ] Defines nested Pydantic models: `SalaryRange` and `JobPosting` 
- [ ] Uses `pydantic.Field(description=...)` on fields (at least some)
- [ ] Uses `llm.prompt(posting_text, schema=JobPosting)`
- [ ] Accesses nested fields: `result.salary_range.min`, `result.salary_range.max`
- [ ] Asserts: `assert_true(len(result.requirements) >= 3)`, `assert_true(len(result.benefits) >= 2)`
- [ ] Asserts: `assert_true(result.is_remote)`
- [ ] Uses kbench assertions

**Source of Truth:** Combines composite Pydantic (like scenario 1.7) + `Field()` descriptions (§4)

---

### Scenario 7.7 — LLM Self-Consistency Check (Multiple Isolated Chats)

**Prompt:**
> Design a benchmark that asks the same factual question to the LLM 3 times in separate conversations and checks if all answers are consistent. Use `chats.new()` for isolation. Return `True` if all answers match.

**Expected Answer:**
- [ ] Collects 3 responses, each in `with kbench.chats.new(f"trial_{i}"):` 
- [ ] Uses `schema=` (e.g., `schema=str` or `schema=int`) for deterministic extraction
- [ ] Compares all 3 results: `assert_equal(answers[0], answers[1])`, etc.
- [ ] Returns `-> bool` indicating consistency
- [ ] Uses `temperature=0` (or relies on default 0) for determinism
- [ ] Does NOT use `chats.fork()` — needs clean slates, not copies

**Source of Truth:** Combines `chats.new()` (§6) + structured output (§4) + return types (§2)

---

### Scenario 7.8 — Tool-Augmented Knowledge Retrieval (Tool + Judge + System)

**Prompt:**
> Build a benchmark where the LLM has access to a `lookup_database(query: str) -> str` tool that returns fake knowledge base entries. Ask the LLM "What is our refund policy?" with the tool available. Then use a judge to evaluate whether the response accurately reflects the tool's returned information rather than hallucinating.

**Expected Answer:**
- [ ] Defines `lookup_database` tool with docstring and return type
- [ ] Uses `llm.prompt("What is our refund policy?", tools=[lookup_database])`
- [ ] Uses `assess_response_with_judge()` with criteria checking accuracy against tool output
- [ ] OR uses `chats.new("judge")` + `judge_llm.prompt(...)` for manual judging
- [ ] Task accepts both `llm` and `judge_llm` parameters
- [ ] Checks judge result for None before accessing results
- [ ] Calls `.run(kbench.llm, kbench.judge_llm)`

**Source of Truth:** Combines tool use (§7) + judge (§5) + system prompt (§6)

---

### Scenario 7.9 — Competitive Coding Benchmark (Code Gen + Test Cases + Dataset Eval)

**Prompt:**
> Build a benchmark modeled after competitive programming. The LLM receives a problem description and generates Python code. Run the code against 2 test cases (with stdin input and expected stdout output). Evaluate 3 problems in parallel and report accuracy.

**Expected Answer:**
- [ ] Defines a per-problem sub-task with `store_task=False`
- [ ] Uses `kbench.system.send("Write a Python program to solve the problem.")`
- [ ] Uses `kbench.tools.python.extract_code(response)` to extract solution code
- [ ] Uses `kbench.tools.python.script_runner.run_code(code, input=test_case_input)` for each test case
- [ ] Asserts `result.stdout` matches expected output for each test case
- [ ] Creates a `pd.DataFrame` of problems + test cases
- [ ] Main task uses `.evaluate()` with `n_jobs=3`
- [ ] Returns accuracy as `-> float`

**Source of Truth:** Modeled after `documentation/examples/code_generation.py`; combines code execution (§7) + dataset eval (§3) + system prompt (§6)

---

### Scenario 7.10 — Tricky Questions with LLM-as-Evaluator (chats.new + Manual Judge Pattern)

**Prompt:**
> Build a benchmark that tests if the LLM knows "9.9 is larger than 9.11". After the LLM answers, open a new chat and have a second LLM evaluate whether the first LLM's answer was correct. The evaluator should end with "The answer is correct" or "The answer is incorrect". Assert based on the evaluator's conclusion.

**Expected Answer:**
- [ ] Task accepts `llm` and `eval_llm` (or `judge_llm`) parameters
- [ ] First call: `response = llm.prompt("What is bigger 9.9 or 9.11?")`
- [ ] Uses `with kbench.chats.new("Evaluation"):` to isolate the judge
- [ ] Inside the new chat, constructs a prompt with the question, correct answer, and student's response
- [ ] Uses `eval_llm.prompt(...)` (NOT `assess_response_with_judge`)
- [ ] Asserts evaluator conclusion: `assert_in("answer is correct", eval_response.lower())`
- [ ] Does NOT put both LLM calls in the same conversation

**Source of Truth:** Directly modeled after `documentation/examples/tricky_questions.py` lines 30-51

---

### Scenario 7.11 — Hallucination Competition (Dynamic Data Generation + Eval)

**Prompt:**
> Build an advanced hallucination benchmark. First, ask the LLM to generate 5 questions that are likely to make other LLMs hallucinate (use a Pydantic model with `question: str` and `topic: str`). Then evaluate another model on those questions using a dataset evaluation, where a critic LLM checks if the model spotted the fake. Return the hallucination rate.

**Expected Answer:**
- [ ] Uses a Pydantic model with a list field to generate questions: `schema=QuestionList`
- [ ] Converts generated questions to a `pd.DataFrame`
- [ ] Defines a sub-task that takes `llm, critic, question, topic` and returns `-> bool`
- [ ] Sub-task uses `chats.new()` to isolate the critic's evaluation
- [ ] Main task calls `.evaluate(llm=[...], critic=[critic_llm], evaluation_data=df)`
- [ ] Computes hallucination rate from `.as_dataframe()` results
- [ ] Returns `-> float`

**Source of Truth:** Modeled after `documentation/examples/hallucinations.py` lines 67-82; combines dynamic data generation + structured output + sub-task eval

---

### Scenario 7.12 — Run Object Introspection (Sub-Task + Run Properties)

**Prompt:**
> Write a main task that runs 3 sub-tasks. After running each sub-task, inspect the `Run` object to check if each passed, collect the assertion results, and build a summary dict with `{"total": 3, "passed": N, "failed_tasks": [...]}`. Return the summary as a dict.

**Expected Answer:**
- [ ] Defines sub-tasks with `store_task=False`
- [ ] Calls `run = subtask.run(llm, ...)` for each sub-task
- [ ] Accesses `run.passed` to check pass/fail status
- [ ] Accesses `run.assertion_results` to examine individual assertions
- [ ] Builds a summary dict with total, passed count, failed task names
- [ ] Returns `-> dict` from the main task
- [ ] Uses kbench assertions inside sub-tasks

**Source of Truth:** Run object properties documented in Skill file §3; combines sub-task composition + Run introspection

---

---

## Category 8: Multi-Agent ChatRoom

These scenarios validate the agent's understanding of the `ChatRoom` / `Participant` API for multi-agent conversations with perspective-aware history.

### Scenario 8.1 — Two-Participant Debate (Basic ChatRoom)

**Prompt:**
> Write a kaggle-benchmarks task that runs a 2-turn debate between two LLM participants named "Pro" and "Con" in a `ChatRoom`. The topic is posted by the room: "Should AI labs need mandatory licensing?". Each participant takes two turns. After the room exits, assert the transcript has at least 5 messages (1 topic post + 4 replies).

**Expected Answer:**
- [ ] Uses `kbench.ChatRoom(system_prompt="...")` (not the deprecated `LLMChat.reply()` self-clone pattern)
- [ ] Uses `room.add_participant(llm, name="Pro", system_prompt="...")` and similarly for "Con"
- [ ] Uses `with room:` as a context manager
- [ ] Uses `room.post("Topic: ...")` for the topic (NOT `Participant.post` — that doesn't exist)
- [ ] Calls `pro.reply()` and `con.reply()` inside the `with` block
- [ ] Asserts using `room.messages` (the ground-truth transcript) after the `with` block exits
- [ ] Uses `kbench.assertions.assert_true(len(room.messages) >= 5)`

**Source of Truth:** Skill file §6 "ChatRoom — Multi-Agent Conversations"; `docs/chatroom/rooms_walkthrough.md`

---

### Scenario 8.2 — Same LLM, Many Participants (No Cloning)

**Prompt:**
> I want both "Alice" and "Bob" in my ChatRoom to be backed by the same underlying `kbench.llm` model. Will they interfere with each other? Do I need to clone the LLM?

**Expected Answer:**
- [ ] Says **no cloning is needed**
- [ ] Explains the same `LLMChat` can back multiple participants without interference
- [ ] Explains that the per-room identity (`name`, `avatar`, `system_prompt`) lives on the lightweight `Participant` wrapper, not on the `LLMChat`
- [ ] Shows passing the same `kbench.llm` to two `add_participant()` calls with different `name=` arguments
- [ ] Mentions that participant identity is `is`-based (each `Participant` is a distinct object), which is what lets perspective projection distinguish "my message" from "their message"
- [ ] Does NOT recommend manually copying or cloning the LLM

**Source of Truth:** Skill file §6 "ChatRoom"; `docs/chatroom/rooms_walkthrough.md` "add_participant" section; `docs/chatroom/pr-summary-rooms.md`

---

### Scenario 8.3 — room.post() vs participant.reply()

**Prompt:**
> What's the difference between `room.post(msg)` and `participant.reply()`? When should I use each?

**Expected Answer:**
- [ ] Explains `room.post()` is a **narrator/system directive** (e.g. game rules, phase transitions, topic prompts) — does NOT invoke an LLM
- [ ] Explains `participant.reply()` triggers an **LLM generation** from that participant's perspective
- [ ] Notes that LLMs are explicitly told (via the auto-generated roster) to treat narrator messages as system instructions, not peer speech
- [ ] Mentions these are the **two primitives** that drive all `ChatRoom` interaction
- [ ] Does NOT confuse them or suggest `Participant.post()` exists (it doesn't)

**Source of Truth:** Skill file §6 "ChatRoom"; `docs/chatroom/rooms_walkthrough.md` "post()" section

---

### Scenario 8.4 — One-Shot Private Directive (visible_to)

**Prompt:**
> In a 4-player game, I want to assign each player a secret role at the start (only that player should see their role). Should I use `room.post(visible_to=...)` or `room.private_channel(...)`? Write the code.

**Expected Answer:**
- [ ] Recommends `room.post(msg, visible_to=[player])` for one-shot directives
- [ ] Explains `private_channel` is for **multi-turn** private conversations (overkill for a one-shot role assignment)
- [ ] Shows a loop assigning roles: `room.post(f"Your secret role is X.", visible_to=[player])` for each player
- [ ] Uses `with room:` context
- [ ] Does NOT use `private_channel` for the one-shot case

**Source of Truth:** Skill file §6 "ChatRoom" private information section; `docs/chatroom/rooms_walkthrough.md` "post()" / "private_channel" sections

---

### Scenario 8.5 — Multi-Turn Private Channel (Werewolf Night Phase)

**Prompt:**
> Write code for a Werewolf game's night phase: two wolves (Alice and Bob) discuss privately for 2 turns to pick a victim, while a third player Cleo does NOT see any of their messages. Use a single `ChatRoom` so the daytime context carries over.

**Expected Answer:**
- [ ] Creates the parent `ChatRoom` with all three participants
- [ ] Uses `room.private_channel([alice, bob], name="Wolf Night")` to create the wolves' channel
- [ ] Channel name is required, keyword-only, and **semantically meaningful** (e.g. `"Wolf Night"`, not `"channel1"`)
- [ ] Enters the channel with `with wolves:` (nested or sequential to `with room:`)
- [ ] Has alice and bob each call `reply()` twice in the channel
- [ ] Notes (or implicitly relies on) the fact that Cleo's perspective never includes the wolf messages
- [ ] Does NOT add Cleo to the wolf channel

**Source of Truth:** Skill file §6 "ChatRoom" private channel section; `docs/chatroom/rooms_walkthrough.md` "private_channel" + "Traced Example: Private Channel"

---

### Scenario 8.6 — Hidden-Role Safety (system_prompt is Not Exposed)

**Prompt:**
> In a Werewolf game, I set each participant's secret role via `system_prompt=` when calling `add_participant`. Are other participants able to see each other's `system_prompt` through the auto-generated roster?

**Expected Answer:**
- [ ] Says **NO** — peers' `system_prompt` is never exposed
- [ ] Explains the roster lists **only names** of other participants, never their system prompts
- [ ] Explicitly mentions this is the anti-leak property that makes hidden-role games like Werewolf safe
- [ ] Does NOT suggest a workaround is needed — the default behavior is already safe
- [ ] May mention `visible_to` / `private_channel` as the proper mechanisms for sharing information selectively

**Source of Truth:** Skill file §6 "ChatRoom"; `docs/chatroom/rooms_walkthrough.md` "_build_roster" section

---

### Scenario 8.7 — Removing a Participant Mid-Game

**Prompt:**
> In a game, when a player is eliminated, I call `room.remove_participant(player)`. What happens to (1) future `player.reply()` calls, (2) historical messages from that player, and (3) the player's membership in any private channels they were in?

**Expected Answer:**
- [ ] (1) Says future `player.reply()` calls raise `RuntimeError`
- [ ] (2) Says historical messages **stay** in the transcript, still attributed to the player's name
- [ ] (3) Says removal does **NOT cascade** to private channels — must remove from each channel explicitly
- [ ] Notes peers' rosters on the next turn will no longer list the removed player
- [ ] Does NOT claim historical messages are deleted

**Source of Truth:** Skill file §6 "ChatRoom" hard-delete removal; `docs/chatroom/rooms_walkthrough.md` "remove_participant" section

---

### Scenario 8.8 — Tools Inside reply() — Not Supported

**Prompt:**
> Can I pass `tools=[my_function]` to `participant.reply()` in a `ChatRoom`?

**Expected Answer:**
- [ ] Says **no** — passing `tools=` currently raises `NotImplementedError`
- [ ] Mentions the workaround: use an orphan `chats.new()` side-chat for tool calls
- [ ] Does NOT make up a working `tools=` example for `reply()`
- [ ] May mention this is a known future-work item

**Source of Truth:** Skill file §6 "ChatRoom"; `src/kaggle_benchmarks/rooms.py` `_generate_reply` lines 292-297

---

### Scenario 8.9 — Choosing ChatRoom vs contexts.enter()

**Prompt:**
> I have two LLMs and want them to talk to each other and be aware of each other's responses. Should I use `kbench.ChatRoom` or `contexts.enter()` with two separate chats?

**Expected Answer:**
- [ ] Recommends **`ChatRoom`** — it is the right tool for "multiple LLMs aware of each other"
- [ ] Explains `ChatRoom` provides automatic perspective projection (each LLM sees its own messages as `assistant` and peers' as attributed `user`)
- [ ] Explains `contexts.enter()` with separate chats is for **fully isolated** agents that should NOT see each other
- [ ] Does NOT recommend manual message routing between two isolated chats for this use case

**Source of Truth:** Skill file §6 "ChatRoom" + "contexts.enter()" sections + "Choosing Conversation Strategy" table

---

### Scenario 8.10 — End-to-End ChatRoom Benchmark with Judge

**Prompt:**
> Build a benchmark that runs a 4-message conversation between two LLM participants in a `ChatRoom` on a topic of your choice. After the room exits, have a judge LLM rate the conversation quality 0-10 in an isolated `chats.new()` chat. Return the score as a float.

**Expected Answer:**
- [ ] Uses `kbench.ChatRoom`, `add_participant`, `with room:`, `room.post`, `participant.reply` correctly
- [ ] Has at least 4 `reply()` calls inside the `with room:` block
- [ ] Builds the transcript from `room.messages` after the `with` block exits
- [ ] Uses `with kbench.chats.new("judge"):` to isolate the judge call
- [ ] Calls `judge_llm.prompt(..., schema=float)` (or schema=int) for the rating
- [ ] Task signature accepts both `llm` and `judge_llm` parameters
- [ ] Has `-> float` return type annotation
- [ ] Calls `.run(kbench.llm, kbench.judge_llm)` at module top level (no `__name__` guard)

**Source of Truth:** Skill file §9 "Pattern I.5: Multi-Agent ChatRoom (Debate)"; Skill file §6 "ChatRoom"

---

## Scoring Guide

| Rating | Criteria |
|--------|----------|
| ✅ Strong Pass | All checkboxes for the scenario are met |
| ⚠️ Partial Pass | Most checkboxes met, minor omissions |
| ❌ Fail | Critical checkboxes missed (especially items marked CRITICAL) |

### Summary Table

| # | Scenario | Category | Difficulty | Result |
|---|----------|----------|------------|--------|
| 1.1 | Simple Q&A regex | Basic Task | Basic | |
| 1.2 | Extract int | Basic Structured | Basic | |
| 1.3 | Extract bool | Basic Structured | Basic | |
| 1.4 | Extract dict | Basic Structured | Basic | |
| 1.5 | Extract dataclass | Basic Structured | Basic | |
| 1.6 | Extract pydantic | Basic Structured | Basic | |
| 1.7 | Composite pydantic | Basic Structured | Basic | |
| 1.8 | Multi-turn memory | Basic Conversation | Basic | |
| 1.9 | Simple greeting | Basic Task | Basic | |
| 1.10 | Reasoning param | Basic Task | Basic | |
| 1.11 | Image URL input | Basic Multimodal | Basic | |
| 1.12 | Audio input | Basic Multimodal | Basic | |
| 1.13 | Video input (URL) | Basic Multimodal | Basic | |
| 2.1 | Simple function tool | Basic Tool | Basic | |
| 2.2 | Multiple tools | Basic Tool | Basic | |
| 2.3 | Tool error handling | Basic Tool | Basic | |
| 3.1 | Extract + run code | Basic Code | Basic | |
| 3.2 | Code + subchat combo | Basic Code | Basic | |
| 4.1 | DataFrame eval (bool) | Dataset Eval | Medium | |
| 4.2 | Sub-task + accuracy | Dataset Eval | Medium | |
| 4.3 | Eval with full params | Dataset Eval | Medium | |
| 4.4 | Multi-model comparison | Dataset Eval | Medium | |
| 4.5 | Math word problems | Dataset Eval | Medium | |
| 4.6 | on_failure="continue" | Dataset Eval | Medium | |
| 4.7 | Resilient prod pattern | Dataset Eval | Advanced | |
| 4.8 | Default on_failure behavior | Dataset Eval / Knowledge | Basic | |
| 5.1 | Structured + judge | Combined | Medium | |
| 5.2 | Hallucination detect | Combined | Medium | |
| 5.3 | System + struct + code | Combined | Medium | |
| 5.4 | Custom assertion + task | Combined | Medium | |
| 5.5 | Game loop + judge | Combined | Advanced | |
| 5.6 | Sub-task composition | Combined | Medium | |
| 5.7 | Code gen CSV filter | Combined | Medium | |
| 5.8 | Positive + negative | Combined | Medium | |
| 6.1 | Assertions vs assert | Knowledge | Basic | |
| 6.2 | Four schema styles | Knowledge | Basic | |
| 6.3 | Missing return anno | Troubleshooting | Basic | |
| 6.4 | Judge returns None | Troubleshooting | Basic | |
| 6.5 | new vs fork | Knowledge | Basic | |
| 6.6 | Sub-task leaderboard | Troubleshooting | Basic | |
| 6.7 | llm must be list | Troubleshooting | Basic | |
| 6.8 | -> None return type | Knowledge | Basic | |
| 6.9 | Temperature param | Knowledge | Basic | |
| 6.10 | Cell markers + magics | Knowledge | Basic | |
| 6.11 | No `__name__` guard | Knowledge | Basic | |
| 7.1 | Sentiment pipeline | Generalization | Advanced | |
| 7.2 | Code review benchmark | Generalization | Advanced | |
| 7.3 | Translation quality | Generalization | Advanced | |
| 7.4 | Reasoning + verify | Generalization | Advanced | |
| 7.5 | FAQ chatbot + fork | Generalization | Advanced | |
| 7.6 | Job posting extraction | Generalization | Advanced | |
| 7.7 | Self-consistency check | Generalization | Advanced | |
| 7.8 | Tool + judge retrieval | Generalization | Advanced | |
| 7.9 | Competitive coding | Generalization | Advanced | |
| 7.10 | Tricky questions eval | Generalization | Advanced | |
| 7.11 | Hallucination competition | Generalization | Advanced | |
| 7.12 | Run introspection | Generalization | Advanced | |
| 8.1 | Two-participant debate | ChatRoom | Basic | |
| 8.2 | Same LLM, many participants | ChatRoom | Knowledge | |
| 8.3 | room.post vs participant.reply | ChatRoom | Knowledge | |
| 8.4 | visible_to for secret roles | ChatRoom | Medium | |
| 8.5 | Private channel (Werewolf night) | ChatRoom | Medium | |
| 8.6 | Hidden-role safety | ChatRoom | Knowledge | |
| 8.7 | remove_participant semantics | ChatRoom | Knowledge | |
| 8.8 | Tools in reply() not supported | ChatRoom | Troubleshooting | |
| 8.9 | ChatRoom vs contexts.enter | ChatRoom | Knowledge | |
| 8.10 | End-to-end ChatRoom + judge | ChatRoom | Advanced | |
