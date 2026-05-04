# Import Styles, Defining Tasks, and Running Tasks

## 1. Import Styles

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

> **Rule:** Do NOT wrap `.run()` or `.evaluate()` calls inside `if __name__ == "__main__":`. Benchmark files are notebook-style scripts — all code runs at the top level. Place `.run()` / `.evaluate()` directly after the task definition, in the same cell or the next cell.

---

## 2. Defining Tasks

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

## 3. Running Tasks

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
