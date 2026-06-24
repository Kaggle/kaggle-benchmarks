# Copyright 2025 Kaggle Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
import pandas as pd

import kaggle_benchmarks as kbench

# %%
# Dataset to evaluate on.
df = pd.DataFrame(
    [
        {
            "question": "What's the capital of Singapore",
            "answer": "Singapore",
        },
        {
            "question": "What's the capital of France",
            "answer": "Paris",
        },
    ]
)


# %%
# First define a task to evaluate on a single row of dataset.
@kbench.task(name="single_qa_task", store_task=False)
def single_qa_task(llm, question, answer) -> dict:
    response = llm.prompt(question)
    return {
        "question": question,
        "gold_target": answer,
        "predicted_answer": response,
        "is_correct": answer.lower() in response.lower(),
    }


# test with a single row.
# single_qa_task.run(kbench.llm, **df.iloc[0].to_dict())


# %%
# Define a task to evaluate the whole dataset.
@kbench.task(name="multi_qa_task")
def multi_qa_task(llm, df) -> tuple[float, float]:
    with kbench.client.enable_cache():
        runs = single_qa_task.evaluate(
            stop_condition=lambda runs: len(runs) == df.shape[0],
            max_attempts=1,
            retry_delay=15,
            llm=[llm],
            evaluation_data=df,
            n_jobs=2,
            timeout=120,
            remove_run_files=True,  # Optionally remove sub runs files.
        )
    eval_df = runs.as_dataframe()

    # Use float() to convert from np.float.
    accuracy = float(eval_df.result.str.get("is_correct").mean())
    std = float(eval_df.result.str.get("is_correct").std())
    return accuracy, std


run = multi_qa_task.run(kbench.llm, df)
run

# %%
# Evaluate multiple models.
llms = [kbench.llm, kbench.judge_llm]
n_total = len(llms) * df.shape[0]
with kbench.client.enable_cache():
    runs = single_qa_task.evaluate(
        stop_condition=lambda runs: len(runs) == n_total,
        max_attempts=1,
        retry_delay=15,
        llm=llms,
        evaluation_data=df,
        n_jobs=4,
        timeout=120,
        remove_run_files=True,  # Optionally remove sub runs files.
    )


eval_df = runs.as_dataframe()
eval_df

# %%
# Resilient evaluation pattern for large/flaky datasets.
#
# For production-scale evals where transient per-sample failures (API
# timeouts, rate limits, 5xx) are expected, pair `on_failure="continue"`
# with `max_attempts > 1` and `enable_cache()`. Behavior:
#   - Successful samples persist to disk (state=COMPLETED).
#   - Failed samples persist to disk (state=ERRORED).
#   - Retries skip cached successes and re-run only the errored samples.
#   - Results from all attempts merge into one Runs in row order.
with kbench.client.enable_cache():
    results = single_qa_task.evaluate(
        llm=[kbench.llm],
        evaluation_data=df,
        on_failure="continue",  # collect failures instead of raising
        max_attempts=3,  # retry transient failures up to twice
        retry_delay=5,
        n_jobs=2,
        timeout=120,
    )

# Split the merged result by status.
print(f"Completed: {len(results.completed_runs)} / {len(results)}")
print(f"Errored:   {len(results.errored_runs)}")
for run in results.errored_runs:
    print(f"  {run.params}: {run.error_message[:200]}")

# Aggregate over completed runs ONLY. Failed runs carry the
# `results.FAILED` sentinel which would break .mean() / .sum().
completed_df = results.completed_runs.as_dataframe()
if len(completed_df) > 0:
    accuracy = float(completed_df.result.str.get("is_correct").mean())
    print(f"Accuracy over completed samples: {accuracy:.4f}")

# %%
