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

import pandas as pd

import kaggle_benchmarks as kbench
from kaggle_benchmarks.kaggle.serialization import (
    generate_run_filename,
    merge_results_from_runfiles,
)

df = pd.DataFrame(
    [
        ("What is the capital of Singapore", "Singapore"),
        ("What is the capital of France", "Paris"),
    ],
    columns=["question", "answer"],
)


def calculate_mean_score(run_results: list[dict]) -> float:
    print(run_results[0])
    scores = [result["numericResult"]["value"] for result in run_results]
    mean_score = sum(scores) / len(scores)
    return mean_score


@kbench.task("q and a")
def q_and_a(llm, question, answer) -> float:
    response = llm.prompt(question)
    with kbench.chats.new("judge"):
        score = kbench.judge_llm.prompt(
            f"""My question: {question}
            My expected answer: {answer}
            LLM answer: {response}
            Given the above information, give a single score from 0 to 10 on how similar the LLM answer is to my expected answer.
            Just output the single score but nothing else.
            """
        )

    return float(score)


# %%
runs = q_and_a.evaluate(llm=[kbench.llm], evaluation_data=df)
run_files = [generate_run_filename(run.task.name, run.id) for run in runs]

# Merge runfiles without deleting them
merge_results_from_runfiles(
    run_files,
    calculate_mean_score,
    output_run_id="Run aggregated",
    delete_run_files=False,
)

# Merge runfiles and delete them after merge
merge_results_from_runfiles(
    run_files,
    calculate_mean_score,
    output_run_id="Run aggregated and delete",
    delete_run_files=True,
)

# %%
