# Copyright 2026 Kaggle Inc.
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

"""Dataset-evaluation benchmark tasks, and their golden tests.

Each task fans a per-row sub-task across a small ``DataFrame`` with
``Task.evaluate`` and returns an aggregate tuple, so these tests assert on
``run.result`` rather than only on ``run.passed``.

Each task is followed by its tests: a scripted one that replays canned responses
through ``fake(...)`` and runs with no API key, and a live one parametrized over
a model pool, which skips when no provider is configured.
"""

import pandas as pd
import pytest
from models import ALL_MODELS, REFERENCE_MODELS, fake

import kaggle_benchmarks as kbench

df = pd.DataFrame(
    [
        {"question": "What's the capital of Singapore", "answer": "Singapore"},
        {"question": "What's the capital of France", "answer": "Paris"},
    ]
)


@kbench.task(name="single_qa_task", store_task=False)
def single_qa_task(llm, question, answer) -> dict:
    response = llm.prompt(question)
    return {
        "question": question,
        "gold_target": answer,
        "predicted_answer": response,
        "is_correct": answer.lower() in response.lower(),
    }


@kbench.task(name="dataset_eval")
def dataset_eval(llm, df) -> tuple[float, float]:
    with kbench.client.enable_cache():
        runs = single_qa_task.evaluate(
            llm=[llm],
            evaluation_data=df,
            n_jobs=2,
            remove_run_files=True,
        )

    eval_df = runs.as_dataframe()

    accuracy = float(eval_df.result.str.get("is_correct").mean())
    std = float(eval_df.result.str.get("is_correct").std())

    return accuracy, std


def test_dataset_eval_scripted():
    # The single cycled response satisfies BOTH rows ("Singapore" and "Paris"),
    # so the accuracy tuple is independent of row order and n_jobs parallelism.
    run = dataset_eval.run(fake(["Singapore and Paris."], cycle=True), df=df)
    assert run.passed
    accuracy, std = run.result
    assert accuracy == pytest.approx(1.0)
    assert std == pytest.approx(0.0)


@pytest.mark.parametrize("llm", ALL_MODELS)
def test_dataset_eval(llm):
    run = dataset_eval.run(llm, df=df)
    assert run.passed
    accuracy, std = run.result
    assert accuracy == pytest.approx(1.0)
    assert std == pytest.approx(0.0)


@kbench.task(name="dataset_eval_with_failure")
def dataset_eval_with_failure(llm, df) -> tuple[int, int]:
    @kbench.task(name="qa_with_failure", store_task=False)
    def qa_with_failure(llm, question, answer) -> dict:
        response = llm.prompt(question)
        # Intentionally fail the first sample to exercise the failure path.
        if "Singapore" in answer:
            raise ValueError(f"Intentional failure for sample: {answer}")
        return {"question": question, "predicted_answer": response}

    results = qa_with_failure.evaluate(
        llm=[llm],
        evaluation_data=df,
        n_jobs=1,
        on_failure="continue",
    )

    return len(results.completed_runs), len(results.errored_runs)


def test_dataset_eval_with_failure_scripted():
    # This task deliberately produces an errored sub-run, so assert on status +
    # result rather than on run.passed.
    run = dataset_eval_with_failure.run(fake(["Somewhere."], cycle=True), df=df)
    assert run.status == kbench.utils.Status.SUCCESS
    completed, errored = run.result
    assert completed == 1  # one of two samples intentionally raises
    assert errored == 1


@pytest.mark.parametrize("llm", REFERENCE_MODELS)
def test_dataset_eval_with_failure(llm):
    run = dataset_eval_with_failure.run(llm, df=df)
    assert run.status == kbench.utils.Status.SUCCESS
    completed, errored = run.result
    assert completed == 1  # one of two samples intentionally raises
    assert errored == 1
