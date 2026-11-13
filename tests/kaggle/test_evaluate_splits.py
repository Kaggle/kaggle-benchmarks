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

import json
import math

import pandas as pd
import pytest

import kaggle_benchmarks as kbench
from kaggle_benchmarks import actors, assertions, kaggle, task
from tests.mocks import MockedChat

CANARY = "s3cr3t-canary"

# The model answers "yes" to everything, so a row passes if its label is "yes".
PUBLIC = pd.DataFrame({"label": ["yes", "no"]})
PRIVATE = pd.DataFrame({"label": ["yes", CANARY, CANARY, CANARY]})


@pytest.fixture
def client(monkeypatch, tmp_path):
    kaggle_client = kaggle.KaggleClient(directory=tmp_path)
    monkeypatch.setattr("kaggle_benchmarks.client", kaggle_client)
    return kaggle_client


@pytest.fixture
def llm():
    return MockedChat.from_contents(["yes"], cycle=True)


@task()
def row(llm, label: str):
    assertions.assert_equal(label, llm.prompt(f"Is it {label}?"))


def run_splits(llm, row_task=row, **kwargs):
    @task()
    def parent(llm) -> kbench.SplitScores:
        return row_task.evaluate_splits(
            public=PUBLIC, private=PRIVATE, llm=[llm], **kwargs
        )

    return parent.run(llm=llm)


def published(client) -> str:
    private = client.directory / "private"
    return "".join(
        p.read_text()
        for p in client.directory.rglob("*")
        if p.is_file() and private not in p.parents
    )


def private_run(client) -> dict:
    (private_file,) = client.directory.glob("private/Parent-*.private.json")
    return json.loads(private_file.read_text())


def test_publishes_the_scores_and_no_private_row(client, llm):
    run = run_splits(llm)

    assert run.result == kbench.SplitScores(overall=2 / 6, public=0.5, private=0.25)
    assert CANARY not in published(client)
    private = private_run(client)
    assert len(private["subruns"]) == 4
    assert CANARY in json.dumps(private)
    # Written once the private half is over, with the scores.
    assert private["state"] == "BENCHMARK_TASK_RUN_STATE_COMPLETED"
    assert len(private["results"]) == 3
    (parent_file,) = client.directory.glob("Parent-*.run.json")
    results = json.loads(parent_file.read_text())["results"]
    assert [(r["type"], r["numericResult"]["value"]) for r in results] == [
        ("AGGREGATED", 2 / 6),
        ("PUBLIC", 0.5),
        ("PRIVATE", 0.25),
    ]
    assert len(run.subruns) == 2
    assert len(list(client.directory.glob("Row-*.run.json"))) == 2


@task()
def passes_or_raises(llm, label: str):
    if label != "yes":
        raise ValueError(label)


@task()
def half_or_raises(llm, label: str) -> float:
    if label != "yes":
        raise ValueError(label)
    return 0.5


@task()
def one_or_nan(llm, label: str) -> float:
    return 1.0 if label == "yes" else math.nan


@pytest.mark.parametrize(
    "row_task, scale",
    [(passes_or_raises, 1.0), (half_or_raises, 0.5), (one_or_nan, 1.0)],
)
def test_a_row_with_no_score_scores_zero_and_stays_in_the_count(llm, row_task, scale):
    run = run_splits(llm, row_task=row_task, on_failure="continue")

    assert run.result == kbench.SplitScores(
        overall=scale * 2 / 6, public=scale / 2, private=scale / 4
    )


def test_a_bool_row_scores_the_same_from_the_cache(client, llm):
    """Its returned bool is what the cache stores, even if an assertion failed."""

    @task()
    def returns_bool(llm, label: str) -> bool:
        assertions.assert_equal(label, "yes")
        return label != CANARY

    fresh = run_splits(llm, row_task=returns_bool)
    client.use_cache = True
    cached = run_splits(llm, row_task=returns_bool)

    assert all(subrun.cached for subrun in cached.subruns)
    assert fresh.result == cached.result
    assert fresh.result == kbench.SplitScores(overall=3 / 6, public=1.0, private=0.25)


def test_split_task_is_not_loaded_from_the_cache(client, llm):
    run_splits(llm)
    client.use_cache = True
    run = run_splits(llm)

    assert not run.cached
    # A cached row scores its stored verdict.
    assert all(subrun.cached for subrun in run.subruns)
    assert run.result == kbench.SplitScores(overall=2 / 6, public=0.5, private=0.25)


@task()
def raises_on_canary(llm, label: str):
    if label == CANARY:
        raise ValueError(label)


def test_a_failed_private_half_is_in_the_private_file(client, llm):
    with pytest.raises(RuntimeError, match="withheld"):
        run_splits(llm, row_task=raises_on_canary)

    private = private_run(client)
    assert private["state"] == "BENCHMARK_TASK_RUN_STATE_ERRORED"
    # The error that left the scope, and the row's own.
    assert CANARY in private["errorMessage"]
    assert any(CANARY in row.get("errorMessage", "") for row in private["subruns"])
    assert CANARY not in published(client)


def test_a_private_file_that_cannot_be_written_costs_nothing_else(
    client, llm, caplog, monkeypatch
):
    def fail(run, parent):
        raise OSError(f"cannot write {CANARY}")

    monkeypatch.setattr(client, "store_private_run", fail)

    run = run_splits(llm)

    assert run.result == kbench.SplitScores(overall=2 / 6, public=0.5, private=0.25)
    assert len(list(client.directory.glob("Parent-*.run.json"))) == 1
    # Only the type of the error: its message can quote a private row.
    assert "Could not write the private run file: OSError" in caplog.text
    assert CANARY not in caplog.text


def test_rows_returning_nan_score_zero_and_are_left_out_of_the_files(client, llm):
    """A run file cannot hold NaN, so one linked NaN row would cost the file."""

    @task()
    def nan_unless_yes(llm, label: str) -> float:
        return 1.0 if label == "yes" else math.nan

    run = run_splits(llm, row_task=nan_unless_yes)

    assert run.result == kbench.SplitScores(overall=2 / 6, public=0.5, private=0.25)
    (parent_file,) = client.directory.glob("Parent-*.run.json")
    parent = json.loads(parent_file.read_text())
    assert len(parent["results"]) == 3
    assert len(parent["subruns"]) == 1
    assert len(private_run(client)["subruns"]) == 1


def test_writing_the_private_file_warns_without_quoting_a_row(client, llm, recwarn):
    class Plain:
        def __init__(self, label):
            self.label = label

        def __repr__(self):
            return f"Plain({self.label})"

    @task()
    def sends_an_object(llm, label: str):
        actors.user.send(Plain(label))

    run_splits(llm, row_task=sends_an_object)

    assert len(private_run(client)["subruns"]) == 4
    assert recwarn  # The serializer warns that the object has no proper form.
    assert not any(CANARY in str(warning.message) for warning in recwarn)


def test_no_private_file_for_a_task_that_stores_no_run(client, llm):
    @task(store_run=False)
    def parent(llm) -> kbench.SplitScores:
        return row.evaluate_splits(public=PUBLIC, private=PRIVATE, llm=[llm])

    parent.run(llm=llm)

    assert not (client.directory / "private").exists()
