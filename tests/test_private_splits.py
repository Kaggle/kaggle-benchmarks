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

"""Evaluating a task over a public and a private half."""

import pandas as pd
import pytest

from kaggle_benchmarks import privacy, tasks
from kaggle_benchmarks.kaggle import benchmark_types_pb2 as types
from kaggle_benchmarks.kaggle import serialization
from kaggle_benchmarks.runs import Split

PUBLIC = pd.DataFrame({"value": [1, 2]})
PRIVATE = pd.DataFrame({"value": [10, 20]})


@pytest.fixture(autouse=True)
def enabled(cfg):
    """The entry point is behind a flag until the backend enforces the split."""
    cfg.enable_private_splits = True


@tasks.task(name="Doubler", store_task=False, store_run=False)
def doubler(value: int = 0) -> float:
    return float(value * 2)


def evaluate_split(task=doubler, **kwargs):
    return task.evaluate(
        evaluation_data=PUBLIC, private_evaluation_data=PRIVATE, **kwargs
    )


# -- The flag --------------------------------------------------------------


def test_the_entry_point_is_off_by_default(cfg):
    cfg.enable_private_splits = False

    with pytest.raises(NotImplementedError, match="ENABLE_PRIVATE_SPLITS"):
        evaluate_split()


# -- Running both halves ---------------------------------------------------


def test_each_row_is_labelled_with_its_half():
    result = evaluate_split()

    assert [r.split for r in result.public] == [Split.PUBLIC] * 2
    assert [r.split for r in result.private] == [Split.PRIVATE] * 2
    assert [r.result for r in result.private] == [20.0, 40.0]


def test_the_marker_never_reaches_the_task():
    seen = []

    @tasks.task(name="Recorder", store_task=False, store_run=False)
    def recorder(**kwargs) -> float:
        seen.append(dict(kwargs))
        return 1.0

    evaluate_split(recorder)

    assert seen == [{"value": v} for v in (1, 2, 10, 20)]


def test_a_task_may_have_a_split_column_of_its_own():
    """Which is why the argument is `_split`: columns arrive as keywords."""

    @tasks.task(name="Splitter", store_task=False, store_run=False)
    def splitter(split: str = "", value: int = 0) -> float:
        return float(value)

    result = splitter.evaluate(
        evaluation_data=pd.DataFrame({"split": ["train"], "value": [1]}),
        private_evaluation_data=pd.DataFrame({"split": ["test"], "value": [2]}),
    )

    assert [r.params["split"] for r in result] == ["train", "test"]
    assert [r.split for r in result] == [Split.PUBLIC, Split.PRIVATE]


def test_public_row_ids_do_not_move_when_a_private_half_is_added():
    """The id becomes the cache filename, so it has to stay put."""
    public = pd.DataFrame({"value": [1, 2]}, index=[7, 9])

    alone = doubler.evaluate(evaluation_data=public)
    with_private = doubler.evaluate(
        evaluation_data=public, private_evaluation_data=PRIVATE
    )

    assert [r.cache_id for r in with_private.public] == [r.cache_id for r in alone]
    assert [r.param_id for r in with_private.private] == ["private_0", "private_1"]


# -- Refusing bad input ----------------------------------------------------


@pytest.mark.parametrize(
    "public, private, message",
    [
        (PUBLIC, [{"value": 1}], "must be a pandas DataFrame"),
        (None, PRIVATE, "needs evaluation_data alongside it"),
        (PUBLIC, pd.DataFrame({"other": [1]}), "must have the same columns"),
        (PUBLIC, PRIVATE.iloc[0:0], "is empty"),
        (
            PUBLIC.assign(**{tasks.SPLIT_COLUMN: 1}),
            PRIVATE.assign(**{tasks.SPLIT_COLUMN: 1}),
            "reserved",
        ),
    ],
)
def test_bad_input_is_refused_up_front(public, private, message):
    with pytest.raises((TypeError, ValueError), match=message):
        doubler.evaluate(evaluation_data=public, private_evaluation_data=private)


def test_a_task_that_cannot_be_scored_is_refused_before_it_runs():
    ran = []

    @tasks.task(name="Dicty", store_task=False, store_run=False)
    def dicty(value: int = 0) -> dict:
        ran.append(value)
        return {"score": float(value)}

    with pytest.raises(ValueError, match="Pass aggregate="):
        evaluate_split(dicty)

    assert ran == []


# -- Scoring ---------------------------------------------------------------


def test_each_half_is_scored_on_its_own():
    assert evaluate_split().split_scores == {Split.PUBLIC: 3.0, Split.PRIVATE: 30.0}


def test_the_scores_are_recorded_on_the_calling_run():
    @tasks.task(name="Wrapper", store_task=False, store_run=False)
    def wrapper() -> float:
        rows = evaluate_split()
        return sum(r.result for r in rows) / len(rows)

    run = wrapper.run()

    assert run.result == pytest.approx(16.5)
    assert run.split_scores == {Split.PUBLIC: 3.0, Split.PRIVATE: 30.0}


def test_the_private_score_is_masked_in_the_repr():
    text = repr(evaluate_split())

    assert "3.0" in text
    assert privacy.MASK in text.split("split_scores=")[-1]


# -- What gets written -----------------------------------------------------


def test_the_two_halves_are_written_after_the_run_s_own_result():
    @tasks.task(name="SerWrapper", store_task=False, store_run=False)
    def wrapper() -> float:
        rows = evaluate_split()
        return sum(r.result for r in rows) / len(rows)

    written = serialization.prepare_run(wrapper.run())["results"]

    assert written == [
        {
            "numeric_result": {"value": 16.5},
            "type": types.BenchmarkTaskRunResultType.AGGREGATED,
        },
        {
            "numeric_result": {"value": 3.0},
            "type": types.BenchmarkTaskRunResultType.PUBLIC,
        },
        {
            "numeric_result": {"value": 30.0},
            "type": types.BenchmarkTaskRunResultType.PRIVATE,
        },
    ]


def test_a_run_without_a_split_is_written_exactly_as_before():
    """Two readers take results[0]: the cache loader and the ATIF converter."""
    written = serialization.prepare_run(doubler.run(value=3))["results"]

    assert written == [
        {
            "numeric_result": {"value": 6.0},
            "type": types.BenchmarkTaskRunResultType.AGGREGATED,
        }
    ]


def test_the_converter_reads_the_halves_as_prefixed_rewards():
    """The handshake with the harbor converter, which already knew the types."""
    from google.protobuf import json_format

    atif = pytest.importorskip("kaggle_benchmarks.kaggle.atif")

    @tasks.task(name="AtifWrapper", store_task=False, store_run=False)
    def wrapper() -> float:
        rows = evaluate_split()
        return sum(r.result for r in rows) / len(rows)

    run_json = json_format.MessageToDict(serialization.dump_run(wrapper.run()))

    assert atif._rewards(run_json, []) == {
        "score": 16.5,
        "public_score": 3.0,
        "private_score": 30.0,
    }


# -- Failures --------------------------------------------------------------


def test_the_failure_summary_does_not_quote_a_private_row(cfg):
    """The batch path, where failures are collected rather than raised."""
    cfg.continue_with_exceptions = True

    @tasks.task(name="Exploder", store_task=False, store_run=False)
    def exploder(value: int = 0) -> float:
        if value > 5:
            raise ValueError(f"boom with {value}")
        return float(value)

    with pytest.raises(RuntimeError) as excinfo:
        exploder.evaluate(
            evaluation_data=pd.DataFrame({"value": [1]}),
            private_evaluation_data=pd.DataFrame({"value": [77]}),
        )

    assert "77" not in str(excinfo.value)
    assert privacy.HIDDEN_BODY in str(excinfo.value)
