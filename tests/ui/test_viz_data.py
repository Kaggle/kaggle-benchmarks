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

import dataclasses
import math

import pandas as pd
import pytest

from kaggle_benchmarks import runs, tasks
from kaggle_benchmarks.ui.viz.data import (
    LeaderboardData,
    metric_direction,
)
from kaggle_benchmarks.usage import Usage


def _data():
    df = pd.DataFrame(
        {
            "model": ["A", "B", "C"],
            "score": [0.9, 0.8, 0.7],
            "cost_usd": [0.10, 0.02, 0.05],
            "latency_ms": [1200, 400, 700],
        }
    )
    task_scores = {
        "A": {0: 1.0, 1: 1.0, 2: 0.0},
        "B": {0: 1.0, 1: 0.0, 2: 0.0},
        "C": {0: 1.0, 1: 1.0, 2: 1.0},
    }
    return LeaderboardData.from_dataframe(
        df, task_scores=task_scores, benchmark_name="Demo"
    )


def test_metric_direction_defaults():
    assert metric_direction("score") == "max"
    assert metric_direction("cost_usd") == "min"
    assert metric_direction("latency_ms") == "min"
    assert metric_direction("totally_unknown_metric") == "max"


def test_from_dataframe_metrics_and_models():
    data = _data()
    assert data.models == ["A", "B", "C"]
    assert set(data.metric_names) == {"score", "cost_usd", "latency_ms"}
    assert data.metrics["score"]["A"] == 0.9


def test_from_dataframe_skips_non_numeric_and_nan():
    df = pd.DataFrame(
        {
            "model": ["A", "B"],
            "score": [0.9, float("nan")],
            "label": ["x", "y"],  # non-numeric, should be excluded
        }
    )
    data = LeaderboardData.from_dataframe(df)
    assert data.metric_names == ["score"]
    # NaN value for B is dropped, leaving only A.
    assert data.metrics["score"] == {"A": 0.9}


def test_default_axes_prefers_cost_vs_quality():
    data = _data()
    x, y = data.default_axes()
    assert x == "cost_usd"  # resource metric preferred for x
    assert y == "score"  # quality metric for y


def test_values_aligns_to_models_with_none_for_missing():
    df = pd.DataFrame({"model": ["A", "B"], "score": [0.9, 0.8]})
    data = LeaderboardData.from_dataframe(df)
    # metric only present for A
    data.metrics["cost_usd"] = {"A": 0.1}
    assert data.values("cost_usd") == [0.1, None]


def test_as_dataframe_roundtrip():
    data = _data()
    frame = data.as_dataframe()
    assert list(frame.index) == ["A", "B", "C"]
    assert frame.loc["A", "score"] == 0.9


def test_task_matrix_shape():
    data = _data()
    matrix = data.task_matrix()
    assert list(matrix.index) == ["A", "B", "C"]
    assert list(matrix.columns) == [0, 1, 2]
    assert matrix.loc["C", 2] == 1.0
    assert matrix.loc["A", 2] == 0.0


def test_win_rate_matrix_values():
    data = _data()
    wr = data.win_rate_matrix()
    # C beats A: shared tasks {0,1,2}; C>=A everywhere, strictly wins task 2 ->
    # C wins task2, ties tasks 0 and 1 -> (1 + 0.5 + 0.5)/3 = 0.6667
    assert wr.loc["C", "A"] == pytest.approx(2.0 / 3.0)
    assert wr.loc["A", "C"] == pytest.approx(1.0 / 3.0)
    # diagonal is NaN
    assert math.isnan(wr.loc["A", "A"])


def test_win_rate_matrix_empty_without_task_scores():
    df = pd.DataFrame({"model": ["A", "B"], "score": [0.9, 0.8]})
    data = LeaderboardData.from_dataframe(df)
    assert data.win_rate_matrix().empty


def test_default_axes_raises_without_metrics():
    data = LeaderboardData(models=["A"], metrics={})
    with pytest.raises(ValueError):
        data.default_axes()


# ---- from_runs ------------------------------------------------------------- #


def _make_run(model_chat, task, param_id, passed, usage=None):
    from kaggle_benchmarks import chats

    chat = None
    if usage is not None:
        chat = chats.Chat(name="c")
        # stub the usage property via a lightweight object
        chat = _ChatWithUsage(usage)
    run = runs.Run(
        task=task,
        params={"llm": model_chat},
        param_id=param_id,
    )
    run.status = _success_status()
    run._forced_passed = passed
    if chat is not None:
        run.chat = chat
    return run


@dataclasses.dataclass
class _ChatWithUsage:
    _usage: Usage

    @property
    def usage(self):
        return self._usage

    history = ()


def _success_status():
    from kaggle_benchmarks import utils

    return utils.Status.SUCCESS


class _Model:
    def __init__(self, name):
        self.name = name


def test_from_runs_aggregates_scalar_metrics(monkeypatch):
    task = tasks.Task(name="t", func=lambda: None)

    model_a = _Model("gpt-x")
    model_b = _Model("claude-y")

    r1 = runs.Run(task=task, params={"llm": model_a}, param_id=0)
    r2 = runs.Run(task=task, params={"llm": model_a}, param_id=1)
    r3 = runs.Run(task=task, params={"llm": model_b}, param_id=0)

    # Force passed values deterministically.
    monkeypatch.setattr(type(r1), "passed", property(lambda self: self.param_id == 0))

    r1.chat = _ChatWithUsage(
        Usage(
            input_tokens=100,
            output_tokens=50,
            input_tokens_cost_nanodollars=1_000_000_000,
            output_tokens_cost_nanodollars=1_000_000_000,
            total_backend_latency_ms=500,
        )
    )
    r2.chat = _ChatWithUsage(
        Usage(
            input_tokens=200,
            output_tokens=100,
            input_tokens_cost_nanodollars=2_000_000_000,
            output_tokens_cost_nanodollars=0,
            total_backend_latency_ms=700,
        )
    )
    r3.chat = _ChatWithUsage(
        Usage(
            input_tokens=100,
            output_tokens=50,
            total_backend_latency_ms=300,
        )
    )

    collection = runs.Runs([r1, r2, r3])
    data = LeaderboardData.from_runs(collection, benchmark_name="B")

    assert data.models == ["gpt-x", "claude-y"]
    # model A: passed on param_id 0 only -> score 0.5
    assert data.metrics["score"]["gpt-x"] == pytest.approx(0.5)
    # model A cost: (2.0 + 2.0)/2 dollars = 2.0
    assert data.metrics["cost_usd"]["gpt-x"] == pytest.approx(2.0)
    # latency mean for A: (500 + 700)/2
    assert data.metrics["latency_ms"]["gpt-x"] == pytest.approx(600)
    # per-task scores captured
    assert data.task_scores["gpt-x"] == {0: 1.0, 1: 0.0}


def test_from_runs_empty():
    data = LeaderboardData.from_runs(runs.Runs([]), benchmark_name="Empty")
    assert data.models == []
    assert data.metric_names == []
