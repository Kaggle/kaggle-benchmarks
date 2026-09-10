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

import pandas as pd
import pytest
from bokeh.plotting import figure

from kaggle_benchmarks.ui.viz import charts
from kaggle_benchmarks.ui.viz.data import LeaderboardData


@pytest.fixture()
def data():
    df = pd.DataFrame(
        {
            "model": ["A", "B", "C"],
            "score": [0.9, 0.8, 0.7],
            "cost_usd": [0.10, 0.02, 0.05],
            "elo": [1500, 1450, 1400],
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


def test_available_charts(data):
    available = charts.available_charts(data)
    assert set(available) == {"bars", "scatter", "heatmap", "winrate", "elo"}


def test_available_charts_no_task_data():
    df = pd.DataFrame({"model": ["A", "B"], "score": [0.9, 0.8]})
    data = LeaderboardData.from_dataframe(df)
    available = charts.available_charts(data)
    assert "heatmap" not in available
    assert "winrate" not in available
    assert "bars" in available


def test_pareto_scatter_builds(data):
    fig = charts.pareto_scatter(data, "cost_usd", "score")
    assert isinstance(fig, figure)
    assert fig.output_backend == "svg"


def test_pareto_scatter_default_axes(data):
    fig = charts.pareto_scatter(data)
    assert isinstance(fig, figure)


def test_pareto_scatter_can_disable_frontier(data):
    fig = charts.pareto_scatter(data, "cost_usd", "score", show_pareto=False)
    assert isinstance(fig, figure)


def test_bar_leaderboard_builds(data):
    fig = charts.bar_leaderboard(data, "score")
    assert isinstance(fig, figure)


def test_bar_leaderboard_top_n(data):
    fig = charts.bar_leaderboard(data, "score", top_n=2)
    assert isinstance(fig, figure)


def test_task_heatmap_builds(data):
    fig = charts.task_heatmap(data)
    assert isinstance(fig, figure)


def test_task_heatmap_requires_task_scores():
    df = pd.DataFrame({"model": ["A", "B"], "score": [0.9, 0.8]})
    data = LeaderboardData.from_dataframe(df)
    with pytest.raises(ValueError):
        charts.task_heatmap(data)


def test_win_rate_matrix_builds(data):
    fig = charts.win_rate_matrix(data)
    assert isinstance(fig, figure)


def test_win_rate_matrix_explicit_matrix():
    df = pd.DataFrame({"model": ["A", "B"], "score": [0.9, 0.8]})
    data = LeaderboardData.from_dataframe(df)
    matrix = pd.DataFrame(
        [[float("nan"), 0.7], [0.3, float("nan")]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    fig = charts.win_rate_matrix(data, matrix=matrix)
    assert isinstance(fig, figure)


def test_elo_plot_builds(data):
    intervals = {"A": (1480, 1520), "B": (1430, 1470), "C": (1380, 1420)}
    fig = charts.elo_plot(data, intervals=intervals)
    assert isinstance(fig, figure)


def test_elo_plot_requires_ratings():
    df = pd.DataFrame({"model": ["A", "B"], "score": [0.9, 0.8]})
    data = LeaderboardData.from_dataframe(df)
    with pytest.raises(ValueError):
        charts.elo_plot(data)


def test_elo_plot_explicit_mapping():
    df = pd.DataFrame({"model": ["A", "B"], "score": [0.9, 0.8]})
    data = LeaderboardData.from_dataframe(df)
    fig = charts.elo_plot(data, elo={"A": 1500, "B": 1400})
    assert isinstance(fig, figure)


def test_chart_registry_matches_labels():
    assert set(charts.CHART_BUILDERS) == set(charts.CHART_LABELS)
