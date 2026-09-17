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
import panel as pn
import pytest

from kaggle_benchmarks.ui.viz import page as page_mod
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
        df, task_scores=task_scores, benchmark_name="My Benchmark"
    )


def test_benchmark_page_builds_panel(data):
    layout = page_mod.benchmark_page(data)
    assert isinstance(layout, pn.viewable.Viewable)


def test_render_page_html_is_self_contained(data):
    html = page_mod.render_page_html(data)
    assert "<!DOCTYPE html>" in html
    assert "My Benchmark" in html
    assert "bokeh" in html.lower()


def test_render_page_html_includes_all_sections(data):
    html = page_mod.render_page_html(data)
    # Every available chart section's heading label should appear.
    for label in (
        "Pareto scatter",
        "Leaderboard bars",
        "Task heatmap",
        "Win-rate matrix",
        "Elo + CI",
    ):
        assert label in html


def test_render_page_html_escapes_benchmark_name():
    df = pd.DataFrame({"model": ["A", "B"], "score": [0.9, 0.8]})
    data = LeaderboardData.from_dataframe(df, benchmark_name="<script>x</script>")
    html = page_mod.render_page_html(data)
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;" in html


def test_page_without_task_data_still_renders():
    df = pd.DataFrame(
        {"model": ["A", "B"], "score": [0.9, 0.8], "cost_usd": [0.1, 0.2]}
    )
    data = LeaderboardData.from_dataframe(df)
    html = page_mod.render_page_html(data)
    # No per-task charts, but scatter + bars remain.
    assert "Pareto scatter" in html
    assert "Task heatmap" not in html


def test_headline_stats_reports_leader(data):
    stats = dict(page_mod._headline_stats(data))
    assert stats["Models"] == "3"
    assert "A" in stats["Leader"]  # A has the top score
