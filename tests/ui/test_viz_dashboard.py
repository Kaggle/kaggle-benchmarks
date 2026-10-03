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

from kaggle_benchmarks.ui.viz import export
from kaggle_benchmarks.ui.viz.dashboard import BenchmarkDashboard, _slug, dashboard
from kaggle_benchmarks.ui.viz.data import LeaderboardData


@pytest.fixture()
def data():
    df = pd.DataFrame(
        {
            "model": ["A", "B", "C"],
            "score": [0.9, 0.8, 0.7],
            "cost_usd": [0.10, 0.02, 0.05],
        }
    )
    task_scores = {"A": {0: 1.0, 1: 0.0}, "B": {0: 1.0, 1: 1.0}, "C": {0: 0.0, 1: 1.0}}
    return LeaderboardData.from_dataframe(
        df, task_scores=task_scores, benchmark_name="My Benchmark"
    )


def test_dashboard_default_view_is_scatter(data):
    dash = dashboard(data)
    assert dash.view == "scatter"
    assert dash.x == "cost_usd"
    assert dash.y == "score"


def test_dashboard_falls_back_to_bars_without_scatter():
    df = pd.DataFrame({"model": ["A", "B"], "score": [0.9, 0.8]})
    data = LeaderboardData.from_dataframe(df)
    dash = dashboard(data)
    assert dash.view == "bars"


def test_dashboard_rejects_empty_data():
    data = LeaderboardData(models=[], metrics={})
    with pytest.raises(ValueError):
        dashboard(data)


def test_deep_link_roundtrip(data):
    dash = dashboard(data, view="scatter", x="cost_usd", y="score")
    qs = dash.to_query_string()
    assert "view=scatter" in qs
    rebuilt = BenchmarkDashboard.from_query_string(data, qs)
    assert rebuilt.config() == {"view": "scatter", "x": "cost_usd", "y": "score"}


def test_deep_link_from_leading_question_mark(data):
    rebuilt = BenchmarkDashboard.from_query_string(data, "?view=bars")
    assert rebuilt.view == "bars"


def test_deep_link_ignores_invalid_view(data):
    # Unknown view falls back to the default rather than raising.
    dash = BenchmarkDashboard(data, view="nonsense")
    assert dash.view in ("scatter", "bars")


def test_build_chart_returns_figure(data):
    from bokeh.plotting import figure

    dash = dashboard(data)
    assert isinstance(dash.build_chart(), figure)


def test_panel_layout_builds(data):
    dash = dashboard(data)
    layout = dash.__panel__()
    assert isinstance(layout, pn.viewable.Viewable)


def test_config_omits_missing_axes():
    df = pd.DataFrame({"model": ["A", "B"], "score": [0.9, 0.8]})
    data = LeaderboardData.from_dataframe(df)
    dash = dashboard(data)  # bars view, single metric -> no x/y defaults
    cfg = dash.config()
    assert cfg["view"] == "bars"


def test_slug():
    assert _slug("My Benchmark!") == "my-benchmark"
    assert _slug("") == "benchmark"
    assert _slug("A / B / C") == "a-b-c"


# ---- export helpers -------------------------------------------------------- #


def test_to_csv_contains_all_metrics(data):
    csv = export.to_csv(data)
    header = csv.splitlines()[0]
    assert "model" in header
    assert "score" in header
    assert "cost_usd" in header


def test_task_matrix_csv(data):
    csv = export.task_matrix_csv(data)
    assert csv  # non-empty because per-task data exists


def test_task_matrix_csv_empty_without_task_data():
    df = pd.DataFrame({"model": ["A", "B"], "score": [0.9, 0.8]})
    data = LeaderboardData.from_dataframe(df)
    assert export.task_matrix_csv(data) == ""


def test_to_html_is_self_contained(data):
    from kaggle_benchmarks.ui.viz import charts

    fig = charts.pareto_scatter(data)
    html = export.to_html(fig, title="X")
    assert "<html" in html.lower()
    assert "bokeh" in html.lower()


def test_to_svg_raises_cleanly_without_driver(data, monkeypatch):
    from kaggle_benchmarks.ui.viz import charts

    fig = charts.bar_leaderboard(data)

    def _boom(*_a, **_k):
        raise RuntimeError("no selenium")

    monkeypatch.setattr("bokeh.io.export.get_svg", _boom)
    with pytest.raises(export.ExportUnavailableError):
        export.to_svg(fig)
