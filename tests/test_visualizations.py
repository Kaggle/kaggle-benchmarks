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

"""Tests for the benchmark visualization library prototype."""

import io

import pytest
from bokeh.plotting import figure as bokeh_figure

from kaggle_benchmarks.ui import visualizations as viz
from kaggle_benchmarks.ui.visualizations import theme
from kaggle_benchmarks.ui.visualizations.config import VIEW_TYPES, ChartConfig
from kaggle_benchmarks.ui.visualizations.data import (
    LeaderboardData,
    Metric,
    from_runs,
)
from kaggle_benchmarks.ui.visualizations.pareto import PointND, pareto_frontier


@pytest.fixture()
def data() -> LeaderboardData:
    return viz.demo_data()


# ---------------------------------------------------------------------------
# Pareto frontier (FR2.2)
# ---------------------------------------------------------------------------


class TestPareto:
    def test_maximize_both_axes(self):
        pts = [
            PointND("a", 1, 1),
            PointND("b", 2, 2),  # dominates a
            PointND("c", 3, 1),
        ]
        front = set(
            pareto_frontier(pts, x_higher_is_better=True, y_higher_is_better=True)
        )
        assert front == {"b", "c"}
        assert "a" not in front

    def test_direction_matters(self):
        # Quality (y up) vs cost (x down): cheaper + better is optimal.
        pts = [
            PointND("cheap_good", 1.0, 0.9),
            PointND("pricey_good", 5.0, 0.92),
            PointND("cheap_bad", 1.0, 0.4),  # dominated by cheap_good
        ]
        front = set(
            pareto_frontier(pts, x_higher_is_better=False, y_higher_is_better=True)
        )
        assert "cheap_good" in front
        assert "pricey_good" in front  # best quality, still non-dominated
        assert "cheap_bad" not in front

    def test_ties_all_kept(self):
        pts = [PointND("a", 1, 1), PointND("b", 1, 1)]
        front = pareto_frontier(pts, x_higher_is_better=True, y_higher_is_better=True)
        assert set(front) == {"a", "b"}

    def test_single_point_on_frontier(self):
        pts = [PointND("only", 3, 4)]
        assert pareto_frontier(
            pts, x_higher_is_better=True, y_higher_is_better=True
        ) == ["only"]

    def test_demo_frontier_nonempty(self, data):
        pts = [
            PointND(m, data.value(m, "cost_usd"), data.value(m, "score"))
            for m in data.models
        ]
        front = pareto_frontier(pts, x_higher_is_better=False, y_higher_is_better=True)
        # The cheapest and the most accurate model must both be on the frontier.
        assert front
        assert "qwen-3-72b" in front  # cheapest
        assert "claude-opus-4.8" in front  # highest score


# ---------------------------------------------------------------------------
# Metric formatting
# ---------------------------------------------------------------------------


class TestMetric:
    def test_percent(self):
        assert Metric("s", "S", fmt="percent").format(0.912) == "91.2%"

    def test_currency_small_and_large(self):
        m = Metric("c", "C", fmt="currency")
        assert m.format(0.28) == "$0.280"
        assert m.format(1234.5) == "$1,234.50"

    def test_duration(self):
        m = Metric("l", "L", fmt="duration_ms")
        assert m.format(500) == "500ms"
        assert m.format(4200) == "4.20s"

    def test_none_is_dash(self):
        assert Metric("x", "X").format(None) == "—"


# ---------------------------------------------------------------------------
# LeaderboardData (FR1.3)
# ---------------------------------------------------------------------------


class TestLeaderboardData:
    def test_scalar_metric_keys(self, data):
        keys = data.scalar_metric_keys
        assert "score" in keys and "cost_usd" in keys

    def test_default_axes_is_cost_vs_quality(self, data):
        x, y = data.default_axes()
        assert data.metric(x).higher_is_better is False  # cost on X
        assert data.metric(y).higher_is_better is True  # quality on Y

    def test_ranked_models_respects_direction(self, data):
        by_score = data.ranked_models("score")
        assert by_score[0] == "claude-opus-4.8"  # highest score first
        by_cost = data.ranked_models("cost_usd")
        assert by_cost[0] == "qwen-3-72b"  # cheapest first (lower is better)

    def test_capability_flags(self, data):
        assert data.has_task_matrix
        assert data.has_pairwise
        assert data.has_elo
        assert data.has_pass_at_k

    def test_missing_value_returns_none(self):
        d = LeaderboardData(
            name="t",
            models=["a"],
            metrics={"score": Metric("score", "S")},
            scores={"a": {}},
        )
        assert d.value("a", "score") is None

    def test_default_axes_raises_without_metrics(self):
        d = LeaderboardData(name="t", models=[], metrics={}, scores={})
        with pytest.raises(ValueError):
            d.default_axes()

    def test_to_csv_includes_tasks(self, data):
        csv = data.to_csv()
        assert "Model-level metrics" in csv
        assert "Task-level success rates" in csv
        assert "coding" in csv


# ---------------------------------------------------------------------------
# from_runs aggregation
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, cost_nd, latency, in_tok, out_tok):
        self.total_cost_nanodollars = cost_nd
        self.total_backend_latency_ms = latency
        self.input_tokens = in_tok
        self.output_tokens = out_tok


class _FakeChat:
    def __init__(self, usage):
        self.usage = usage


class _FakeModel:
    def __init__(self, name):
        self.name = name


class _FakeRun:
    def __init__(self, model, passed, usage=None, param_id=None):
        self.params = {"llm": _FakeModel(model)}
        self.passed = passed
        self.chat = _FakeChat(usage) if usage else None
        self.param_id = param_id
        self.evaluated_subject = self.params["llm"]


class TestFromRuns:
    def test_aggregates_score_and_cost(self):
        runs = [
            _FakeRun("m1", True, _FakeUsage(2_000_000_000, 100, 10, 20), "t1"),
            _FakeRun("m1", False, _FakeUsage(2_000_000_000, 300, 10, 20), "t2"),
            _FakeRun("m2", True, _FakeUsage(500_000_000, 50, 5, 5), "t1"),
        ]
        data = from_runs(runs, name="agg")
        assert data.value("m1", "score") == pytest.approx(0.5)
        assert data.value("m2", "score") == pytest.approx(1.0)
        assert data.value("m1", "cost_usd") == pytest.approx(2.0)
        assert data.has_task_matrix

    def test_score_only_runs_produce_valid_data(self):
        runs = [_FakeRun("m1", True), _FakeRun("m1", False)]
        data = from_runs(runs)
        assert data.value("m1", "score") == pytest.approx(0.5)
        # No usage => cost metric dropped entirely.
        assert "cost_usd" not in data.metrics

    def test_unknown_model_skipped(self):
        bad = _FakeRun("m1", True)
        bad.params = {}
        bad.evaluated_subject = None
        data = from_runs([bad])
        assert data.models == []


# ---------------------------------------------------------------------------
# Deep-link serialization (FR4.1)
# ---------------------------------------------------------------------------


class TestChartConfig:
    def test_roundtrip(self):
        c = ChartConfig(
            view="scatter", x="cost_usd", y="score", show_pareto=False, log_x=True
        )
        c2 = ChartConfig.from_query(c.to_query())
        assert c2 == c

    def test_defaults_omitted_from_query(self):
        q = ChartConfig(view="bars").to_query()
        assert "pareto" not in q and "logx" not in q and "logy" not in q

    def test_invalid_view_normalized(self):
        assert ChartConfig.from_query("view=nope").view == "bars"

    def test_malformed_query_is_safe(self):
        c = ChartConfig.from_query("?&&=x=broken&view=")
        assert c.view in VIEW_TYPES

    def test_deep_link_appends_query(self):
        link = ChartConfig(view="elo").deep_link("https://k.co/b")
        assert link == "https://k.co/b?view=elo"
        link2 = ChartConfig(view="elo").deep_link("https://k.co/b?tab=1")
        assert link2.startswith("https://k.co/b?tab=1&")


# ---------------------------------------------------------------------------
# Chart builders (FR2.1-2.5, FR5.1)
# ---------------------------------------------------------------------------


class TestCharts:
    @pytest.mark.parametrize("view", list(VIEW_TYPES))
    def test_every_view_builds_a_figure(self, data, view):
        fig = viz.build_chart(data, ChartConfig(view=view))
        assert isinstance(fig, bokeh_figure)

    def test_scatter_axes_from_config(self, data):
        fig = viz.build_chart(
            data, ChartConfig(view="scatter", x="latency_ms", y="score")
        )
        assert fig.xaxis[0].axis_label == "Latency"
        assert fig.yaxis[0].axis_label == "Score"

    def test_missing_data_views_render_placeholder(self):
        # score-only data => heatmap/winrate/elo/passk should not crash.
        d = LeaderboardData(
            name="bare",
            models=["a", "b"],
            metrics={"score": Metric("score", "Score", fmt="percent")},
            scores={"a": {"score": 0.5}, "b": {"score": 0.7}},
        )
        for view in ("heatmap", "winrate", "elo", "passk"):
            fig = viz.build_chart(d, ChartConfig(view=view))
            assert isinstance(fig, bokeh_figure)

    def test_unknown_view_falls_back_to_bars(self, data):
        fig = viz.build_chart(data, ChartConfig(view="does-not-exist"))
        assert isinstance(fig, bokeh_figure)


# ---------------------------------------------------------------------------
# Theme (FR1.1)
# ---------------------------------------------------------------------------


class TestTheme:
    def test_light_and_dark_differ(self):
        assert (
            theme.get_palette("light").background
            != theme.get_palette("dark").background
        )

    def test_default_maps_to_light(self):
        assert theme.get_palette("default").name == "light"

    def test_unknown_theme_falls_back(self):
        assert theme.get_palette("neon").name == "light"

    def test_categorical_colors_are_stable(self):
        p = theme.get_palette("light")
        assert p.color_for(0) == p.color_for(len(p.categorical))


# ---------------------------------------------------------------------------
# Export (FR3.2)
# ---------------------------------------------------------------------------


class TestExport:
    def test_html_export_is_self_contained(self, data):
        fig = viz.build_chart(data, ChartConfig(view="scatter"))
        html = viz.export.to_html(fig)
        assert "<html" in html.lower()
        assert len(html) > 1000

    def test_webdriver_available_returns_bool(self):
        assert isinstance(viz.export.webdriver_available(), bool)


# ---------------------------------------------------------------------------
# Dashboard (FR1.2, FR3.1)
# ---------------------------------------------------------------------------


class TestDashboard:
    def test_dashboard_renders(self, data):
        import panel as pn

        pn.extension()
        dash = viz.dashboard(data)
        view = dash.__panel__()
        with io.StringIO() as f:
            view.save(f)
            f.seek(0)
            assert len(f.read()) > 1000

    def test_csv_download_callback(self, data):
        dash = viz.dashboard(data)
        payload = dash._download_csv.callback().read()
        assert "Model-level metrics" in payload

    def test_initial_view_is_scatter_when_multimetric(self, data):
        dash = viz.dashboard(data)
        assert dash.config.view == "scatter"

    def test_deep_link_updates_on_view_change(self, data):
        dash = viz.dashboard(data)
        dash._view_chips.value = "elo"
        assert "view=elo" in dash._deep_link.value

    def test_only_supported_views_offered(self):
        d = LeaderboardData(
            name="bars-only",
            models=["a", "b"],
            metrics={"score": Metric("score", "Score", fmt="percent")},
            scores={"a": {"score": 0.5}, "b": {"score": 0.7}},
        )
        dash = viz.dashboard(d)
        # Single scalar metric => no scatter/heatmap/etc, just the bar table.
        assert list(dash._view_chips.options.values()) == ["bars"]
