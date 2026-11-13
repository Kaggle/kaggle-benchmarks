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

"""Native premium chart builders for benchmark visualizations (FR2.*).

Every function here returns a styled Bokeh ``figure`` so it can be embedded in
a Panel dashboard, exported to PNG/SVG, or rendered standalone. The builders
share one styling source (``theme``) and one data source (``LeaderboardData``)
so the whole library reads as a single cohesive system.

Chart coverage (P0 MVP from the PRD):
  * FR2.1 hybrid table-with-bars leaderboard  -> ``bar_leaderboard``
  * FR2.2 Pareto scatter / XY trade-off plot  -> ``pareto_scatter``
  * FR2.3 per-task success heatmap            -> ``task_heatmap``
  * FR2.4 pairwise win-rate matrix            -> ``win_rate_matrix``
  * FR2.5 bootstrap-CI Elo plot               -> ``elo_plot``
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from bokeh.models import (
    ColorBar,
    ColumnDataSource,
    FactorRange,
    HoverTool,
    LinearColorMapper,
)
from bokeh.plotting import figure
from bokeh.transform import transform

from kaggle_benchmarks.ui.viz import pareto as pareto_mod
from kaggle_benchmarks.ui.viz import theme as theme_mod
from kaggle_benchmarks.ui.viz.data import LeaderboardData, metric_direction

# Tools every chart exposes. ``save`` gives users the FR3.2 one-click PNG
# export straight from the Bokeh toolbar; the SVG path is handled by the
# export helpers module.
_BASE_TOOLS = "pan,wheel_zoom,box_zoom,reset,save"


def _new_figure(theme: theme_mod.Theme, **kwargs: Any) -> figure:
    """Create a figure pre-wired for the premium theme and SVG export."""
    fig = figure(
        tools=kwargs.pop("tools", _BASE_TOOLS),
        toolbar_location=kwargs.pop("toolbar_location", "above"),
        sizing_mode=kwargs.pop("sizing_mode", "stretch_width"),
        height=kwargs.pop("height", 460),
        **kwargs,
    )
    # SVG backend keeps exports crisp at any resolution (FR3.2 "high-res
    # vector"). Bokeh still renders to canvas in the browser.
    fig.output_backend = "svg"
    return fig


def _format_value(metric: str, value: float | None) -> str:
    """Human-friendly formatting for tooltips and bar labels."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if metric in ("score", "accuracy", "pass_rate", "win_rate"):
        return f"{value * 100:.1f}%"
    if metric in ("cost_usd", "cost"):
        return f"${value:,.4f}"
    if metric in ("latency_ms", "latency"):
        return f"{value:,.0f} ms"
    if metric.endswith("tokens"):
        return f"{value:,.0f}"
    if metric == "elo":
        return f"{value:,.0f}"
    return f"{value:,.3g}"


def _axis_label(metric: str) -> str:
    """Turn a metric key into a readable axis label."""
    pretty = {
        "score": "Score",
        "accuracy": "Accuracy",
        "pass_rate": "Pass rate",
        "win_rate": "Win rate",
        "cost_usd": "Cost (USD)",
        "cost": "Cost",
        "latency_ms": "Latency (ms)",
        "input_tokens": "Input tokens",
        "output_tokens": "Output tokens",
        "total_tokens": "Total tokens",
        "elo": "Elo",
    }
    return pretty.get(metric, metric.replace("_", " ").title())


# --------------------------------------------------------------------------- #
# FR2.2 Pareto scatter / XY plot
# --------------------------------------------------------------------------- #
def pareto_scatter(
    data: LeaderboardData,
    x: str | None = None,
    y: str | None = None,
    *,
    theme: str | theme_mod.Theme | None = None,
    show_pareto: bool = True,
    x_dir: str | None = None,
    y_dir: str | None = None,
) -> figure:
    """Trade-off scatter with an auto-highlighted Pareto frontier (FR2.2).

    Any two scalar metrics can be mapped to X and Y (dimensional flexibility,
    FR1.3). When ``x``/``y`` are omitted, ``LeaderboardData.default_axes`` picks
    a resource-vs-quality pairing so the chart is meaningful on first render.
    """
    t = theme_mod.resolve_theme(theme)
    if x is None or y is None:
        default_x, default_y = data.default_axes()
        x = x or default_x
        y = y or default_y

    x_dir = x_dir or metric_direction(x)
    y_dir = y_dir or metric_direction(y)

    models, xs, ys = [], [], []
    for model in data.models:
        xv = data.metrics.get(x, {}).get(model)
        yv = data.metrics.get(y, {}).get(model)
        if xv is None or yv is None:
            continue
        models.append(model)
        xs.append(xv)
        ys.append(yv)

    colors = t.colors_for(models)
    source = ColumnDataSource(
        dict(
            model=models,
            x=xs,
            y=ys,
            color=[colors[m] for m in models],
            x_label=[_format_value(x, v) for v in xs],
            y_label=[_format_value(y, v) for v in ys],
        )
    )

    fig = _new_figure(
        t, title=f"{data.benchmark_name}: {_axis_label(y)} vs {_axis_label(x)}"
    )
    fig.xaxis.axis_label = _axis_label(x)
    fig.yaxis.axis_label = _axis_label(y)

    if show_pareto and models:
        frontier = pareto_mod.pareto_indices(xs, ys, x_dir=x_dir, y_dir=y_dir)
        if len(frontier) >= 2:
            fig.line(
                [xs[i] for i in frontier],
                [ys[i] for i in frontier],
                line_color=t.pareto,
                line_width=3,
                line_dash="dashed",
                legend_label="Pareto frontier",
            )
        # Emphasize frontier points with an outer ring in the pareto accent.
        if frontier:
            fig.scatter(
                [xs[i] for i in frontier],
                [ys[i] for i in frontier],
                size=22,
                marker="circle",
                fill_alpha=0,
                line_color=t.pareto,
                line_width=3,
            )

    renderer = fig.scatter(
        "x",
        "y",
        source=source,
        size=14,
        fill_color="color",
        line_color=t.background,
        line_width=1.5,
        fill_alpha=0.95,
    )

    # Model labels next to each point so the chart is self-describing when
    # screenshotted without an interactive tooltip.
    from bokeh.models import LabelSet

    fig.add_layout(
        LabelSet(
            x="x",
            y="y",
            text="model",
            source=source,
            x_offset=8,
            y_offset=6,
            text_color=t.muted_text,
            text_font=theme_mod.FONT,
            text_font_size="10px",
        )
    )

    hover = HoverTool(
        renderers=[renderer],
        tooltips=[
            ("Model", "@model"),
            (_axis_label(x), "@x_label"),
            (_axis_label(y), "@y_label"),
        ],
    )
    fig.add_tools(hover)
    theme_mod.style_figure(fig, t)
    return fig


# --------------------------------------------------------------------------- #
# FR2.1 Hybrid table-with-bars leaderboard
# --------------------------------------------------------------------------- #
def bar_leaderboard(
    data: LeaderboardData,
    metric: str | None = None,
    *,
    theme: str | theme_mod.Theme | None = None,
    top_n: int | None = None,
) -> figure:
    """Horizontal bar leaderboard for a single metric (FR2.1).

    Replaces the static text table with scannable bars, sorted by the metric's
    optimization direction (best on top). This is the low-effort win that makes
    the default leaderboard feel less static.
    """
    t = theme_mod.resolve_theme(theme)
    metric = metric or data.default_axes()[1]
    direction = metric_direction(metric)

    pairs = [
        (m, v)
        for m in data.models
        if (v := data.metrics.get(metric, {}).get(m)) is not None
    ]
    # Sort so the best model is drawn at the top of the chart.
    pairs.sort(key=lambda kv: kv[1], reverse=(direction == "max"))
    if top_n is not None:
        pairs = pairs[:top_n]
    # Bokeh's categorical y-range draws the first factor at the bottom, so
    # reverse to put the best model on top.
    pairs = pairs[::-1]

    models = [m for m, _ in pairs]
    vals = [v for _, v in pairs]
    colors = t.colors_for(list(reversed(models)))
    source = ColumnDataSource(
        dict(
            model=models,
            value=vals,
            color=[colors[m] for m in models],
            label=[_format_value(metric, v) for v in vals],
        )
    )

    fig = _new_figure(
        t,
        title=f"{data.benchmark_name}: {_axis_label(metric)}",
        y_range=FactorRange(*models),
        height=max(220, 52 * len(models) + 80),
    )
    fig.hbar(
        y="model",
        right="value",
        height=0.68,
        source=source,
        fill_color="color",
        line_color=None,
    )
    from bokeh.models import LabelSet

    fig.add_layout(
        LabelSet(
            x="value",
            y="model",
            text="label",
            source=source,
            x_offset=6,
            y_offset=-7,
            text_color=t.text,
            text_font=theme_mod.FONT,
            text_font_size="11px",
        )
    )
    fig.xaxis.axis_label = _axis_label(metric)
    fig.x_range.start = 0
    hover = HoverTool(tooltips=[("Model", "@model"), (_axis_label(metric), "@label")])
    fig.add_tools(hover)
    theme_mod.style_figure(fig, t)
    fig.ygrid.grid_line_color = None
    return fig


# --------------------------------------------------------------------------- #
# FR2.3 Per-task success heatmap
# --------------------------------------------------------------------------- #
def task_heatmap(
    data: LeaderboardData,
    *,
    theme: str | theme_mod.Theme | None = None,
) -> figure:
    """Model x task success heatmap (FR2.3).

    Color-encodes each model's success on each task so viewers can see which
    tasks are hard and where a model's capability profile differs. Requires
    per-task data (``LeaderboardData.task_scores``).
    """
    t = theme_mod.resolve_theme(theme)
    matrix = data.task_matrix()
    if matrix.empty:
        raise ValueError(
            "task_heatmap requires per-task scores; LeaderboardData.task_scores "
            "is empty."
        )

    tasks = [str(c) for c in matrix.columns]
    models = [str(i) for i in matrix.index]

    xs, ys, vals = [], [], []
    for model in matrix.index:
        for task in matrix.columns:
            v = matrix.loc[model, task]
            xs.append(str(task))
            ys.append(str(model))
            vals.append(
                None
                if v is None or (isinstance(v, float) and math.isnan(v))
                else float(v)
            )

    source = ColumnDataSource(
        dict(
            task=xs,
            model=ys,
            value=vals,
            label=["—" if v is None else f"{v * 100:.0f}%" for v in vals],
        )
    )
    mapper = LinearColorMapper(
        palette=list(t.sequential), low=0.0, high=1.0, nan_color=t.grid
    )

    fig = _new_figure(
        t,
        title=f"{data.benchmark_name}: Per-task success",
        x_range=FactorRange(*tasks),
        y_range=FactorRange(*models[::-1]),
        height=max(220, 42 * len(models) + 120),
        tools="pan,box_zoom,reset,save",
    )
    fig.rect(
        x="task",
        y="model",
        width=1,
        height=1,
        source=source,
        fill_color=transform("value", mapper),
        line_color=t.background,
        line_width=1,
    )
    fig.xaxis.major_label_orientation = math.pi / 4
    fig.xaxis.axis_label = "Task"
    color_bar = ColorBar(
        color_mapper=mapper,
        title="Success",
        title_text_color=t.muted_text,
        major_label_text_color=t.muted_text,
        background_fill_color=t.surface,
        width=12,
    )
    fig.add_layout(color_bar, "right")
    hover = HoverTool(
        tooltips=[("Model", "@model"), ("Task", "@task"), ("Success", "@label")]
    )
    fig.add_tools(hover)
    theme_mod.style_figure(fig, t, hide_grid=True)
    return fig


# --------------------------------------------------------------------------- #
# FR2.4 Pairwise win-rate matrix
# --------------------------------------------------------------------------- #
def win_rate_matrix(
    data: LeaderboardData,
    *,
    theme: str | theme_mod.Theme | None = None,
    matrix: Any = None,
) -> figure:
    """Pairwise win-rate matrix (FR2.4, relevant for Game Arena).

    ``cell[row][col]`` shows how often the row model beats the column model.
    Derived from per-task scores by default; pass ``matrix`` (a square
    DataFrame of win rates in ``[0, 1]``) to supply arena results directly.
    """
    t = theme_mod.resolve_theme(theme)
    wr = data.win_rate_matrix() if matrix is None else matrix
    if wr is None or wr.empty:
        raise ValueError(
            "win_rate_matrix requires per-task scores or an explicit matrix."
        )

    rows = [str(i) for i in wr.index]
    cols = [str(c) for c in wr.columns]

    xs, ys, vals = [], [], []
    for row in wr.index:
        for col in wr.columns:
            v = wr.loc[row, col]
            xs.append(str(col))
            ys.append(str(row))
            vals.append(
                None
                if v is None or (isinstance(v, float) and math.isnan(v))
                else float(v)
            )

    source = ColumnDataSource(
        dict(
            col=xs,
            row=ys,
            value=vals,
            label=["—" if v is None else f"{v * 100:.0f}%" for v in vals],
        )
    )
    # Diverging around 0.5: below is a loss (red), above is a win (green).
    mapper = LinearColorMapper(
        palette=list(t.sequential), low=0.0, high=1.0, nan_color=t.surface
    )

    fig = _new_figure(
        t,
        title=f"{data.benchmark_name}: Pairwise win rate (row beats column)",
        x_range=FactorRange(*cols),
        y_range=FactorRange(*rows[::-1]),
        height=max(240, 46 * len(rows) + 140),
        tools="pan,box_zoom,reset,save",
    )
    fig.rect(
        x="col",
        y="row",
        width=1,
        height=1,
        source=source,
        fill_color=transform("value", mapper),
        line_color=t.background,
        line_width=1,
    )
    from bokeh.models import LabelSet

    fig.add_layout(
        LabelSet(
            x="col",
            y="row",
            text="label",
            source=source,
            text_align="center",
            text_baseline="middle",
            text_color=t.text,
            text_font=theme_mod.FONT,
            text_font_size="10px",
        )
    )
    fig.xaxis.major_label_orientation = math.pi / 4
    hover = HoverTool(
        tooltips=[("Row", "@row"), ("Column", "@col"), ("Win rate", "@label")]
    )
    fig.add_tools(hover)
    theme_mod.style_figure(fig, t, hide_grid=True)
    return fig


# --------------------------------------------------------------------------- #
# FR2.5 Bootstrap-CI Elo plot
# --------------------------------------------------------------------------- #
def elo_plot(
    data: LeaderboardData,
    *,
    elo: Mapping[str, float] | None = None,
    intervals: Mapping[str, tuple[float, float]] | None = None,
    theme: str | theme_mod.Theme | None = None,
    metric: str = "elo",
) -> figure:
    """Elo rating plot with bootstrap confidence intervals (FR2.5).

    Draws each model's rating as a point with an error bar for its CI, sorted
    best-first -- the LMSYS-style view for Game Arena. Ratings come from the
    ``elo`` metric on ``data`` (or an explicit ``elo`` mapping); ``intervals``
    supplies ``(low, high)`` bounds per model when available.
    """
    t = theme_mod.resolve_theme(theme)
    ratings: dict[str, float] = (
        dict(elo) if elo is not None else dict(data.metrics.get(metric, {}))
    )
    if not ratings:
        raise ValueError(
            "elo_plot requires an 'elo' metric on the data or an explicit elo mapping."
        )

    ordered = sorted(ratings.items(), key=lambda kv: kv[1])  # ascending -> best on top
    models = [m for m, _ in ordered]
    values = [v for _, v in ordered]
    intervals = intervals or {}
    lows = [intervals.get(m, (v, v))[0] for m, v in ordered]
    highs = [intervals.get(m, (v, v))[1] for m, v in ordered]

    colors = t.colors_for(list(reversed(models)))
    source = ColumnDataSource(
        dict(
            model=models,
            elo=values,
            low=lows,
            high=highs,
            color=[colors[m] for m in models],
            ci_label=[
                f"{v:.0f}  [{lo:.0f}, {hi:.0f}]"
                for v, lo, hi in zip(values, lows, highs)
            ],
        )
    )

    fig = _new_figure(
        t,
        title=f"{data.benchmark_name}: Elo rating (95% bootstrap CI)",
        y_range=FactorRange(*models),
        height=max(240, 48 * len(models) + 100),
    )
    # Horizontal CI whiskers.
    fig.segment(
        x0="low",
        y0="model",
        x1="high",
        y1="model",
        source=source,
        line_color=t.axis,
        line_width=2,
    )
    renderer = fig.scatter(
        x="elo",
        y="model",
        source=source,
        size=15,
        fill_color="color",
        line_color=t.background,
        line_width=1.5,
    )
    fig.xaxis.axis_label = "Elo"
    hover = HoverTool(
        renderers=[renderer], tooltips=[("Model", "@model"), ("Elo", "@ci_label")]
    )
    fig.add_tools(hover)
    theme_mod.style_figure(fig, t)
    fig.ygrid.grid_line_color = None
    return fig


# Registry mapping the view-chip id used by the dashboard to a builder and a
# human label. Kept here so the dashboard and any external caller agree on the
# canonical set of chart types.
CHART_BUILDERS: dict[str, Any] = {
    "scatter": pareto_scatter,
    "bars": bar_leaderboard,
    "heatmap": task_heatmap,
    "winrate": win_rate_matrix,
    "elo": elo_plot,
}

CHART_LABELS: dict[str, str] = {
    "scatter": "Pareto scatter",
    "bars": "Leaderboard bars",
    "heatmap": "Task heatmap",
    "winrate": "Win-rate matrix",
    "elo": "Elo + CI",
}


def available_charts(data: LeaderboardData) -> list[str]:
    """Which chart ids can actually be rendered for this data.

    Lets the dashboard show only view chips that will produce a real chart
    (e.g. hide the heatmap when there is no per-task data).
    """
    charts: list[str] = []
    if len(data.metric_names) >= 1:
        charts.append("bars")
    if len(data.metric_names) >= 2:
        charts.append("scatter")
    if data.task_scores:
        charts.append("heatmap")
        if len(data.models) >= 2:
            charts.append("winrate")
    if data.metrics.get("elo"):
        charts.append("elo")
    return charts


def scalar_metrics(data: LeaderboardData) -> Sequence[str]:
    """Metrics eligible for axis dropdowns (FR1.3)."""
    return data.metric_names
