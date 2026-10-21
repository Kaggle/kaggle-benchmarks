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

"""Bokeh chart builders for the benchmark visualization library.

Each ``build_*`` function takes a :class:`~kaggle_benchmarks.ui.visualizations.
data.LeaderboardData` (plus a resolved :class:`ChartConfig`) and returns a
styled Bokeh ``figure``. They are intentionally free of any Panel widget state
so they can be reused for static image export (FR3.2) and OpenGraph server-side
rendering (FR4.1) without a running dashboard.

Implemented visualization types (numbers reference Ryan's Visualization Zoo):

* ``build_bar_table``        -- Hybrid table with bars (#24, FR2.1)
* ``build_scatter``          -- Pareto scatter / XY plot (#2, FR2.2)
* ``build_heatmap``          -- Per-task success heatmap (#10, FR2.3)
* ``build_winrate_matrix``   -- Pairwise win-rate matrix (#3, FR2.4)
* ``build_elo``              -- Bootstrap-CI Elo plot (#21, FR2.5)
* ``build_pass_at_k``        -- pass@k / worst-at-k curves (#8/#14, FR5.1)
"""

from __future__ import annotations

import math

from bokeh.models import (
    ColumnDataSource,
    FactorRange,
    HoverTool,
    Label,
    LabelSet,
    Whisker,
)
from bokeh.plotting import figure
from bokeh.transform import linear_cmap

from kaggle_benchmarks.ui.visualizations import theme
from kaggle_benchmarks.ui.visualizations.config import ChartConfig
from kaggle_benchmarks.ui.visualizations.data import LeaderboardData
from kaggle_benchmarks.ui.visualizations.pareto import (
    PointND,
    frontier_line,
    pareto_frontier,
)

_TOOLS = "pan,box_zoom,wheel_zoom,reset,save"


def build_chart(data: LeaderboardData, config: ChartConfig):
    """Dispatch to the builder for ``config.view``.

    Falls back to the bar leaderboard for unknown views so the dashboard always
    renders something.
    """
    config = config.normalized()
    builders = {
        "bars": build_bar_table,
        "scatter": build_scatter,
        "heatmap": build_heatmap,
        "winrate": build_winrate_matrix,
        "elo": build_elo,
        "passk": build_pass_at_k,
    }
    return builders.get(config.view, build_bar_table)(data, config)


def _new_figure(palette: theme.Palette, **kwargs):
    fig = figure(
        tools=_TOOLS,
        toolbar_location="above",
        sizing_mode="stretch_width",
        height=kwargs.pop("height", 460),
        **kwargs,
    )
    return fig


# ---------------------------------------------------------------------------
# FR2.1 Hybrid table with bars (#24)
# ---------------------------------------------------------------------------


def build_bar_table(data: LeaderboardData, config: ChartConfig):
    """Horizontal bar leaderboard -- a scannable upgrade over the flat table."""
    palette = theme.get_palette()
    metric_key = config.y or data.primary_metric_key()
    metric = data.metric(metric_key)
    models = data.ranked_models(metric_key)
    # Bokeh draws the first factor at the bottom; reverse so #1 is on top.
    order = list(reversed(models))

    values = [data.value(m, metric_key) or 0.0 for m in order]
    labels = [metric.format(data.value(m, metric_key)) for m in order]
    colors = [palette.accent] * len(order)
    # Highlight the top model in the frontier color to draw the eye.
    if order:
        colors[-1] = palette.frontier

    source = ColumnDataSource(
        dict(model=order, value=values, label=labels, color=colors)
    )

    fig = _new_figure(
        palette,
        y_range=FactorRange(*order),
        x_axis_label=metric.label,
        height=max(240, 46 * len(order) + 90),
    )
    fig.hbar(
        y="model",
        right="value",
        height=0.72,
        source=source,
        color="color",
        line_color=None,
    )
    # Value labels at the end of each bar (the "table" half of the hybrid).
    fig.add_layout(
        LabelSet(
            x="value",
            y="model",
            text="label",
            source=source,
            x_offset=6,
            y_offset=-7,
            text_font_size="11px",
            text_color=palette.muted_text,
        )
    )
    fig.add_tools(HoverTool(tooltips=[("Model", "@model"), (metric.label, "@label")]))
    fig.x_range.start = 0
    theme.style_figure(fig, palette, title=f"{data.name} — {metric.label}")
    fig.ygrid.grid_line_color = None
    return fig


# ---------------------------------------------------------------------------
# FR2.2 Pareto scatter / XY plot (#2)
# ---------------------------------------------------------------------------


def build_scatter(data: LeaderboardData, config: ChartConfig):
    """Trade-off scatter with an auto-computed, highlighted Pareto frontier."""
    palette = theme.get_palette()
    x_key, y_key = _resolve_axes(data, config)
    x_metric, y_metric = data.metric(x_key), data.metric(y_key)

    points, models = [], []
    for m in data.models:
        xv, yv = data.value(m, x_key), data.value(m, y_key)
        if xv is None or yv is None:
            continue
        points.append(PointND(m, xv, yv))
        models.append(m)

    frontier = (
        pareto_frontier(
            points,
            x_higher_is_better=x_metric.higher_is_better,
            y_higher_is_better=y_metric.higher_is_better,
        )
        if config.show_pareto
        else []
    )
    frontier_set = set(frontier)

    color_map = palette.color_map(models)
    source = ColumnDataSource(
        dict(
            model=models,
            x=[p.x for p in points],
            y=[p.y for p in points],
            x_label=[x_metric.format(p.x) for p in points],
            y_label=[y_metric.format(p.y) for p in points],
            color=[
                palette.frontier if m in frontier_set else color_map[m] for m in models
            ],
            size=[16 if m in frontier_set else 12 for m in models],
        )
    )

    axis_kwargs = {}
    if config.log_x or x_metric.log:
        axis_kwargs["x_axis_type"] = "log"
    if config.log_y or y_metric.log:
        axis_kwargs["y_axis_type"] = "log"

    fig = _new_figure(
        palette,
        x_axis_label=x_metric.label,
        y_axis_label=y_metric.label,
        **axis_kwargs,
    )

    # Frontier polyline drawn under the markers.
    if frontier:
        line_pts = frontier_line(
            points, frontier, x_higher_is_better=x_metric.higher_is_better
        )
        fig.line(
            [p.x for p in line_pts],
            [p.y for p in line_pts],
            line_color=palette.frontier,
            line_width=2,
            line_dash="dashed",
            legend_label="Pareto frontier",
        )

    fig.scatter(
        "x",
        "y",
        size="size",
        source=source,
        fill_color="color",
        line_color=palette.background,
        line_width=1.5,
        fill_alpha=0.9,
    )
    fig.add_layout(
        LabelSet(
            x="x",
            y="y",
            text="model",
            source=source,
            x_offset=9,
            y_offset=4,
            text_font_size="10px",
            text_color=palette.text,
        )
    )
    fig.add_tools(
        HoverTool(
            tooltips=[
                ("Model", "@model"),
                (x_metric.label, "@x_label"),
                (y_metric.label, "@y_label"),
            ]
        )
    )
    if fig.legend:
        fig.legend.location = "bottom_right"
        fig.legend.background_fill_color = palette.surface
        fig.legend.label_text_color = palette.text
        fig.legend.border_line_color = None
    theme.style_figure(fig, palette, title=f"{y_metric.label} vs {x_metric.label}")
    return fig


# ---------------------------------------------------------------------------
# FR2.3 Per-task success heatmap (#10)
# ---------------------------------------------------------------------------


def build_heatmap(data: LeaderboardData, config: ChartConfig):
    """Model x task matrix with color-encoded success rates."""
    palette = theme.get_palette()
    if not data.has_task_matrix:
        return _empty(palette, "No per-task data available for this benchmark.")

    tasks = data.tasks
    models = data.ranked_models(data.primary_metric_key())

    xs, ys, rates, labels = [], [], [], []
    for m in models:
        for t in tasks:
            rate = data.task_scores.get(m, {}).get(t)
            xs.append(t)
            ys.append(m)
            rates.append(float("nan") if rate is None else rate)
            labels.append("—" if rate is None else f"{rate * 100:.0f}%")

    source = ColumnDataSource(dict(task=xs, model=ys, rate=rates, label=labels))
    mapper = linear_cmap(
        "rate", [palette.heat_low, palette.heat_high], low=0.0, high=1.0
    )

    fig = _new_figure(
        palette,
        x_range=FactorRange(*tasks),
        y_range=FactorRange(*list(reversed(models))),
        height=max(240, 34 * len(models) + 120),
    )
    fig.rect(
        x="task",
        y="model",
        width=1,
        height=1,
        source=source,
        fill_color=mapper,
        line_color=palette.background,
    )
    fig.add_tools(
        HoverTool(
            tooltips=[
                ("Model", "@model"),
                ("Task", "@task"),
                ("Success", "@label"),
            ]
        )
    )
    theme.style_figure(fig, palette, title=f"{data.name} — per-task success")
    fig.xgrid.grid_line_color = None
    fig.ygrid.grid_line_color = None
    fig.xaxis.major_label_orientation = math.pi / 4
    return fig


# ---------------------------------------------------------------------------
# FR2.4 Pairwise win-rate matrix (#3)
# ---------------------------------------------------------------------------


def build_winrate_matrix(data: LeaderboardData, config: ChartConfig):
    """Row-beats-column win-rate matrix for Game Arena benchmarks."""
    palette = theme.get_palette()
    if not data.has_pairwise:
        return _empty(palette, "No pairwise data available for this benchmark.")

    models = [m for m in data.models if m in data.pairwise]

    xs, ys, rates, labels = [], [], [], []
    for row in models:
        for col in models:
            if row == col:
                rate = float("nan")
                label = "—"
            else:
                rate = data.pairwise.get(row, {}).get(col)
                rate = float("nan") if rate is None else rate
                label = "—" if math.isnan(rate) else f"{rate * 100:.0f}%"
            xs.append(col)
            ys.append(row)
            rates.append(rate)
            labels.append(label)

    source = ColumnDataSource(dict(col=xs, row=ys, rate=rates, label=labels))
    mapper = linear_cmap(
        "rate", [palette.heat_low, palette.heat_high], low=0.0, high=1.0
    )

    fig = _new_figure(
        palette,
        x_range=FactorRange(*models),
        y_range=FactorRange(*list(reversed(models))),
        x_axis_label="… vs opponent",
        y_axis_label="Win rate of …",
        height=max(280, 40 * len(models) + 130),
    )
    fig.rect(
        x="col",
        y="row",
        width=1,
        height=1,
        source=source,
        fill_color=mapper,
        line_color=palette.background,
    )
    fig.add_layout(
        LabelSet(
            x="col",
            y="row",
            text="label",
            source=source,
            text_font_size="10px",
            text_color=palette.text,
            text_align="center",
            text_baseline="middle",
        )
    )
    fig.add_tools(
        HoverTool(
            tooltips=[("Row", "@row"), ("Column", "@col"), ("Win rate", "@label")]
        )
    )
    theme.style_figure(fig, palette, title=f"{data.name} — pairwise win rate")
    fig.xgrid.grid_line_color = None
    fig.ygrid.grid_line_color = None
    fig.xaxis.major_label_orientation = math.pi / 4
    return fig


# ---------------------------------------------------------------------------
# FR2.5 Bootstrap-CI Elo plot (#21)
# ---------------------------------------------------------------------------


def build_elo(data: LeaderboardData, config: ChartConfig):
    """Elo ratings with bootstrap confidence intervals (error bars)."""
    palette = theme.get_palette()
    if not data.has_elo:
        return _empty(palette, "No Elo ratings available for this benchmark.")

    ordered = sorted(data.elo.items(), key=lambda kv: kv[1][0], reverse=True)
    models = [m for m, _ in ordered]
    ratings = [r for _, (r, _) in ordered]
    radii = [ci for _, (_, ci) in ordered]
    lowers = [r - ci for r, ci in zip(ratings, radii)]
    uppers = [r + ci for r, ci in zip(ratings, radii)]

    source = ColumnDataSource(
        dict(
            model=models,
            rating=ratings,
            lower=lowers,
            upper=uppers,
            label=[f"{r:.0f} ±{ci:.0f}" for r, ci in zip(ratings, radii)],
        )
    )

    fig = _new_figure(
        palette,
        x_range=FactorRange(*models),
        y_axis_label="Elo rating",
        height=460,
    )
    fig.scatter(
        "model",
        "rating",
        source=source,
        size=13,
        fill_color=palette.accent,
        line_color=palette.background,
    )
    whisker = Whisker(
        base="model", upper="upper", lower="lower", source=source, level="annotation"
    )
    whisker.line_color = palette.muted_text
    whisker.upper_head.line_color = palette.muted_text
    whisker.lower_head.line_color = palette.muted_text
    fig.add_layout(whisker)
    fig.add_tools(HoverTool(tooltips=[("Model", "@model"), ("Elo", "@label")]))
    theme.style_figure(fig, palette, title=f"{data.name} — Elo (95% CI)")
    fig.xaxis.major_label_orientation = math.pi / 4
    return fig


# ---------------------------------------------------------------------------
# FR5.1 pass@k curves (#8/#14)
# ---------------------------------------------------------------------------


def build_pass_at_k(data: LeaderboardData, config: ChartConfig):
    """pass@k curves per model (rising reliability-vs-budget curves)."""
    palette = theme.get_palette()
    if not data.has_pass_at_k:
        return _empty(palette, "No pass@k data available for this benchmark.")

    fig = _new_figure(
        palette,
        x_axis_label="k (samples)",
        y_axis_label="pass@k",
        height=460,
    )
    color_map = palette.color_map(data.pass_at_k.keys())
    for model, curve in data.pass_at_k.items():
        ks = sorted(curve)
        ys = [curve[k] for k in ks]
        src = ColumnDataSource(
            dict(
                k=ks,
                y=ys,
                label=[f"{v * 100:.0f}%" for v in ys],
                model=[model] * len(ks),
            )
        )
        fig.line(
            "k",
            "y",
            source=src,
            line_width=2.5,
            color=color_map[model],
            legend_label=model,
        )
        fig.scatter(
            "k",
            "y",
            source=src,
            size=8,
            color=color_map[model],
            line_color=palette.background,
        )
    fig.add_tools(
        HoverTool(tooltips=[("Model", "@model"), ("k", "@k"), ("pass@k", "@label")])
    )
    if fig.legend:
        fig.legend.location = "bottom_right"
        fig.legend.background_fill_color = palette.surface
        fig.legend.label_text_color = palette.text
        fig.legend.border_line_color = None
    theme.style_figure(fig, palette, title=f"{data.name} — pass@k")
    return fig


def _resolve_axes(data: LeaderboardData, config: ChartConfig) -> tuple[str, str]:
    """Pick valid X/Y metric keys, falling back to sensible defaults."""
    keys = set(data.scalar_metric_keys)
    dx, dy = data.default_axes()
    x = config.x if config.x in keys else dx
    y = config.y if config.y in keys else dy
    return x, y


def _empty(palette: theme.Palette, message: str):
    """A styled placeholder figure for views lacking the required data."""
    fig = _new_figure(palette, height=300)
    fig.add_layout(
        Label(
            x=0.5,
            y=0.5,
            x_units="screen",
            y_units="screen",
            text=message,
            text_color=palette.muted_text,
            text_align="center",
        )
    )
    theme.style_figure(fig, palette)
    fig.xaxis.visible = False
    fig.yaxis.visible = False
    fig.grid.grid_line_color = None
    return fig
