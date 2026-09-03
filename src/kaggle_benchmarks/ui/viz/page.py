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

"""The full revamped benchmark page (FR1.2 "main page chart embedding").

Where ``dashboard.py`` is a single interactive chart with a view switcher,
this module composes the *whole* benchmark page: a branded header, the hero
trade-off chart front-and-center, the hybrid bar leaderboard, and -- embedded
naturally as you scroll -- the per-task heatmap, pairwise win-rate matrix, and
Elo plot. This is the "prominent and discoverable placement" principle: every
visitor lands on the multi-dimensional visuals, not a static table.

``benchmark_page`` returns a Panel component for notebooks/servers;
``render_page_html`` produces a self-contained static HTML document (no server
needed) suitable for screenshots, social previews, and one-file sharing.
"""

from __future__ import annotations

import html as html_lib
from typing import Any

from kaggle_benchmarks.ui.viz import charts as charts_mod
from kaggle_benchmarks.ui.viz import theme as theme_mod
from kaggle_benchmarks.ui.viz.data import LeaderboardData

# The order sections appear on the page. Hero (scatter) first, then the
# scannable leaderboard, then the deeper-dive matrices.
_SECTION_ORDER = ("scatter", "bars", "heatmap", "winrate", "elo")

_SECTION_BLURBS = {
    "scatter": "Cost vs. quality trade-off. Models on the dashed Pareto "
    "frontier deliver the best score at their price point.",
    "bars": "The leaderboard at a glance -- sorted best-first.",
    "heatmap": "Where each model wins and struggles, task by task.",
    "winrate": "Head-to-head: how often the row model beats the column model.",
    "elo": "Elo ratings with bootstrap confidence intervals.",
}


def _sections(data: LeaderboardData) -> list[str]:
    available = set(charts_mod.available_charts(data))
    return [s for s in _SECTION_ORDER if s in available]


def _headline_stats(data: LeaderboardData) -> list[tuple[str, str]]:
    """A few big numbers for the page header."""
    stats: list[tuple[str, str]] = [("Models", str(len(data.models)))]
    if data.tasks:
        stats.append(("Tasks", str(len(data.tasks))))
    stats.append(("Metrics", str(len(data.metric_names))))

    # Highlight the current leader on the primary quality metric.
    _, quality = data.default_axes()
    column = data.metrics.get(quality, {})
    if column:
        leader = max(column, key=column.get)
        stats.append(
            ("Leader", f"{leader} ({charts_mod._format_value(quality, column[leader])})")
        )
    return stats


# --------------------------------------------------------------------------- #
# Panel component (interactive)
# --------------------------------------------------------------------------- #
def benchmark_page(
    data: LeaderboardData,
    *,
    theme: str | theme_mod.Theme | None = None,
    hero_x: str | None = None,
    hero_y: str | None = None,
) -> Any:
    """Compose the full benchmark page as a Panel column.

    Renders the header, an interactive hero dashboard (view chips + axis
    dropdowns), and every other applicable chart embedded below it.
    """
    import panel as pn

    from kaggle_benchmarks.ui.viz.dashboard import BenchmarkDashboard

    t = theme_mod.resolve_theme(theme)
    sections = _sections(data)

    header = pn.pane.HTML(
        _header_html(data, t), sizing_mode="stretch_width"
    )

    # Hero: the interactive dashboard, defaulting to the trade-off scatter.
    hero = BenchmarkDashboard(
        data,
        theme=t,
        view="scatter" if "scatter" in sections else None,
        x=hero_x,
        y=hero_y,
    )

    blocks: list[Any] = [header, hero.__panel__()]

    # Embed the remaining charts (everything the hero switcher isn't the
    # natural home for) as you scroll.
    for section in sections:
        if section == "scatter":
            continue  # already the hero
        blocks.append(
            pn.pane.HTML(
                _section_heading_html(section, t), sizing_mode="stretch_width"
            )
        )
        fig = charts_mod.CHART_BUILDERS[section](data, theme=t)
        blocks.append(pn.pane.Bokeh(fig, sizing_mode="stretch_width"))

    return pn.Column(
        *blocks,
        sizing_mode="stretch_width",
        styles={"background": t.background, "padding": "0 24px 32px"},
        stylesheets=[_page_css(t)],
    )


# --------------------------------------------------------------------------- #
# Static HTML page (no server required)
# --------------------------------------------------------------------------- #
def render_page_html(
    data: LeaderboardData,
    *,
    theme: str | theme_mod.Theme | None = None,
    title: str | None = None,
    hero_x: str | None = None,
    hero_y: str | None = None,
) -> str:
    """Render the whole benchmark page to a self-contained HTML string.

    Every applicable chart is laid out top-to-bottom inside a branded,
    dark-mode page. The output needs no server -- open it in a browser, paste
    it into a doc, or screenshot it for social. This is what makes a benchmark
    page shareable as a single artifact.
    """
    from bokeh.embed import components
    from bokeh.resources import CDN

    t = theme_mod.resolve_theme(theme)
    title = title or data.benchmark_name
    sections = _sections(data)

    # Build each figure and collect its script/div.
    figures = []
    for section in sections:
        if section == "scatter":
            fig = charts_mod.pareto_scatter(data, hero_x, hero_y, theme=t)
        else:
            fig = charts_mod.CHART_BUILDERS[section](data, theme=t)
        fig.sizing_mode = "stretch_width"
        fig.height = 520 if section in ("scatter", "heatmap", "winrate") else 420
        figures.append((section, fig))

    scripts, divs = components({sec: fig for sec, fig in figures})

    section_html = []
    for section, _ in figures:
        section_html.append(
            f"""
        <section class="chart-card">
          {_section_heading_inner(section)}
          <div class="chart-holder">{divs[section]}</div>
        </section>"""
        )

    body = f"""
    {_header_html(data, t)}
    <main class="page-main">
      {"".join(section_html)}
    </main>
    <footer class="page-footer">
      Generated with the Kaggle native benchmark visualization library ·
      data &amp; charts exportable as CSV / SVG
    </footer>
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_lib.escape(title)} · Kaggle Benchmarks</title>
  {CDN.render()}
  <style>{_full_page_css(t)}</style>
</head>
<body>
  {body}
  {scripts}
</body>
</html>"""


# --------------------------------------------------------------------------- #
# HTML / CSS fragments
# --------------------------------------------------------------------------- #
def _header_html(data: LeaderboardData, t: theme_mod.Theme) -> str:
    stats = _headline_stats(data)
    stat_html = "".join(
        f"""<div class="stat"><span class="stat-value">{html_lib.escape(str(v))}</span>
        <span class="stat-label">{html_lib.escape(k)}</span></div>"""
        for k, v in stats
    )
    return f"""
    <header class="page-header">
      <div class="brand-row">
        <span class="brand-dot"></span>
        <span class="brand-name">Kaggle Benchmarks</span>
      </div>
      <h1 class="page-title">{html_lib.escape(data.benchmark_name)}</h1>
      <p class="page-subtitle">Interactive, multi-dimensional results —
        explore trade-offs, per-task performance, and head-to-head matchups.</p>
      <div class="stat-row">{stat_html}</div>
    </header>"""


def _section_heading_inner(section: str) -> str:
    label = charts_mod.CHART_LABELS.get(section, section)
    blurb = _SECTION_BLURBS.get(section, "")
    return f"""<div class="chart-heading">
        <h2>{html_lib.escape(label)}</h2>
        <p>{html_lib.escape(blurb)}</p>
      </div>"""


def _section_heading_html(section: str, t: theme_mod.Theme) -> str:
    return f'<div class="viz-section-heading">{_section_heading_inner(section)}</div>'


def _page_css(t: theme_mod.Theme) -> str:
    """CSS for the Panel-embedded page chrome."""
    return f"""
:host {{ color: {t.text}; font-family: {theme_mod.FONT}; }}
.viz-section-heading h2 {{ color: {t.text}; margin: 24px 0 2px; font-size: 20px; }}
.viz-section-heading p {{ color: {t.muted_text}; margin: 0 0 8px; font-size: 13px; }}
"""


def _full_page_css(t: theme_mod.Theme) -> str:
    """CSS for the standalone static HTML page."""
    return f"""
:root {{ color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: {t.background};
  color: {t.text};
  font-family: {theme_mod.FONT};
  -webkit-font-smoothing: antialiased;
}}
.page-header {{
  max-width: 1100px;
  margin: 0 auto;
  padding: 40px 24px 8px;
}}
.brand-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 18px; }}
.brand-dot {{
  width: 12px; height: 12px; border-radius: 50%;
  background: {t.brand};
  box-shadow: 0 0 14px {t.brand};
}}
.brand-name {{ color: {t.muted_text}; font-weight: 600; letter-spacing: .04em;
  text-transform: uppercase; font-size: 12px; }}
.page-title {{
  margin: 0;
  font-size: 40px;
  font-weight: 800;
  letter-spacing: -0.02em;
  background: linear-gradient(90deg, {t.text}, {t.brand});
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}}
.page-subtitle {{ color: {t.muted_text}; font-size: 15px; max-width: 640px; margin: 10px 0 24px; }}
.stat-row {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 8px; }}
.stat {{
  background: {t.surface};
  border: 1px solid {t.grid};
  border-radius: 12px;
  padding: 12px 18px;
  display: flex; flex-direction: column; gap: 2px;
  min-width: 96px;
}}
.stat-value {{ font-size: 20px; font-weight: 700; color: {t.text}; }}
.stat-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
  color: {t.muted_text}; }}
.page-main {{ max-width: 1100px; margin: 0 auto; padding: 8px 24px; }}
.chart-card {{
  background: {t.surface};
  border: 1px solid {t.grid};
  border-radius: 16px;
  padding: 20px 20px 8px;
  margin: 22px 0;
  box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}}
.chart-heading h2 {{ margin: 0 0 2px; font-size: 20px; font-weight: 700; color: {t.text}; }}
.chart-heading p {{ margin: 0 0 12px; font-size: 13px; color: {t.muted_text}; }}
.chart-holder {{ width: 100%; }}
.page-footer {{
  max-width: 1100px; margin: 8px auto 0; padding: 24px;
  color: {t.muted_text}; font-size: 12px; text-align: center;
  border-top: 1px solid {t.grid};
}}
"""
