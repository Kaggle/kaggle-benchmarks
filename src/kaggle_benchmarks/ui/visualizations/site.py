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

"""Static, client-side benchmark page generator.

Unlike the Panel dashboard (which needs a live Python server for its widgets),
this produces a SINGLE self-contained ``.html`` file that runs entirely in the
browser -- open it with a double-click, no server, no proxy. It reproduces the
Kaggle benchmark-page experience the PRD describes:

* fast-access **view chips** above the chart (Leaderboard / Trade-off / ...),
* independent **X / Y axis dropdowns** for the scatter,
* a **Pareto frontier** toggle,
* **Download CSV** and per-chart image save (via Bokeh's toolbar).

It works by pre-rendering every reachable chart state to Bokeh JSON with
``bokeh.embed.json_item`` and embedding them all; vanilla JS then shows the one
matching the current controls and reads the deep-link state from the URL hash.
This is the artifact to hand to a designer / PM who just wants to *see and
click* the visualizations.
"""

from __future__ import annotations

import base64
import json

from bokeh.embed import json_item
from bokeh.resources import CDN

from kaggle_benchmarks.ui.visualizations import charts, theme
from kaggle_benchmarks.ui.visualizations.config import VIEW_TYPES, ChartConfig
from kaggle_benchmarks.ui.visualizations.data import LeaderboardData


def _available_views(data: LeaderboardData) -> list[str]:
    """View tokens the data can actually populate (mirrors the dashboard)."""
    views = ["bars"]
    if len(data.scalar_metric_keys) >= 2:
        views.append("scatter")
    if data.has_task_matrix:
        views.append("heatmap")
    if data.has_pairwise:
        views.append("winrate")
    if data.has_elo:
        views.append("elo")
    if data.has_pass_at_k:
        views.append("passk")
    return views


def _plot_key(view: str, x: str | None, y: str | None, pareto: bool) -> str:
    """Stable id for a pre-rendered chart state."""
    if view == "scatter":
        return f"scatter|{x}|{y}|{int(pareto)}"
    if view == "bars":
        return f"bars|{y}"
    return view


def _render_all(data: LeaderboardData, views: list[str]) -> dict[str, dict]:
    """Pre-render every reachable chart state to Bokeh embed JSON.

    * scatter: every (x, y) metric pair with x != y, both Pareto on and off,
    * bars: one per selectable metric,
    * others: a single figure each.
    """
    keys = data.scalar_metric_keys
    plots: dict[str, dict] = {}

    for view in views:
        if view == "scatter":
            for x in keys:
                for y in keys:
                    if x == y:
                        continue
                    for pareto in (True, False):
                        cfg = ChartConfig(view="scatter", x=x, y=y, show_pareto=pareto)
                        fig = charts.build_chart(data, cfg)
                        key = _plot_key("scatter", x, y, pareto)
                        plots[key] = json_item(fig, key)
        elif view == "bars":
            for y in keys:
                cfg = ChartConfig(view="bars", y=y)
                fig = charts.build_chart(data, cfg)
                key = _plot_key("bars", None, y, False)
                plots[key] = json_item(fig, key)
        else:
            cfg = ChartConfig(view=view)
            fig = charts.build_chart(data, cfg)
            plots[view] = json_item(fig, view)

    return plots


def generate_site(
    data: LeaderboardData,
    *,
    title: str = "Kaggle Benchmarks",
) -> str:
    """Return a complete, self-contained HTML document for ``data``.

    The result has no external dependency except the Bokeh CDN JS (needed to
    paint the charts). Everything else -- data, every chart state, the chip and
    dropdown logic, the CSV download -- is inlined.
    """
    views = _available_views(data)
    palette = theme.get_palette()
    keys = data.scalar_metric_keys
    default_x, default_y = data.default_axes()
    initial_view = "scatter" if "scatter" in views else "bars"

    plots = _render_all(data, views)
    plots_json = _safe_json(plots)
    bokeh_scripts = "\n".join(
        f'<script src="{src}" crossorigin="anonymous"></script>' for src in CDN.js_files
    )

    metric_labels = {k: data.metric(k).label for k in keys}
    view_labels = {v: VIEW_TYPES[v] for v in views}

    csv_b64 = base64.b64encode(data.to_csv().encode("utf-8")).decode("ascii")
    csv_name = _slug(data.name) + ".csv"

    config_js = _safe_json(
        {
            "views": views,
            "viewLabels": view_labels,
            "metricKeys": keys,
            "metricLabels": metric_labels,
            "defaultX": default_x,
            "defaultY": default_y,
            "initialView": initial_view,
            "benchmarkName": data.name,
        }
    )

    return _TEMPLATE.format(
        title=title,
        benchmark_name=_esc(data.name),
        model_count=len(data.models),
        metric_count=len(keys),
        task_count=len(data.tasks),
        bokeh_scripts=bokeh_scripts,
        plots_json=plots_json,
        config_js=config_js,
        csv_b64=csv_b64,
        csv_name=csv_name,
        bg=palette.background,
        surface=palette.surface,
        text=palette.text,
        muted=palette.muted_text,
        grid=palette.grid,
        accent=palette.accent,
        frontier=palette.frontier,
    )


def write_site(
    data: LeaderboardData,
    path: str,
    *,
    title: str = "Kaggle Benchmarks",
) -> str:
    """Write the static site to ``path`` and return the path."""
    html = generate_site(data, title=title)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def _safe_json(obj) -> str:
    """JSON-encode for safe embedding inside a ``<script>`` block.

    ``json.dumps`` leaves ``<``/``>``/``&`` intact, so a benchmark or model
    name containing ``</script>`` could otherwise break out of the tag. Escape
    those to their unicode escapes, which JSON parses back to the same string.
    """
    return (
        json.dumps(obj)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# The page shell. Kept as one string so the output is a single portable file.
# JS braces are doubled to survive ``str.format``.
_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
{bokeh_scripts}
<style>
  :root {{
    --bg: {bg}; --surface: {surface}; --text: {text}; --muted: {muted};
    --grid: {grid}; --accent: {accent}; --frontier: {frontier};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--text);
    font-family: Inter, "Helvetica Neue", Helvetica, Arial, sans-serif;
  }}
  header.kg {{
    background: var(--accent); color: #fff; padding: 14px 24px;
    display: flex; align-items: center; gap: 12px;
  }}
  header.kg .logo {{ font-weight: 800; font-size: 20px; letter-spacing: -0.5px; }}
  header.kg .crumb {{ opacity: 0.9; font-size: 14px; }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 24px; }}
  .title {{ font-size: 28px; font-weight: 700; margin: 4px 0 2px; }}
  .subtitle {{ color: var(--muted); font-size: 14px; margin-bottom: 18px; }}
  .stats {{ display: flex; gap: 28px; margin: 12px 0 22px; }}
  .stat .n {{ font-size: 22px; font-weight: 700; }}
  .stat .l {{ font-size: 12px; color: var(--muted); text-transform: uppercase;
             letter-spacing: 0.5px; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }}
  .chip {{
    border: 1px solid var(--grid); background: var(--surface); color: var(--text);
    border-radius: 999px; padding: 8px 16px; font-size: 14px; cursor: pointer;
    transition: all 0.12s ease;
  }}
  .chip:hover {{ border-color: var(--accent); }}
  .chip.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
  .controls {{
    display: flex; flex-wrap: wrap; gap: 16px; align-items: end;
    margin-bottom: 12px; padding: 14px; background: var(--surface);
    border-radius: 10px; border: 1px solid var(--grid);
  }}
  .control label {{ display: block; font-size: 12px; color: var(--muted);
    margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
  select {{
    background: var(--bg); color: var(--text); border: 1px solid var(--grid);
    border-radius: 8px; padding: 8px 10px; font-size: 14px; min-width: 150px;
  }}
  .toggle {{ display: flex; align-items: center; gap: 8px; font-size: 14px; }}
  .hidden {{ display: none !important; }}
  .chart-card {{
    background: var(--surface); border: 1px solid var(--grid);
    border-radius: 12px; padding: 16px; min-height: 420px;
  }}
  .actions {{ display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap; }}
  .btn {{
    text-decoration: none; display: inline-block; padding: 10px 18px;
    border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer;
    border: 1px solid var(--grid); background: var(--surface); color: var(--text);
  }}
  .btn.primary {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
  .sharebar {{ margin-top: 14px; display: flex; gap: 8px; align-items: center; }}
  .sharebar input {{
    flex: 1; padding: 9px 12px; border-radius: 8px; border: 1px solid var(--grid);
    background: var(--bg); color: var(--text); font-size: 13px;
  }}
</style>
</head>
<body>
<header class="kg">
  <span class="logo">kaggle</span>
  <span class="crumb">Benchmarks &nbsp;/&nbsp; {benchmark_name}</span>
</header>
<div class="wrap">
  <div class="title">{benchmark_name}</div>
  <div class="subtitle">Interactive benchmark visualizations &mdash; toggle views, remap axes, and share.</div>
  <div class="stats">
    <div class="stat"><div class="n">{model_count}</div><div class="l">Models</div></div>
    <div class="stat"><div class="n">{metric_count}</div><div class="l">Metrics</div></div>
    <div class="stat"><div class="n">{task_count}</div><div class="l">Tasks</div></div>
  </div>

  <div class="chips" id="chips"></div>

  <div class="controls" id="controls">
    <div class="control" id="ctl-x">
      <label>X axis</label>
      <select id="sel-x"></select>
    </div>
    <div class="control" id="ctl-y">
      <label>Y axis</label>
      <select id="sel-y"></select>
    </div>
    <div class="control toggle" id="ctl-pareto">
      <input type="checkbox" id="chk-pareto" checked/>
      <label style="margin:0;text-transform:none;letter-spacing:0;">Show Pareto frontier</label>
    </div>
  </div>

  <div class="chart-card"><div id="chart"></div></div>

  <div class="actions">
    <a class="btn primary" id="dl-csv" download="{csv_name}"
       href="data:text/csv;base64,{csv_b64}">&#11015; Download data (CSV)</a>
  </div>

  <div class="sharebar">
    <input id="deeplink" readonly/>
    <button class="btn" id="copylink">Copy link</button>
  </div>
</div>

<script type="application/json" id="plots-data">{plots_json}</script>
<script type="application/json" id="app-config">{config_js}</script>
<script>
(function() {{
  const PLOTS = JSON.parse(document.getElementById("plots-data").textContent);
  const CFG = JSON.parse(document.getElementById("app-config").textContent);

  const state = {{
    view: CFG.initialView,
    x: CFG.defaultX,
    y: CFG.defaultY,
    pareto: true,
  }};

  function plotKey() {{
    if (state.view === "scatter")
      return "scatter|" + state.x + "|" + state.y + "|" + (state.pareto ? 1 : 0);
    if (state.view === "bars") return "bars|" + state.y;
    return state.view;
  }}

  // ---- build chips ----
  const chips = document.getElementById("chips");
  CFG.views.forEach(v => {{
    const b = document.createElement("button");
    b.className = "chip" + (v === state.view ? " active" : "");
    b.textContent = CFG.viewLabels[v];
    b.dataset.view = v;
    b.onclick = () => {{ state.view = v; syncControls(); render(); }};
    chips.appendChild(b);
  }});

  // ---- build axis dropdowns ----
  const selX = document.getElementById("sel-x");
  const selY = document.getElementById("sel-y");
  CFG.metricKeys.forEach(k => {{
    const ox = new Option(CFG.metricLabels[k], k, false, k === state.x);
    const oy = new Option(CFG.metricLabels[k], k, false, k === state.y);
    selX.add(ox); selY.add(oy);
  }});
  selX.onchange = () => {{ state.x = selX.value; render(); }};
  selY.onchange = () => {{ state.y = selY.value; render(); }};
  document.getElementById("chk-pareto").onchange = (e) => {{
    state.pareto = e.target.checked; render();
  }};

  function syncControls() {{
    document.querySelectorAll(".chip").forEach(c =>
      c.classList.toggle("active", c.dataset.view === state.view));
    const isScatter = state.view === "scatter";
    const isBars = state.view === "bars";
    document.getElementById("ctl-x").classList.toggle("hidden", !isScatter);
    document.getElementById("ctl-pareto").classList.toggle("hidden", !isScatter);
    // Bars uses the Y dropdown as its single "metric" selector.
    document.getElementById("ctl-y").classList.toggle("hidden", !(isScatter || isBars));
    document.querySelector("#ctl-y label").textContent = isBars ? "Metric" : "Y axis";
    document.getElementById("controls").classList.toggle(
      "hidden", !(isScatter || isBars));
  }}

  function updateDeepLink() {{
    const p = new URLSearchParams();
    p.set("view", state.view);
    if (state.view === "scatter") {{
      p.set("x", state.x); p.set("y", state.y);
      if (!state.pareto) p.set("pareto", "0");
    }} else if (state.view === "bars") {{
      p.set("y", state.y);
    }}
    const url = location.origin + location.pathname + "#" + p.toString();
    document.getElementById("deeplink").value = url;
    history.replaceState(null, "", "#" + p.toString());
  }}

  function readHash() {{
    if (!location.hash) return;
    const p = new URLSearchParams(location.hash.slice(1));
    if (p.get("view") && CFG.views.includes(p.get("view"))) state.view = p.get("view");
    if (p.get("x") && CFG.metricKeys.includes(p.get("x"))) state.x = p.get("x");
    if (p.get("y") && CFG.metricKeys.includes(p.get("y"))) state.y = p.get("y");
    if (p.get("pareto") === "0") state.pareto = false;
    document.getElementById("chk-pareto").checked = state.pareto;
    selX.value = state.x; selY.value = state.y;
  }}

  const chart = document.getElementById("chart");
  function render() {{
    let key = plotKey();
    let item = PLOTS[key];
    if (!item && state.view === "scatter" && state.x === state.y) {{
      // Guard: identical axes have no pre-rendered plot; nudge to defaults.
      state.y = CFG.metricKeys.find(k => k !== state.x) || state.y;
      selY.value = state.y;
      key = plotKey(); item = PLOTS[key];
    }}
    chart.innerHTML = "";
    if (item) {{
      Bokeh.embed.embed_item(item, "chart");
    }} else {{
      chart.textContent = "No chart available for this selection.";
    }}
    updateDeepLink();
  }}

  document.getElementById("copylink").onclick = () => {{
    const el = document.getElementById("deeplink");
    el.select();
    navigator.clipboard && navigator.clipboard.writeText(el.value);
  }};

  readHash();
  syncControls();
  render();
}})();
</script>
</body>
</html>
"""
