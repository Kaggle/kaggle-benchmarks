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

# %% [markdown]
# ---
# title: Benchmark Visualizations
# ---
#
# A premium, Kaggle-branded chart library for benchmark leaderboards. The same
# data drives every chart type, Pareto frontiers are highlighted automatically,
# and the exact view state serializes into a shareable deep link.
# %%
from kaggle_benchmarks.ui import visualizations as viz

# A realistic multi-metric benchmark. In practice, build this from your runs:
#   data = viz.from_runs(my_runs.completed_runs, name="My Benchmark")
data = viz.demo_data()

# %% [markdown]
# ## Interactive dashboard
#
# View chips switch visualization type; the X/Y dropdowns map *any* scalar
# metric to either axis (Cost vs Accuracy, Latency vs Score, ...). The Download
# button exports the full task-level dataset as CSV, and the deep-link field
# reproduces the exact chart for anyone you share it with.
# %%
viz.dashboard(data)

# %% [markdown]
# ## A shareable, no-server web page
#
# `write_site` bundles every view, the chips, the axis dropdowns, and the CSV
# download into a single self-contained HTML file whose controls run entirely
# in the browser. Open it with a double-click — no Python server needed — which
# makes it easy to hand to a PM or designer, or to run from the shell with
# `python -m kaggle_benchmarks.ui.visualizations`.
# %%
viz.write_site(data, "benchmark.html")

# %% [markdown]
# ## A single static chart
#
# Every chart is a plain Bokeh figure, so you can build one directly for export
# or embedding — no dashboard required.
# %%
config = viz.ChartConfig(view="scatter", x="cost_usd", y="score")
fig = viz.build_chart(data, config)
fig

# %% [markdown]
# ## Shareable deep link (FR4.1)
#
# The full view state — visualization type, axes, Pareto toggle — round-trips
# through a URL query string.
# %%
link = config.deep_link("https://www.kaggle.com/benchmarks/kaggle/frontier-bench")
print(link)
assert viz.ChartConfig.from_query(config.to_query()) == config

# %% [markdown]
# ## One-click export (FR3.2)
#
# `to_html` produces a self-contained, interactive file anywhere. `to_png` /
# `to_svg` produce high-resolution raster/vector images when a headless browser
# is available (as on Kaggle).
# %%
html = viz.export.to_html(fig)
print(f"Interactive HTML export: {len(html):,} bytes")
