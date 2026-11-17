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
# # Benchmark visualizations
#
# The native visualization library turns benchmark results into premium,
# screenshottable charts: a Pareto trade-off scatter, a hybrid bar
# leaderboard, a per-task success heatmap, a pairwise win-rate matrix, and an
# Elo-with-confidence-intervals plot. Every chart reads from one normalized
# ``LeaderboardData`` object, so any scalar metric can be mapped to any axis
# with zero per-chart engineering (dimensional flexibility).

# %%
import pandas as pd

from kaggle_benchmarks.ui import viz

# %% [markdown]
# ## 1. Build the data layer
#
# In a real workflow you would call ``runs.leaderboard_data()`` on the result
# of ``task.evaluate(...)``. Here we use a synthetic aggregated leaderboard so
# the example runs without any model credentials.

# %%
leaderboard = pd.DataFrame(
    [
        {
            "model": "gemini-2.5-pro",
            "score": 0.91,
            "cost_usd": 0.042,
            "latency_ms": 1800,
            "elo": 1503,
        },
        {
            "model": "claude-opus-4.8",
            "score": 0.93,
            "cost_usd": 0.075,
            "latency_ms": 2100,
            "elo": 1521,
        },
        {
            "model": "gpt-5.5",
            "score": 0.89,
            "cost_usd": 0.060,
            "latency_ms": 1600,
            "elo": 1498,
        },
        {
            "model": "gemini-2.5-flash",
            "score": 0.82,
            "cost_usd": 0.008,
            "latency_ms": 700,
            "elo": 1450,
        },
        {
            "model": "small-oss-7b",
            "score": 0.61,
            "cost_usd": 0.001,
            "latency_ms": 300,
            "elo": 1372,
        },
    ]
)

# Per-task success (1 = solved, 0 = failed) powers the heatmap and the
# pairwise win-rate matrix.
task_scores = {
    "gemini-2.5-pro": {"coding": 1, "math": 1, "reasoning": 1, "retrieval": 0},
    "claude-opus-4.8": {"coding": 1, "math": 1, "reasoning": 1, "retrieval": 1},
    "gpt-5.5": {"coding": 1, "math": 0, "reasoning": 1, "retrieval": 1},
    "gemini-2.5-flash": {"coding": 1, "math": 1, "reasoning": 0, "retrieval": 0},
    "small-oss-7b": {"coding": 0, "math": 1, "reasoning": 0, "retrieval": 0},
}

data = viz.LeaderboardData.from_dataframe(
    leaderboard,
    task_scores=task_scores,
    benchmark_name="Capabilities Benchmark",
)

print("Metrics available for axis selection:", data.metric_names)
print("Default (x, y) axes:", data.default_axes())
print("Renderable chart types:", viz.available_charts(data))

# %% [markdown]
# ## 2. The one-line interactive dashboard
#
# ``viz.dashboard`` returns an embeddable Panel component with fast-access view
# chips, X/Y axis dropdowns, and CSV / chart export buttons. Display it in a
# notebook by evaluating it as the last expression in a cell.

# %%
dashboard = viz.dashboard(data)  # dark mode by default
# dashboard  # <- uncomment in a notebook to render interactively

# %% [markdown]
# ## 3. Individual charts
#
# Each chart is also directly callable and returns a styled Bokeh figure you
# can embed, export, or compose.

# %%
# FR2.2 -- Pareto trade-off scatter. The Pareto frontier (cheapest model at
# each quality level) is auto-computed and highlighted.
scatter = viz.pareto_scatter(data, x="cost_usd", y="score")

# FR2.1 -- hybrid bar leaderboard for a single metric.
bars = viz.bar_leaderboard(data, metric="score")

# FR2.3 -- per-task success heatmap (model x task).
heatmap = viz.task_heatmap(data)

# FR2.4 -- pairwise win-rate matrix (row model beats column model).
winrate = viz.win_rate_matrix(data)

# FR2.5 -- Elo ratings with bootstrap confidence intervals.
elo = viz.elo_plot(
    data,
    intervals={
        "gemini-2.5-pro": (1490, 1516),
        "claude-opus-4.8": (1508, 1534),
        "gpt-5.5": (1485, 1511),
        "gemini-2.5-flash": (1436, 1464),
        "small-oss-7b": (1355, 1389),
    },
)

# %% [markdown]
# ## 4. Shareability: deep links and exports
#
# The dashboard configuration serializes to a URL query string so a shared
# link reproduces the exact chart. Data and charts export to CSV / HTML (and
# SVG when a browser driver is available).

# %%
query = dashboard.to_query_string()
print("Deep link query string:", query)

# Rebuild the identical dashboard from the query string.
restored = viz.BenchmarkDashboard.from_query_string(data, query)
assert restored.config() == dashboard.config()

# FR3.1 -- download the full, rich dataset (not just the headline score).
csv_text = viz.to_csv(data)
print("\nCSV export (first line):", csv_text.splitlines()[0])

# FR3.2 -- export a chart to a self-contained, interactive HTML document.
html = viz.to_html(scatter, title="Capabilities Benchmark")
print("HTML export size (chars):", len(html))
