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

"""Native benchmark visualization library (Kaggle Benchmark visualizations PRD).

A centralized, premium chart library controlled by Kaggle: consistent theming,
dark-mode by default, dimensional flexibility (any scalar metric -> any axis),
automatic Pareto frontiers, and one-click CSV/PNG/SVG export.

Typical use::

    from kaggle_benchmarks.ui import viz

    data = viz.LeaderboardData.from_runs(runs, benchmark_name="My Benchmark")
    viz.dashboard(data)                      # interactive, embeddable
    viz.pareto_scatter(data, "cost_usd", "score")  # a single chart
"""

from kaggle_benchmarks.ui.viz import export
from kaggle_benchmarks.ui.viz.charts import (
    available_charts,
    bar_leaderboard,
    elo_plot,
    pareto_scatter,
    task_heatmap,
    win_rate_matrix,
)
from kaggle_benchmarks.ui.viz.dashboard import BenchmarkDashboard, dashboard
from kaggle_benchmarks.ui.viz.data import LeaderboardData, metric_direction
from kaggle_benchmarks.ui.viz.export import to_csv, to_html, to_png, to_svg
from kaggle_benchmarks.ui.viz.pareto import pareto_indices
from kaggle_benchmarks.ui.viz.theme import DARK, LIGHT, Theme, resolve_theme

__all__ = [
    "DARK",
    "LIGHT",
    "BenchmarkDashboard",
    "LeaderboardData",
    "Theme",
    "available_charts",
    "bar_leaderboard",
    "dashboard",
    "elo_plot",
    "export",
    "metric_direction",
    "pareto_indices",
    "pareto_scatter",
    "resolve_theme",
    "task_heatmap",
    "to_csv",
    "to_html",
    "to_png",
    "to_svg",
    "win_rate_matrix",
]
