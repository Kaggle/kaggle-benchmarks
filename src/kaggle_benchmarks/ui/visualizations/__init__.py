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

"""Kaggle benchmark visualization library (PRD prototype).

A native, Kaggle-branded chart library for benchmark leaderboards. Decouples
visualization type from data so a single component serves any metric pairing,
auto-highlights Pareto frontiers, and serializes view state into shareable deep
links.

Quick start::

    from kaggle_benchmarks.ui import visualizations as viz

    data = viz.demo_data()            # or viz.from_runs(my_runs)
    viz.dashboard(data)               # renders in a notebook

    # Or build a single static chart for export:
    fig = viz.build_chart(data, viz.ChartConfig(view="scatter"))
    viz.export.to_svg(fig)
"""

from kaggle_benchmarks.ui.visualizations import charts, export, theme
from kaggle_benchmarks.ui.visualizations.charts import build_chart
from kaggle_benchmarks.ui.visualizations.config import (
    DEFAULT_VIEW,
    VIEW_TYPES,
    ChartConfig,
)
from kaggle_benchmarks.ui.visualizations.dashboard import (
    BenchmarkDashboard,
    dashboard,
)
from kaggle_benchmarks.ui.visualizations.data import (
    STANDARD_METRICS,
    LeaderboardData,
    Metric,
    from_runs,
)
from kaggle_benchmarks.ui.visualizations.demo import demo_data
from kaggle_benchmarks.ui.visualizations.pareto import (
    PointND,
    pareto_frontier,
)

__all__ = [
    "BenchmarkDashboard",
    "ChartConfig",
    "DEFAULT_VIEW",
    "LeaderboardData",
    "Metric",
    "PointND",
    "STANDARD_METRICS",
    "VIEW_TYPES",
    "build_chart",
    "charts",
    "dashboard",
    "demo_data",
    "export",
    "from_runs",
    "pareto_frontier",
    "theme",
]
