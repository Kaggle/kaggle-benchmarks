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

"""Normalized data model behind the benchmark chart library.

The central idea (PRD "Dimensional flexibility"): decouple the *visualization
type* from the *data*. Every chart consumes a :class:`LeaderboardData`, which
exposes:

* a table of models x scalar metrics (drives bars, scatter/Pareto),
* an optional model x task success matrix (drives the heatmap and pass@k),
* an optional pairwise win matrix + Elo table (drives Game Arena charts).

Because metrics are described by a :class:`Metric` registry -- each carrying a
direction (higher/lower is better) and a formatter -- a single scatter component
can serve Cost vs Accuracy, Latency vs Recall, or any other pairing the user
picks from a dropdown, with no per-combination engineering.
"""

from __future__ import annotations

import dataclasses
import io
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd


@dataclasses.dataclass(frozen=True)
class Metric:
    """Describes one scalar column that can be mapped to a chart axis.

    Attributes:
        key: Stable identifier used in data frames and deep-link URLs.
        label: Human-readable axis / dropdown label.
        higher_is_better: Direction, used by Pareto and default sort. Quality
            metrics (accuracy) are ``True``; resource costs (price, latency)
            are ``False``.
        unit: Optional unit suffix for formatting (e.g. ``"$"``, ``"ms"``).
        fmt: One of ``"percent"``, ``"number"``, ``"currency"``,
            ``"duration_ms"`` -- controls how values are rendered.
        log: Whether the axis defaults to a log scale (useful for cost).
    """

    key: str
    label: str
    higher_is_better: bool = True
    unit: str = ""
    fmt: str = "number"
    log: bool = False

    def format(self, value: float | None) -> str:
        """Render ``value`` for tooltips / table cells."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "—"
        if self.fmt == "percent":
            return f"{value * 100:.1f}%"
        if self.fmt == "currency":
            # Values are in dollars; show cents-level precision for small costs.
            if abs(value) < 1:
                return f"${value:.3f}"
            return f"${value:,.2f}"
        if self.fmt == "duration_ms":
            if value >= 1000:
                return f"{value / 1000:.2f}s"
            return f"{value:.0f}ms"
        # Plain number.
        if float(value).is_integer():
            return f"{int(value):,}{self.unit}"
        return f"{value:,.3g}{self.unit}"


@dataclasses.dataclass
class LeaderboardData:
    """Everything the chart library needs to render a benchmark.

    Args:
        name: Benchmark display name.
        models: Ordered list of model/candidate names (one row each).
        metrics: Registry of scalar metrics keyed by ``Metric.key``.
        scores: ``{model: {metric_key: value}}`` scalar table.
        task_scores: Optional ``{model: {task: success_rate}}`` for the
            per-task heatmap. Values are 0..1.
        pairwise: Optional ``{model_a: {model_b: win_rate}}`` for Game Arena.
        elo: Optional ``{model: (rating, ci_radius)}`` bootstrap Elo.
        pass_at_k: Optional ``{model: {k: pass_rate}}`` for pass@k curves.
    """

    name: str
    models: list[str]
    metrics: dict[str, Metric]
    scores: dict[str, dict[str, float]]
    task_scores: dict[str, dict[str, float]] = dataclasses.field(default_factory=dict)
    pairwise: dict[str, dict[str, float]] = dataclasses.field(default_factory=dict)
    elo: dict[str, tuple[float, float]] = dataclasses.field(default_factory=dict)
    pass_at_k: dict[str, dict[int, float]] = dataclasses.field(default_factory=dict)

    # --- capability flags used to decide which view chips to surface --------

    @property
    def scalar_metric_keys(self) -> list[str]:
        """Metric keys that have at least one finite value across models."""
        keys = []
        for key in self.metrics:
            if any(self._finite(self.scores.get(m, {}).get(key)) for m in self.models):
                keys.append(key)
        return keys

    @property
    def tasks(self) -> list[str]:
        """Sorted union of task names present in ``task_scores``."""
        seen: dict[str, None] = {}
        for model in self.models:
            for task in self.task_scores.get(model, {}):
                seen.setdefault(task, None)
        return list(seen)

    @property
    def has_task_matrix(self) -> bool:
        return bool(self.tasks)

    @property
    def has_pairwise(self) -> bool:
        return bool(self.pairwise)

    @property
    def has_elo(self) -> bool:
        return bool(self.elo)

    @property
    def has_pass_at_k(self) -> bool:
        return bool(self.pass_at_k)

    # --- convenience accessors ---------------------------------------------

    def value(self, model: str, metric_key: str) -> float | None:
        """Scalar value for ``model``/``metric_key`` or ``None`` if missing."""
        v = self.scores.get(model, {}).get(metric_key)
        return v if self._finite(v) else None

    def metric(self, metric_key: str) -> Metric:
        return self.metrics[metric_key]

    def default_axes(self) -> tuple[str, str]:
        """Sensible (x, y) default: primary quality metric vs a cost metric.

        Mirrors the PRD default of "main metric vs. cost". Falls back
        gracefully when only one kind of metric is available.
        """
        keys = self.scalar_metric_keys
        quality = [k for k in keys if self.metrics[k].higher_is_better]
        cost = [k for k in keys if not self.metrics[k].higher_is_better]
        if cost and quality:
            return cost[0], quality[0]
        if len(keys) >= 2:
            return keys[0], keys[1]
        if keys:
            return keys[0], keys[0]
        raise ValueError("LeaderboardData has no scalar metrics to plot.")

    def primary_metric_key(self) -> str:
        """The main quality metric used to sort bar charts / heatmaps."""
        keys = self.scalar_metric_keys
        quality = [k for k in keys if self.metrics[k].higher_is_better]
        return (quality or keys)[0]

    def ranked_models(self, metric_key: str) -> list[str]:
        """Models sorted best-first by ``metric_key`` (respecting direction)."""
        higher = self.metrics[metric_key].higher_is_better

        def sort_key(model: str):
            v = self.value(model, metric_key)
            # Missing values sink to the bottom regardless of direction.
            if v is None:
                return (1, 0.0)
            return (0, -v if higher else v)

        return sorted(self.models, key=sort_key)

    # --- exports (FR3.1) ----------------------------------------------------

    def scalar_dataframe(self) -> pd.DataFrame:
        """Wide model x metric table, columns labeled by metric key."""
        rows = []
        for model in self.models:
            row: dict[str, Any] = {"model": model}
            for key in self.metrics:
                row[key] = self.value(model, key)
            rows.append(row)
        return pd.DataFrame(rows).set_index("model")

    def task_dataframe(self) -> pd.DataFrame:
        """Long model x task success table (empty when no task matrix)."""
        rows = []
        for model in self.models:
            for task in self.tasks:
                rows.append(
                    {
                        "model": model,
                        "task": task,
                        "success_rate": self.task_scores.get(model, {}).get(task),
                    }
                )
        return pd.DataFrame(rows)

    def to_csv(self, *, include_tasks: bool = True) -> str:
        """Full dataset as CSV text for the Download button (FR3.1).

        Emits the scalar table always, and appends the task-level breakdown
        when present -- making concrete the PRD point that Kaggle's data is
        "much richer than just performance".
        """
        buf = io.StringIO()
        buf.write(f"# Kaggle benchmark: {self.name}\n")
        buf.write("# Model-level metrics\n")
        self.scalar_dataframe().to_csv(buf)
        if include_tasks and self.has_task_matrix:
            buf.write("\n# Task-level success rates\n")
            self.task_dataframe().to_csv(buf, index=False)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


# Metrics the SDK captures for (almost) every run, so they are always available
# as axis options even when the benchmark author defines only a score.
STANDARD_METRICS: dict[str, Metric] = {
    "score": Metric("score", "Score", higher_is_better=True, fmt="percent"),
    "cost_usd": Metric(
        "cost_usd", "Cost (USD)", higher_is_better=False, fmt="currency", log=True
    ),
    "latency_ms": Metric(
        "latency_ms", "Latency", higher_is_better=False, fmt="duration_ms"
    ),
    "input_tokens": Metric(
        "input_tokens", "Input tokens", higher_is_better=False, fmt="number"
    ),
    "output_tokens": Metric(
        "output_tokens", "Output tokens", higher_is_better=False, fmt="number"
    ),
}


def _mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None and not _isnan(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _isnan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


# Attach a tidy staticmethod-like helper to the class for the finiteness check
# so it can be used before an instance exists.
def _finite(v: Any) -> bool:
    return v is not None and not _isnan(v)


LeaderboardData._finite = staticmethod(_finite)  # type: ignore[attr-defined]


def from_runs(
    runs: "Sequence[Any]",
    *,
    name: str = "Benchmark",
    by: str = "llm",
    extra_metrics: Mapping[str, Metric] | None = None,
) -> LeaderboardData:
    """Aggregate a flat sequence of :class:`~kaggle_benchmarks.runs.Run` into a
    :class:`LeaderboardData`, grouping by the ``by`` parameter (default the
    ``llm`` candidate).

    For each model we compute:

    * ``score`` -- mean pass rate over that model's runs,
    * ``cost_usd`` / ``latency_ms`` / ``*_tokens`` -- summed/averaged from run
      chat usage when available,
    * a per-task success matrix keyed by each run's ``param_id`` (task id).

    This is intentionally forgiving: runs missing usage or task ids simply
    don't contribute to those dimensions, so a bare pass/fail benchmark still
    produces a valid (score-only) leaderboard.
    """
    grouped: dict[str, list[Any]] = {}
    for run in runs:
        model = _model_name(run, by)
        if model is None:
            continue
        grouped.setdefault(model, []).append(run)

    models = list(grouped)
    scores: dict[str, dict[str, float]] = {}
    task_scores: dict[str, dict[str, float]] = {}

    for model, model_runs in grouped.items():
        passed = [1.0 if _run_passed(r) else 0.0 for r in model_runs]
        row: dict[str, float] = {}
        mean_score = _mean(passed)
        if mean_score is not None:
            row["score"] = mean_score

        costs = [c for c in (_run_cost_usd(r) for r in model_runs) if c is not None]
        if costs:
            row["cost_usd"] = sum(costs) / len(costs)

        latencies = [
            v for v in (_run_latency_ms(r) for r in model_runs) if v is not None
        ]
        if latencies:
            row["latency_ms"] = sum(latencies) / len(latencies)

        in_toks = [v for v in (_run_tokens(r, "input") for r in model_runs) if v]
        out_toks = [v for v in (_run_tokens(r, "output") for r in model_runs) if v]
        if in_toks:
            row["input_tokens"] = sum(in_toks) / len(in_toks)
        if out_toks:
            row["output_tokens"] = sum(out_toks) / len(out_toks)

        scores[model] = row

        # Per-task matrix keyed on param_id (the eval-data index) when present.
        per_task: dict[str, list[float]] = {}
        for r in model_runs:
            task_id = getattr(r, "param_id", None)
            if task_id is None:
                continue
            per_task.setdefault(str(task_id), []).append(1.0 if _run_passed(r) else 0.0)
        if per_task:
            task_scores[model] = {t: _mean(v) or 0.0 for t, v in per_task.items()}

    metrics = dict(STANDARD_METRICS)
    if extra_metrics:
        metrics.update(extra_metrics)
    # Only keep metric definitions that actually have data.
    present = {k for row in scores.values() for k, v in row.items() if _finite(v)}
    metrics = {k: v for k, v in metrics.items() if k in present}

    return LeaderboardData(
        name=name,
        models=models,
        metrics=metrics,
        scores=scores,
        task_scores=task_scores,
    )


def _model_name(run: Any, by: str) -> str | None:
    params = getattr(run, "params", None) or {}
    val = params.get(by)
    if val is None:
        # Fall back to the evaluated_subject helper (first LLMChat param).
        val = getattr(run, "evaluated_subject", None)
    if val is None:
        return None
    return getattr(val, "name", None) or str(val)


def _run_passed(run: Any) -> bool:
    try:
        return bool(run.passed)
    except Exception:
        return False


def _run_cost_usd(run: Any) -> float | None:
    usage = _run_usage(run)
    if usage is None or usage.total_cost_nanodollars is None:
        return None
    return usage.total_cost_nanodollars / 1e9


def _run_latency_ms(run: Any) -> float | None:
    usage = _run_usage(run)
    if usage is None:
        return None
    return usage.total_backend_latency_ms


def _run_tokens(run: Any, kind: str) -> int | None:
    usage = _run_usage(run)
    if usage is None:
        return None
    return getattr(usage, f"{kind}_tokens", None)


def _run_usage(run: Any):
    chat = getattr(run, "chat", None)
    if chat is None:
        return None
    try:
        return chat.usage
    except Exception:
        return None
