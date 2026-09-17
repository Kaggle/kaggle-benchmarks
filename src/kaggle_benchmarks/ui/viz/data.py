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

"""The data layer that decouples benchmark results from visualizations.

``LeaderboardData`` is the single structured input every chart in this package
consumes. Building it once from a set of runs (or a raw DataFrame) means each
chart -- scatter, bar table, heatmap, win-rate matrix, Elo plot -- reads from
the same normalized representation. This is what makes "dimensional
flexibility" (FR1.3) cheap: any scalar column can be mapped to any X/Y axis
without per-chart engineering.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from kaggle_benchmarks import runs

# Metric name -> the optimization direction used for Pareto frontiers and
# axis defaults. "max" means bigger is better (quality), "min" means smaller
# is better (a resource cost). Unknown metrics default to "max".
_DEFAULT_DIRECTIONS: dict[str, str] = {
    "score": "max",
    "accuracy": "max",
    "pass_rate": "max",
    "win_rate": "max",
    "elo": "max",
    "cost": "min",
    "cost_usd": "min",
    "latency_ms": "min",
    "latency": "min",
    "input_tokens": "min",
    "output_tokens": "min",
    "total_tokens": "min",
}

# Metrics that represent a resource constraint. Used to auto-pick a sensible
# default X axis for the trade-off scatter (FR1.2 defaults).
_RESOURCE_METRICS = ("cost_usd", "cost", "latency_ms", "total_tokens", "output_tokens")


def metric_direction(metric: str) -> str:
    """Return ``"max"`` or ``"min"`` optimization direction for a metric."""
    return _DEFAULT_DIRECTIONS.get(metric, "max")


@dataclasses.dataclass
class LeaderboardData:
    """Normalized, chart-ready view of a benchmark's results.

    Attributes:
        models: Ordered list of model/candidate names (one row per model).
        metrics: ``{metric_name: {model: scalar_value}}`` for every scalar
            metric available. These populate the axis dropdowns and the bar
            table / scatter plot.
        task_scores: ``{model: {task_id: score}}`` per-task success values in
            ``[0, 1]`` used by the heatmap. May be empty if unavailable.
        benchmark_name: Human-readable name for titles and exports.
    """

    models: list[str]
    metrics: dict[str, dict[str, float]]
    task_scores: dict[str, dict[Any, float]] = dataclasses.field(default_factory=dict)
    benchmark_name: str = "Benchmark"

    # ---- introspection helpers -------------------------------------------------

    @property
    def metric_names(self) -> list[str]:
        """Scalar metrics available for axis mapping, in a stable order."""
        return list(self.metrics.keys())

    @property
    def tasks(self) -> list[Any]:
        """Sorted union of task ids across all models (heatmap columns)."""
        seen: dict[Any, None] = {}
        for per_task in self.task_scores.values():
            for task_id in per_task:
                seen.setdefault(task_id, None)
        try:
            return sorted(seen)
        except TypeError:
            # Mixed/unsortable task id types -- fall back to insertion order.
            return list(seen)

    def values(self, metric: str) -> list[float | None]:
        """Metric values aligned to ``self.models`` (``None`` when missing)."""
        column = self.metrics.get(metric, {})
        return [column.get(model) for model in self.models]

    def default_axes(self) -> tuple[str, str]:
        """Pick sensible (x, y) defaults: a resource metric vs. a quality one.

        Encodes the FR1.2 default of "main metric vs. cost" so a chart is
        useful the moment the page loads, before the author configures axes.
        """
        names = self.metric_names
        if not names:
            raise ValueError("LeaderboardData has no metrics to plot.")

        quality = next((m for m in names if metric_direction(m) == "max"), names[0])
        x = next((m for m in _RESOURCE_METRICS if m in names), None)
        if x is None:
            x = next((m for m in names if m != quality), quality)
        return x, quality

    def as_dataframe(self) -> pd.DataFrame:
        """Wide model x metric table -- the CSV download payload (FR3.1)."""
        frame = pd.DataFrame({"model": self.models})
        for metric in self.metric_names:
            frame[metric] = [self.metrics[metric].get(m) for m in self.models]
        return frame.set_index("model")

    def task_matrix(self) -> pd.DataFrame:
        """Model x task success matrix (rows=models, cols=tasks) for heatmaps."""
        tasks = self.tasks
        rows = {
            model: [self.task_scores.get(model, {}).get(t) for t in tasks]
            for model in self.models
        }
        return pd.DataFrame.from_dict(rows, orient="index", columns=tasks)

    def win_rate_matrix(self) -> pd.DataFrame:
        """Pairwise win-rate matrix derived from per-task scores (FR2.4).

        ``cell[a][b]`` is the fraction of shared tasks where model ``a``
        scored strictly higher than model ``b``. Ties count as half a win to
        each side, matching the usual head-to-head convention. Requires
        per-task data; returns an empty frame otherwise.
        """
        if not self.task_scores:
            return pd.DataFrame()

        matrix = pd.DataFrame(index=self.models, columns=self.models, dtype=float)
        for a in self.models:
            for b in self.models:
                if a == b:
                    matrix.loc[a, b] = math.nan
                    continue
                matrix.loc[a, b] = _pairwise_win_rate(
                    self.task_scores.get(a, {}), self.task_scores.get(b, {})
                )
        return matrix

    # ---- constructors ----------------------------------------------------------

    @classmethod
    def from_runs(
        cls,
        runs: "runs.Runs",
        *,
        model_by: str = "llm",
        benchmark_name: str = "Benchmark",
    ) -> "LeaderboardData":
        """Build a ``LeaderboardData`` from a completed ``Runs`` collection.

        Aggregates, per model:
          * ``score`` -- mean of ``run.passed`` (accuracy / pass rate)
          * ``cost_usd`` -- mean total cost per run (dollars)
          * ``latency_ms`` -- mean backend latency per run
          * ``input_tokens`` / ``output_tokens`` / ``total_tokens`` -- means

        Only metrics that have at least one real value are kept, so a benchmark
        that never recorded cost will simply not offer a cost axis.

        The per-task matrix uses ``run.param_id`` (the dataset index) as the
        task id, which lets the heatmap show model x task success.
        """
        # model -> list of per-run records
        by_model: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []

        for run in runs.runs:
            model = _model_name(run, model_by)
            if model is None:
                continue
            if model not in by_model:
                by_model[model] = []
                order.append(model)
            by_model[model].append(_run_record(run))

        metrics: dict[str, dict[str, float]] = {}
        task_scores: dict[str, dict[Any, float]] = {}

        metric_keys = (
            "score",
            "cost_usd",
            "latency_ms",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        )
        for model in order:
            records = by_model[model]
            for key in metric_keys:
                vals = [r[key] for r in records if r.get(key) is not None]
                if vals:
                    metrics.setdefault(key, {})[model] = sum(vals) / len(vals)

            per_task = {
                r["task_id"]: float(r["passed"])
                for r in records
                if r.get("task_id") is not None
            }
            if per_task:
                task_scores[model] = per_task

        return cls(
            models=order,
            metrics=metrics,
            task_scores=task_scores,
            benchmark_name=benchmark_name,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        model_column: str = "model",
        metric_columns: Sequence[str] | None = None,
        task_scores: Mapping[str, Mapping[Any, float]] | None = None,
        benchmark_name: str = "Benchmark",
    ) -> "LeaderboardData":
        """Build directly from a wide model x metric DataFrame.

        Every numeric column other than ``model_column`` becomes a selectable
        metric unless ``metric_columns`` restricts the set. This is the
        convenient path for prototyping charts from an already-aggregated
        leaderboard table.
        """
        if model_column not in df.columns:
            raise ValueError(f"model column {model_column!r} not in DataFrame")

        models = [str(m) for m in df[model_column].tolist()]

        if metric_columns is None:
            metric_columns = [
                c
                for c in df.columns
                if c != model_column and pd.api.types.is_numeric_dtype(df[c])
            ]

        metrics: dict[str, dict[str, float]] = {}
        for col in metric_columns:
            column: dict[str, float] = {}
            for model, value in zip(models, df[col].tolist()):
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    continue
                column[model] = float(value)
            if column:
                metrics[str(col)] = column

        normalized_tasks: dict[str, dict[Any, float]] = {}
        if task_scores:
            for model, per_task in task_scores.items():
                normalized_tasks[str(model)] = {
                    task_id: float(v) for task_id, v in per_task.items()
                }

        return cls(
            models=models,
            metrics=metrics,
            task_scores=normalized_tasks,
            benchmark_name=benchmark_name,
        )


def _pairwise_win_rate(
    a_scores: Mapping[Any, float], b_scores: Mapping[Any, float]
) -> float:
    """Fraction of shared tasks where ``a`` beats ``b`` (ties = 0.5)."""
    shared = set(a_scores) & set(b_scores)
    if not shared:
        return math.nan
    wins = 0.0
    for task_id in shared:
        av, bv = a_scores[task_id], b_scores[task_id]
        if av > bv:
            wins += 1.0
        elif av == bv:
            wins += 0.5
    return wins / len(shared)


def _model_name(run: Any, model_by: str) -> str | None:
    """Extract the model/candidate name from a run.

    Prefers the explicitly requested param (``model_by``); falls back to the
    run's detected ``evaluated_subject`` so it works even when the param key
    differs from the default.
    """
    params = getattr(run, "params", {}) or {}
    if model_by in params:
        value = params[model_by]
        return getattr(value, "name", None) or str(value)

    subject = getattr(run, "evaluated_subject", None)
    if subject is not None:
        return getattr(subject, "model", None) or getattr(subject, "name", None)
    return None


def _run_record(run: Any) -> dict[str, Any]:
    """Flatten a single run into the scalar fields the data layer aggregates."""
    record: dict[str, Any] = {
        "task_id": getattr(run, "param_id", None),
        "passed": bool(getattr(run, "passed", False)),
        "score": 1.0 if getattr(run, "passed", False) else 0.0,
    }

    chat = getattr(run, "chat", None)
    usage = getattr(chat, "usage", None) if chat is not None else None
    if usage is not None:
        cost_nano = usage.total_cost_nanodollars
        record["cost_usd"] = cost_nano / 1e9 if cost_nano is not None else None
        record["latency_ms"] = usage.total_backend_latency_ms
        record["input_tokens"] = usage.input_tokens
        record["output_tokens"] = usage.output_tokens
        if usage.input_tokens is not None or usage.output_tokens is not None:
            record["total_tokens"] = (usage.input_tokens or 0) + (
                usage.output_tokens or 0
            )
    return record


def infer_task_scores_from_runs(
    runs: Iterable[Any], model_by: str = "llm"
) -> dict[str, dict[Any, float]]:
    """Standalone helper mirroring the per-task extraction in ``from_runs``.

    Exposed for callers that already have scalar metrics elsewhere but want the
    heatmap/win-rate matrix data.
    """
    out: dict[str, dict[Any, float]] = {}
    for run in runs:
        model = _model_name(run, model_by)
        task_id = getattr(run, "param_id", None)
        if model is None or task_id is None:
            continue
        out.setdefault(model, {})[task_id] = (
            1.0 if getattr(run, "passed", False) else 0.0
        )
    return out
