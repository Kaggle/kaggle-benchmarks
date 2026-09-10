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

"""Pareto-frontier computation for trade-off charts.

The Pareto frontier is the single most important visual element for any chart
that pits a quality metric against a resource constraint (cost, latency,
tokens) -- see the "Pareto prominence" principle in the visualizations PRD.

This module is deliberately dependency-free (pure Python) so it is trivial to
unit test and reuse outside of the Bokeh charts.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

Direction = Literal["max", "min"]


def _dominates(
    a: tuple[float, float],
    b: tuple[float, float],
    x_dir: Direction,
    y_dir: Direction,
) -> bool:
    """Return True if point ``a`` Pareto-dominates point ``b``.

    ``a`` dominates ``b`` when it is at least as good on both axes and strictly
    better on at least one, where "better" respects each axis' optimization
    direction (higher is better for ``"max"``, lower for ``"min"``).
    """

    def at_least_as_good(av: float, bv: float, direction: Direction) -> bool:
        return av >= bv if direction == "max" else av <= bv

    def strictly_better(av: float, bv: float, direction: Direction) -> bool:
        return av > bv if direction == "max" else av < bv

    return (
        at_least_as_good(a[0], b[0], x_dir)
        and at_least_as_good(a[1], b[1], y_dir)
        and (strictly_better(a[0], b[0], x_dir) or strictly_better(a[1], b[1], y_dir))
    )


def pareto_indices(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    x_dir: Direction = "min",
    y_dir: Direction = "max",
) -> list[int]:
    """Return the indices of the Pareto-optimal points.

    Args:
        xs: X-axis values (typically a resource constraint like cost).
        ys: Y-axis values (typically a quality metric like accuracy).
        x_dir: ``"min"`` if smaller x is better (the default -- costs), else
            ``"max"``.
        y_dir: ``"max"`` if larger y is better (the default -- quality), else
            ``"min"``.

    Points with a non-finite (``None``/``NaN``) coordinate are skipped -- a
    model missing one of the plotted metrics cannot be on the frontier.

    The returned indices reference the original input positions and are sorted
    by ascending x for convenient line drawing.
    """
    if len(xs) != len(ys):
        raise ValueError(f"xs and ys length mismatch: {len(xs)} != {len(ys)}")

    points: list[tuple[int, float, float]] = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        if x is None or y is None:
            continue
        # Reject NaN (x != x is only true for NaN).
        if x != x or y != y:  # noqa: PLR0124
            continue
        points.append((i, float(x), float(y)))

    frontier: list[int] = []
    for i, xi, yi in points:
        dominated = any(
            _dominates((xj, yj), (xi, yi), x_dir, y_dir)
            for j, xj, yj in points
            if j != i
        )
        if not dominated:
            frontier.append(i)

    # Sort by ascending x so a connecting line steps cleanly across the chart.
    # Ties on x are broken by y so the ordering is deterministic.
    index_to_point = {i: (x, y) for i, x, y in points}
    frontier.sort(key=lambda i: (index_to_point[i][0], index_to_point[i][1]))
    return frontier
