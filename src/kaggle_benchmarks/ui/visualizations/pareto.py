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

"""Pareto-frontier computation (PRD "Pareto prominence").

Kept deliberately pure and dependency-free (no Bokeh, no pandas) so it is
trivial to unit test and reuse. A point is on the frontier if no other point is
at least as good on every axis and strictly better on at least one, where
"better" is defined per-axis by a direction flag.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class PointND:
    """A labeled point in the trade-off space."""

    label: str
    x: float
    y: float


def _dominates(
    a: PointND,
    b: PointND,
    *,
    x_higher_is_better: bool,
    y_higher_is_better: bool,
) -> bool:
    """Return ``True`` if ``a`` Pareto-dominates ``b``.

    ``a`` dominates ``b`` when it is no worse than ``b`` on both axes and
    strictly better on at least one, accounting for each axis's direction.
    """

    def better_or_equal(av: float, bv: float, higher: bool) -> bool:
        return av >= bv if higher else av <= bv

    def strictly_better(av: float, bv: float, higher: bool) -> bool:
        return av > bv if higher else av < bv

    no_worse = better_or_equal(a.x, b.x, x_higher_is_better) and better_or_equal(
        a.y, b.y, y_higher_is_better
    )
    strictly = strictly_better(a.x, b.x, x_higher_is_better) or strictly_better(
        a.y, b.y, y_higher_is_better
    )
    return no_worse and strictly


def pareto_frontier(
    points: list[PointND],
    *,
    x_higher_is_better: bool,
    y_higher_is_better: bool,
) -> list[str]:
    """Return the labels of the non-dominated (Pareto-optimal) points.

    Ties (identical coordinates) are all kept on the frontier so co-located
    best models are highlighted together rather than one arbitrarily dropped.
    """
    frontier: list[str] = []
    for candidate in points:
        dominated = any(
            other is not candidate
            and _dominates(
                other,
                candidate,
                x_higher_is_better=x_higher_is_better,
                y_higher_is_better=y_higher_is_better,
            )
            for other in points
        )
        if not dominated:
            frontier.append(candidate.label)
    return frontier


def frontier_line(
    points: list[PointND],
    frontier_labels: list[str],
    *,
    x_higher_is_better: bool,
) -> list[PointND]:
    """Order the frontier points into a polyline for drawing.

    Sorting by x (in the direction that improves quality) yields a monotone
    staircase suitable for a connected line under the scatter markers.
    """
    on_frontier = [p for p in points if p.label in set(frontier_labels)]
    return sorted(on_frontier, key=lambda p: p.x, reverse=not x_higher_is_better)
