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

import math

import pytest

from kaggle_benchmarks.ui.viz.pareto import pareto_indices


def test_pareto_min_cost_max_quality():
    # cost (min better) vs quality (max better)
    # A: cheap+good, B: expensive+good, C: cheap+bad, D: dominated everywhere
    costs = [0.01, 0.10, 0.01, 0.20]
    quality = [0.9, 0.95, 0.5, 0.4]
    #   A (0.01, 0.9): on frontier (cheapest of the high-quality-ish)
    #   B (0.10, 0.95): on frontier (best quality)
    #   C (0.01, 0.5): dominated by A (same cost, higher quality)
    #   D (0.20, 0.4): dominated by everything
    frontier = set(pareto_indices(costs, quality, x_dir="min", y_dir="max"))
    assert frontier == {0, 1}


def test_pareto_sorted_ascending_x():
    costs = [0.10, 0.01]
    quality = [0.95, 0.9]
    # Both are Pareto optimal; result must be sorted by ascending x.
    frontier = pareto_indices(costs, quality, x_dir="min", y_dir="max")
    assert frontier == [1, 0]  # index 1 has x=0.01, index 0 has x=0.10


def test_pareto_all_on_frontier_when_strictly_trading_off():
    costs = [1, 2, 3, 4]
    quality = [0.4, 0.6, 0.8, 0.9]
    frontier = pareto_indices(costs, quality, x_dir="min", y_dir="max")
    assert frontier == [0, 1, 2, 3]


def test_pareto_single_dominator():
    costs = [1, 2, 3]
    quality = [0.9, 0.5, 0.2]
    # Point 0 dominates the rest (cheaper AND better).
    frontier = pareto_indices(costs, quality, x_dir="min", y_dir="max")
    assert frontier == [0]


def test_pareto_direction_max_max():
    # Both axes: bigger is better (e.g. two quality metrics).
    a = [0.9, 0.8, 0.1]
    b = [0.1, 0.85, 0.05]
    frontier = set(pareto_indices(a, b, x_dir="max", y_dir="max"))
    # index 2 is dominated by both others.
    assert 2 not in frontier
    assert frontier == {0, 1}


def test_pareto_skips_none_and_nan():
    xs = [0.1, None, 0.2, float("nan")]
    ys = [0.9, 0.5, math.nan, 0.4]
    frontier = pareto_indices(xs, ys, x_dir="min", y_dir="max")
    # Only index 0 has both coordinates finite.
    assert frontier == [0]


def test_pareto_empty():
    assert pareto_indices([], []) == []


def test_pareto_length_mismatch_raises():
    with pytest.raises(ValueError):
        pareto_indices([1, 2], [1])


def test_pareto_ties_do_not_dominate_each_other():
    # Identical points: neither strictly dominates, so both survive.
    xs = [0.1, 0.1]
    ys = [0.9, 0.9]
    frontier = pareto_indices(xs, ys, x_dir="min", y_dir="max")
    assert set(frontier) == {0, 1}
