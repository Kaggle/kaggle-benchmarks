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

import pytest
from bokeh.plotting import figure

from kaggle_benchmarks.ui.viz import theme as theme_mod


def test_resolve_theme_default_is_dark():
    assert theme_mod.resolve_theme(None).name == "dark"


def test_resolve_theme_by_name():
    assert theme_mod.resolve_theme("light").name == "light"
    assert theme_mod.resolve_theme("DARK").name == "dark"


def test_resolve_theme_passthrough():
    assert theme_mod.resolve_theme(theme_mod.LIGHT) is theme_mod.LIGHT


def test_resolve_theme_unknown_raises():
    with pytest.raises(ValueError):
        theme_mod.resolve_theme("neon")


def test_colors_for_is_deterministic_and_wraps():
    t = theme_mod.DARK
    labels = [f"m{i}" for i in range(len(t.palette) + 3)]
    colors = t.colors_for(labels)
    # every label mapped
    assert set(colors) == set(labels)
    # palette wraps: label N+0 and label at wrap share the first color
    assert colors[labels[0]] == colors[labels[len(t.palette)]]


def test_style_figure_applies_theme_colors():
    t = theme_mod.DARK
    fig = figure()
    theme_mod.style_figure(fig, t)
    assert fig.background_fill_color == t.surface
    assert fig.title.text_color == t.text


def test_style_figure_hide_grid():
    fig = figure()
    theme_mod.style_figure(fig, theme_mod.LIGHT, hide_grid=True)
    for grid in fig.grid:
        assert grid.grid_line_color is None
