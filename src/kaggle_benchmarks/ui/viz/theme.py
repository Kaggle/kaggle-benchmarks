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

"""Centralized theming for the native benchmark chart library (FR1.1).

This module is the single source of truth for the "premium" Kaggle look:
brand colors, dark/light palettes, and shared Bokeh figure defaults. Every
chart builder in this package pulls its styling from here so that all
benchmark visualizations stay visually consistent and dark-mode compatible.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Iterator, Sequence

# Kaggle brand blue and a complementary categorical palette. The palette is
# deliberately vivid and high-contrast so a single chart reads well when
# screenshotted onto a social feed.
KAGGLE_BLUE = "#20BEFF"
KAGGLE_BLUE_DARK = "#0A9BD8"

# Categorical palette for series (models). Ordered for maximum adjacent
# contrast; wraps around via ``color_cycle`` when there are more series.
CATEGORICAL_PALETTE: tuple[str, ...] = (
    "#20BEFF",  # kaggle blue
    "#FF6F59",  # coral
    "#7B61FF",  # violet
    "#28C76F",  # green
    "#FFB020",  # amber
    "#FF5CAD",  # pink
    "#00C2C7",  # teal
    "#A0D911",  # lime
    "#F5476A",  # rose
    "#5B8DEF",  # periwinkle
)

# Sequential ramp for heatmaps (low -> high success). Runs from a muted
# red through neutral to Kaggle green so failures and successes are obvious
# in both light and dark mode.
SEQUENTIAL_RAMP: tuple[str, ...] = (
    "#8B1E3F",
    "#C0392B",
    "#E67E22",
    "#F1C40F",
    "#A0D911",
    "#28C76F",
    "#0E8A4F",
)

# The Pareto frontier gets its own unmistakable accent so it stays the most
# prominent element of any trade-off chart (see PRD "Pareto prominence").
PARETO_COLOR = "#FFB020"


@dataclasses.dataclass(frozen=True)
class Theme:
    """Resolved colors for a single light/dark mode.

    Passed to the chart builders so a figure can be styled without every
    builder re-deriving the palette.
    """

    name: str
    background: str
    surface: str
    text: str
    muted_text: str
    grid: str
    axis: str
    palette: tuple[str, ...] = CATEGORICAL_PALETTE
    sequential: tuple[str, ...] = SEQUENTIAL_RAMP
    pareto: str = PARETO_COLOR
    brand: str = KAGGLE_BLUE

    def color_cycle(self) -> Iterator[str]:
        """Infinite iterator over the categorical palette (wraps around)."""
        return itertools.cycle(self.palette)

    def colors_for(self, labels: Sequence[str]) -> dict[str, str]:
        """Deterministic label -> color mapping.

        The same label always gets the same color within one call, and colors
        wrap around the palette when there are more labels than colors.
        """
        cycle = self.color_cycle()
        return {label: next(cycle) for label in labels}


DARK = Theme(
    name="dark",
    background="#0B0F14",
    surface="#141A21",
    text="#E8EDF2",
    muted_text="#8A97A6",
    grid="#1E2833",
    axis="#3A4756",
)

LIGHT = Theme(
    name="light",
    background="#FFFFFF",
    surface="#F7F9FB",
    text="#0B1420",
    muted_text="#5A6B7B",
    grid="#E4E9EF",
    axis="#C2CCD6",
)

_THEMES = {"dark": DARK, "light": LIGHT}

# Dark mode is the default: it is the more "premium"/screenshottable look and
# matches the reference points cited in the PRD (Artificial Analysis, LMSYS).
DEFAULT_THEME = "dark"


def resolve_theme(theme: str | Theme | None) -> Theme:
    """Coerce a theme name (or ``Theme``) into a ``Theme`` instance."""
    if theme is None:
        return _THEMES[DEFAULT_THEME]
    if isinstance(theme, Theme):
        return theme
    key = theme.lower()
    if key not in _THEMES:
        raise ValueError(
            f"Unknown theme {theme!r}. Available themes: {sorted(_THEMES)}"
        )
    return _THEMES[key]


# Base font stack shared across every chart. Matches Kaggle's product UI.
FONT = "Inter, Helvetica Neue, Helvetica, Arial, sans-serif"


def style_figure(fig, theme: Theme, *, hide_grid: bool = False) -> None:
    """Apply the shared premium styling to a Bokeh figure in place.

    Centralizing this keeps axis colors, grid, fonts, and background
    consistent across all chart types so the library reads as one cohesive
    system rather than a grab-bag of Bokeh defaults.
    """
    fig.background_fill_color = theme.surface
    fig.border_fill_color = theme.background
    fig.outline_line_color = None

    fig.title.text_color = theme.text
    fig.title.text_font = FONT
    fig.title.text_font_size = "15px"
    fig.title.text_font_style = "bold"

    for axis in fig.axis:
        axis.axis_label_text_color = theme.muted_text
        axis.axis_label_text_font = FONT
        axis.axis_label_text_font_style = "normal"
        axis.major_label_text_color = theme.muted_text
        axis.major_label_text_font = FONT
        axis.axis_line_color = theme.axis
        axis.major_tick_line_color = theme.axis
        axis.minor_tick_line_color = None

    for grid in fig.grid:
        if hide_grid:
            grid.grid_line_color = None
        else:
            grid.grid_line_color = theme.grid
            grid.grid_line_alpha = 0.6

    if fig.legend:
        for legend in fig.legend:
            legend.background_fill_color = theme.surface
            legend.background_fill_alpha = 0.85
            legend.border_line_color = theme.grid
            legend.label_text_color = theme.text
            legend.label_text_font = FONT
            legend.label_text_font_size = "11px"
