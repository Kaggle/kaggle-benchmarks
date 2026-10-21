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

"""Kaggle-branded visual theme for the benchmark chart library (FR1.1).

This module is the single source of truth for the colors, fonts, and Bokeh
styling used across every chart in :mod:`kaggle_benchmarks.ui.visualizations`.
Centralizing it here is what guarantees the "premium, cohesive, dark-mode
compatible" aesthetic the PRD calls for: a chart never hard-codes a color, it
asks the active :class:`Palette` for one.
"""

from __future__ import annotations

import dataclasses

# Kaggle brand blue and a complementary categorical ramp used to color models /
# series consistently across every chart. The ramp is colorblind-aware and holds
# up on both light and dark backgrounds.
KAGGLE_BLUE = "#20BEFF"

_CATEGORICAL = (
    "#20BEFF",  # kaggle blue
    "#FF7F0E",  # orange
    "#2CA02C",  # green
    "#D62728",  # red
    "#9467BD",  # purple
    "#00C1B4",  # teal
    "#E377C2",  # pink
    "#BCBD22",  # olive
    "#1F77B4",  # deep blue
    "#8C564B",  # brown
)


@dataclasses.dataclass(frozen=True)
class Palette:
    """Resolved colors for a single light/dark mode.

    Charts pull every color they draw from an instance of this class so that a
    theme switch is a one-line change and never leaves a stray hard-coded color
    behind.
    """

    name: str
    background: str
    surface: str
    text: str
    muted_text: str
    grid: str
    axis: str
    accent: str
    # Frontier / Pareto highlight color. Deliberately warm so the frontier
    # "pops" against the blue categorical ramp (Pareto prominence principle).
    frontier: str
    # Diverging heatmap endpoints (low -> high success rate).
    heat_low: str
    heat_high: str
    categorical: tuple[str, ...] = _CATEGORICAL

    def color_for(self, index: int) -> str:
        """Stable categorical color for the ``index``-th series."""
        return self.categorical[index % len(self.categorical)]

    def color_map(self, keys) -> dict[str, str]:
        """Map an ordered iterable of series keys to stable colors."""
        return {key: self.color_for(i) for i, key in enumerate(keys)}


LIGHT = Palette(
    name="light",
    background="#FFFFFF",
    surface="#F5F7FA",
    text="#1A1A1A",
    muted_text="#5F6B7A",
    grid="#E4E8EC",
    axis="#B0B8C1",
    accent=KAGGLE_BLUE,
    frontier="#FF7F0E",
    heat_low="#F5F7FA",
    heat_high=KAGGLE_BLUE,
)

DARK = Palette(
    name="dark",
    background="#0E1117",
    surface="#161B22",
    text="#E6EDF3",
    muted_text="#9DA7B3",
    grid="#21262D",
    axis="#3D444D",
    accent=KAGGLE_BLUE,
    frontier="#FFA94D",
    heat_low="#161B22",
    heat_high=KAGGLE_BLUE,
)

PALETTES = {"light": LIGHT, "dark": DARK}


def get_palette(theme: str | None = None) -> Palette:
    """Return the :class:`Palette` for ``theme``.

    ``theme`` defaults to the SDK's configured ``ui_theme`` so charts match the
    surrounding notebook / page automatically. Unknown themes fall back to
    light mode rather than raising, so a chart never fails to render over a
    theming mistake.
    """
    if theme is None:
        try:
            from kaggle_benchmarks._config import config

            theme = config.ui_theme
        except Exception:
            theme = "light"
    # The SDK config uses "default" to mean the light theme.
    normalized = (theme or "light").lower()
    if normalized == "default":
        normalized = "light"
    return PALETTES.get(normalized, LIGHT)


# Shared typography. Bokeh wants font strings, so keep this simple.
FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"


def style_figure(fig, palette: Palette, *, title: str | None = None) -> None:
    """Apply the Kaggle house style to a Bokeh figure in place.

    Every chart builder funnels through this so borders, gridlines, fonts, and
    the watermark are identical across chart types.
    """
    fig.background_fill_color = palette.background
    fig.border_fill_color = palette.background
    fig.outline_line_color = None

    if title is not None:
        fig.title.text = title
    if fig.title is not None:
        fig.title.text_color = palette.text
        fig.title.text_font = FONT
        fig.title.text_font_size = "16px"
        fig.title.text_font_style = "bold"

    for grid in (fig.xgrid, fig.ygrid):
        grid.grid_line_color = palette.grid
        grid.grid_line_alpha = 0.6
        grid.minor_grid_line_color = None

    for axis in fig.axis:
        axis.axis_line_color = palette.axis
        axis.major_tick_line_color = palette.axis
        axis.minor_tick_line_color = None
        axis.major_label_text_color = palette.muted_text
        axis.axis_label_text_color = palette.text
        axis.major_label_text_font = FONT
        axis.axis_label_text_font = FONT
        axis.axis_label_text_font_style = "normal"
        axis.axis_label_text_font_size = "13px"
