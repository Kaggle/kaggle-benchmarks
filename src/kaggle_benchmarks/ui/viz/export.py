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

"""Export helpers for shareability (FR3.1 CSV, FR3.2 PNG/SVG).

These wrap Bokeh/pandas so the dashboard's download buttons -- and any script
-- can turn a chart or the underlying data into a shareable artifact with one
call. PNG rendering needs a headless browser (Playwright/Selenium); when it is
unavailable we degrade gracefully to SVG rather than raising, so the rest of
the UI keeps working.
"""

from __future__ import annotations

import io

import pandas as pd
from bokeh.plotting import figure

from kaggle_benchmarks.ui.viz.data import LeaderboardData


class ExportUnavailableError(RuntimeError):
    """Raised when server-side image export needs a browser driver it lacks.

    The chart builders use the SVG output backend, so the primary FR3.2
    "one-click export" happens *client-side* via the Bokeh toolbar's save
    button (no driver required). These server-side helpers are an extra for
    scripts/pipelines and depend on selenium being installed.
    """


def to_svg(fig: figure) -> str:
    """Render a Bokeh figure to a standalone SVG string (FR3.2, server-side).

    Requires the figure to use the SVG backend (the chart builders set this)
    plus a selenium webdriver. Raises :class:`ExportUnavailableError` with a
    clear message when the driver is missing so callers can fall back to
    :func:`to_html` or the client-side toolbar export.
    """
    fig.output_backend = "svg"
    try:
        from bokeh.io.export import get_svg

        svgs = get_svg(fig)
    except Exception as exc:  # missing selenium/webdriver
        raise ExportUnavailableError(
            "Server-side SVG export needs selenium. Use the chart toolbar's "
            "save button for one-click SVG export, or to_html() for a "
            "self-contained interactive export."
        ) from exc
    return svgs[0] if svgs else ""


def to_html(fig: figure, *, title: str = "Kaggle Benchmark") -> str:
    """Render a figure to a self-contained, interactive HTML document.

    Always available (no browser driver needed). The result is openable in any
    browser and screenshottable, making it a reliable shareable artifact and
    the dashboard's default chart-export fallback.
    """
    from bokeh.embed import file_html
    from bokeh.resources import CDN

    return file_html(fig, CDN, title)


def to_png(fig: figure, *, scale: float = 2.0) -> bytes | None:
    """Render a Bokeh figure to high-resolution PNG bytes (FR3.2).

    ``scale`` (>1) yields a retina/high-DPI image that stays crisp when pasted
    into a social post. Returns ``None`` if no headless browser is available so
    callers can fall back to :func:`to_svg` instead of crashing.
    """
    try:
        from bokeh.io.export import get_screenshot_as_png
    except Exception:
        return None
    try:
        image = get_screenshot_as_png(fig, scale_factor=scale)
        buffer = io.BytesIO()
        image.save(buffer, format="png")
        return buffer.getvalue()
    except Exception:
        # Typically a missing/unlaunchable browser driver; degrade to SVG.
        return None


def to_csv(data: LeaderboardData) -> str:
    """Serialize the full model x metric table to CSV text (FR3.1).

    This is the "download the full benchmark dataset" payload -- it exposes
    every captured metric, not just the headline score, which is the point of
    making the data's richness visible.
    """
    return data.as_dataframe().to_csv()


def task_matrix_csv(data: LeaderboardData) -> str:
    """Serialize the model x task success matrix to CSV (FR3.1, task-level).

    Serves CUJ 2's "download the full benchmark with task-level metrics as a
    CSV". Returns an empty string when no per-task data is available.
    """
    matrix: pd.DataFrame = data.task_matrix()
    if matrix.empty:
        return ""
    return matrix.to_csv()
