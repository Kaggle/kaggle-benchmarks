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

"""One-click image export for charts (PRD FR3.2).

Bokeh can serialize a figure to a self-contained interactive HTML file with no
external tooling, and to high-resolution PNG/SVG when a headless browser is
available (Kaggle notebooks and CI ship one). These helpers wrap those paths so
the dashboard's export buttons "just work" where a browser exists and fail with
an actionable message where it doesn't -- while the always-present in-browser
Bokeh *save* toolbar button remains a guaranteed fallback.
"""

from __future__ import annotations

import io

from bokeh.plotting import figure as _Figure


def to_html(fig: _Figure) -> str:
    """Serialize a chart to a standalone, self-contained HTML document.

    Always available (pure Python, no browser needed). Great for embedding a
    live, interactive chart in a blog post or sharing as a file.
    """
    from bokeh.embed import file_html
    from bokeh.resources import CDN

    return file_html(fig, CDN, "Kaggle benchmark chart")


def to_svg(fig: _Figure) -> str:
    """Render a chart to a single SVG string (vector, infinitely scalable).

    Requires a headless browser driver (Selenium + geckodriver/chromedriver).
    Raises :class:`ExportUnavailable` with guidance when none is available.
    """
    driver = _webdriver_or_raise()
    fig.output_backend = "svg"
    try:
        from bokeh.io.export import get_svg

        svgs = get_svg(fig, driver=driver)
        return svgs[0] if isinstance(svgs, list) else svgs
    finally:
        driver.quit()


def to_png(fig: _Figure, *, scale: float = 2.0) -> bytes:
    """Render a chart to high-resolution PNG bytes.

    ``scale`` multiplies the pixel density (2.0 -> retina quality). Requires a
    headless browser driver; raises :class:`ExportUnavailable` otherwise.
    """
    driver = _webdriver_or_raise()
    try:
        from bokeh.io.export import get_screenshot_as_png

        image = get_screenshot_as_png(fig, driver=driver, scale_factor=scale)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        driver.quit()


class ExportUnavailable(RuntimeError):
    """Raised when PNG/SVG export is requested without a headless browser."""


def webdriver_available() -> bool:
    """Whether PNG/SVG raster/vector export can run in this environment.

    Lets callers (e.g. the dashboard) disable the PNG/SVG buttons up front
    instead of surfacing an error only after a click.
    """
    try:
        _webdriver_or_raise().quit()
        return True
    except Exception:
        return False


def _webdriver_or_raise():
    try:
        from bokeh.io.webdriver import webdriver_control

        return webdriver_control.create()
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ExportUnavailable(
            "PNG/SVG export needs a headless browser (Selenium + a Chrome or "
            "Firefox driver). Install it, or use the chart toolbar's save "
            "button for an in-browser PNG, or export_html() for a shareable "
            "interactive file."
        ) from exc
