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

"""The embeddable benchmark-visualization dashboard (FR1.2, FR1.3, FR3.*).

``BenchmarkDashboard`` is the top-level, screenshottable component. It puts the
multi-dimensional charts front and center (per the "prominent and discoverable
placement" principle): fast-access view chips select the chart type, axis
dropdowns re-map any scalar metric to X/Y (dimensional flexibility), and
download/export buttons cover CSV + PNG/SVG shareability.

The chart config (view + axes) is also serialized to a deep-link query string
(``to_query_string`` / ``from_query_string``), the groundwork for FR4.1
deep-linking and OpenGraph previews.
"""

from __future__ import annotations

import io
import urllib.parse
from typing import Any

import panel as pn

from kaggle_benchmarks.ui.viz import charts as charts_mod
from kaggle_benchmarks.ui.viz import export as export_mod
from kaggle_benchmarks.ui.viz import theme as theme_mod
from kaggle_benchmarks.ui.viz.data import LeaderboardData


class BenchmarkDashboard:
    """Interactive dashboard over a :class:`LeaderboardData`.

    Args:
        data: The normalized benchmark data to visualize.
        theme: ``"dark"`` (default) or ``"light"``, or a ``Theme``.
        view: Initial chart id (see ``charts.CHART_BUILDERS``). Defaults to the
            Pareto scatter when available, else the bar leaderboard.
        x, y: Initial axis metrics for XY charts (defaults chosen from data).
    """

    def __init__(
        self,
        data: LeaderboardData,
        *,
        theme: str | theme_mod.Theme | None = None,
        view: str | None = None,
        x: str | None = None,
        y: str | None = None,
    ) -> None:
        self.data = data
        self.theme = theme_mod.resolve_theme(theme)
        self._available = charts_mod.available_charts(data)
        if not self._available:
            raise ValueError("LeaderboardData has nothing to visualize.")

        self.view = view if view in self._available else self._default_view()
        default_x, default_y = (
            data.default_axes() if len(data.metric_names) >= 2 else (None, None)
        )
        self.x = x or default_x
        self.y = y or default_y

    # ---- config / deep-linking -------------------------------------------------

    def _default_view(self) -> str:
        for preferred in ("scatter", "bars"):
            if preferred in self._available:
                return preferred
        return self._available[0]

    def config(self) -> dict[str, str]:
        """The current chart configuration (view + axes)."""
        cfg = {"view": self.view}
        if self.x:
            cfg["x"] = self.x
        if self.y:
            cfg["y"] = self.y
        return cfg

    def to_query_string(self) -> str:
        """Serialize the chart config to a URL query string (FR4.1 groundwork).

        Sharing a URL carrying this query string reproduces the exact chart the
        author configured.
        """
        return urllib.parse.urlencode(self.config())

    @classmethod
    def from_query_string(
        cls,
        data: LeaderboardData,
        query: str,
        *,
        theme: str | theme_mod.Theme | None = None,
    ) -> "BenchmarkDashboard":
        """Rebuild a dashboard from a deep-link query string."""
        params = urllib.parse.parse_qs(query.lstrip("?"))

        def first(key: str) -> str | None:
            values = params.get(key)
            return values[0] if values else None

        return cls(
            data,
            theme=theme,
            view=first("view"),
            x=first("x"),
            y=first("y"),
        )

    # ---- rendering -------------------------------------------------------------

    def build_chart(self):
        """Build the currently-selected Bokeh figure."""
        builder = charts_mod.CHART_BUILDERS[self.view]
        if self.view == "scatter":
            return builder(self.data, self.x, self.y, theme=self.theme)
        return builder(self.data, theme=self.theme)

    def __panel__(self) -> pn.viewable.Viewable:
        return self._layout()

    def _repr_mimebundle_(self, include=None, exclude=None):
        return self._layout()._repr_mimebundle_(include, exclude)

    def _layout(self) -> pn.viewable.Viewable:
        # View chips: prominent, fast-access chart-type selector placed above
        # the chart so multi-dimensional views are immediately discoverable.
        chip_options = {
            charts_mod.CHART_LABELS.get(cid, cid): cid for cid in self._available
        }
        view_chips = pn.widgets.RadioButtonGroup(
            name="View",
            options=chip_options,
            value=self.view,
            button_type="primary",
            button_style="outline",
        )

        # Axis dropdowns (FR1.3) -- only meaningful for the XY scatter.
        metric_options = {charts_mod._axis_label(m): m for m in self.data.metric_names}
        x_select = pn.widgets.Select(
            name="X axis", options=metric_options, value=self.x, width=200
        )
        y_select = pn.widgets.Select(
            name="Y axis", options=metric_options, value=self.y, width=200
        )

        chart_pane = pn.pane.Bokeh(self.build_chart(), sizing_mode="stretch_width")

        def refresh(*_events: Any) -> None:
            self.view = view_chips.value
            self.x = x_select.value
            self.y = y_select.value
            # Axis dropdowns only apply to the scatter; hide them otherwise so
            # the controls always match the active chart.
            axis_row.visible = self.view == "scatter"
            chart_pane.object = self.build_chart()

        view_chips.param.watch(refresh, "value")
        x_select.param.watch(refresh, "value")
        y_select.param.watch(refresh, "value")

        axis_row = pn.Row(x_select, y_select)
        axis_row.visible = self.view == "scatter"

        controls = pn.Row(
            view_chips,
            pn.layout.HSpacer(),
            *self._export_buttons(chart_pane),
            sizing_mode="stretch_width",
        )

        return pn.Column(
            controls,
            axis_row,
            chart_pane,
            sizing_mode="stretch_width",
            styles={"background": self.theme.background, "padding": "12px"},
            stylesheets=[_dashboard_css(self.theme)],
        )

    def _export_buttons(self, chart_pane: pn.pane.Bokeh) -> list[pn.viewable.Viewable]:
        """CSV download (FR3.1) + PNG/SVG export (FR3.2) buttons."""
        name = _slug(self.data.benchmark_name)

        csv_download = pn.widgets.FileDownload(
            label="⬇ Data (CSV)",
            filename=f"{name}.csv",
            callback=lambda: io.StringIO(export_mod.to_csv(self.data)),
            button_type="default",
        )

        # Chart export: prefer vector SVG; if no server-side driver is present
        # fall back to a self-contained interactive HTML file (always works).
        # The Bokeh toolbar's own save button covers client-side one-click SVG.
        def chart_file() -> io.StringIO:
            try:
                return io.StringIO(export_mod.to_svg(chart_pane.object))
            except export_mod.ExportUnavailableError:
                return io.StringIO(
                    export_mod.to_html(
                        chart_pane.object, title=self.data.benchmark_name
                    )
                )

        chart_ext = "svg" if _svg_export_available() else "html"
        chart_download = pn.widgets.FileDownload(
            label="⬇ Chart",
            filename=f"{name}-{self.view}.{chart_ext}",
            callback=chart_file,
            button_type="default",
        )

        buttons: list[pn.viewable.Viewable] = [csv_download, chart_download]

        # Task-level CSV is only offered when there is per-task data (CUJ 2).
        if self.data.task_scores:
            task_download = pn.widgets.FileDownload(
                label="⬇ Task-level (CSV)",
                filename=f"{name}-tasks.csv",
                callback=lambda: io.StringIO(export_mod.task_matrix_csv(self.data)),
                button_type="default",
            )
            buttons.insert(1, task_download)

        return buttons


def _svg_export_available() -> bool:
    """Whether a selenium webdriver is present for server-side SVG export."""
    try:
        import selenium  # noqa: F401
    except Exception:
        return False
    return True


def _slug(text: str) -> str:
    """Filesystem-friendly slug for export filenames."""
    keep = [c.lower() if c.isalnum() else "-" for c in text]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "benchmark"


def _dashboard_css(theme: theme_mod.Theme) -> str:
    """Minimal stylesheet so the chrome matches the chart theme."""
    return f"""
:host {{
  color: {theme.text};
  font-family: {theme_mod.FONT};
}}
.bk-btn {{
  font-family: {theme_mod.FONT};
}}
"""


def dashboard(
    data: LeaderboardData,
    *,
    theme: str | theme_mod.Theme | None = None,
    **kwargs: Any,
) -> BenchmarkDashboard:
    """Convenience factory mirroring the ``charts.*`` builder style."""
    return BenchmarkDashboard(data, theme=theme, **kwargs)
