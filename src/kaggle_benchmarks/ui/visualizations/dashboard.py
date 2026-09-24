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

"""Interactive Panel dashboard tying the chart library together (FR1.2).

Surfaces, per the PRD "prominent and discoverable placement" principle:

* fast-access **view chips** for each visualization type (FR1.2),
* independent **X / Y axis dropdowns** mapping any scalar metric (FR1.3),
* a **Pareto** toggle (Pareto prominence),
* a **Download CSV** button for the full, task-level dataset (FR3.1),
* **image export** buttons (SVG / PNG / interactive HTML) (FR3.2),
* a **deep-link** field serializing the exact view state to a URL (FR4.1).

The whole thing is driven by a single :class:`ChartConfig`, so any control
change re-renders through the same code path used for exports and deep links.
"""

from __future__ import annotations

import io

import panel as pn

from kaggle_benchmarks.ui.visualizations import charts, export, theme
from kaggle_benchmarks.ui.visualizations.config import VIEW_TYPES, ChartConfig
from kaggle_benchmarks.ui.visualizations.data import LeaderboardData


class BenchmarkDashboard:
    """Stateful controller wrapping a :class:`LeaderboardData` in a Panel view.

    Use :func:`dashboard` for the common case; instantiate directly when you
    need to drive the config programmatically (e.g. tests or embedding).
    """

    def __init__(
        self,
        data: LeaderboardData,
        config: ChartConfig | None = None,
        *,
        base_url: str = "https://www.kaggle.com/benchmarks",
    ):
        self.data = data
        self.base_url = base_url
        self.config = (config or self._initial_config()).normalized()

        self._chart_pane = pn.pane.Bokeh(sizing_mode="stretch_width")
        self._deep_link = pn.widgets.TextInput(
            name="Deep link (share this URL)", sizing_mode="stretch_width"
        )
        self._build_controls()
        self._sync_axis_visibility()
        self._render()

    # --- config helpers ----------------------------------------------------

    def _initial_config(self) -> ChartConfig:
        x, y = self.data.default_axes()
        # Land on the trade-off scatter when metrics allow it -- that's the
        # "immediately visible multi-dimensional" default the PRD asks for;
        # otherwise fall back to the bar leaderboard.
        view = "scatter" if len(self.data.scalar_metric_keys) >= 2 else "bars"
        return ChartConfig(view=view, x=x, y=y)

    def _available_views(self) -> list[str]:
        """Only offer view chips the data can actually populate."""
        views = ["bars"]
        if len(self.data.scalar_metric_keys) >= 2:
            views.append("scatter")
        if self.data.has_task_matrix:
            views.append("heatmap")
        if self.data.has_pairwise:
            views.append("winrate")
        if self.data.has_elo:
            views.append("elo")
        if self.data.has_pass_at_k:
            views.append("passk")
        return views

    # --- control wiring ----------------------------------------------------

    def _build_controls(self) -> None:
        views = self._available_views()
        self._view_chips = pn.widgets.RadioButtonGroup(
            name="View",
            options={VIEW_TYPES[v]: v for v in views},
            value=self.config.view if self.config.view in views else views[0],
            button_type="primary",
        )
        self._view_chips.param.watch(self._on_view, "value")

        metric_options = {
            self.data.metric(k).label: k for k in self.data.scalar_metric_keys
        }
        self._x_select = pn.widgets.Select(
            name="X axis", options=metric_options, value=self.config.x
        )
        self._y_select = pn.widgets.Select(
            name="Y axis", options=metric_options, value=self.config.y
        )
        self._x_select.param.watch(self._on_axis, "value")
        self._y_select.param.watch(self._on_axis, "value")

        self._pareto_toggle = pn.widgets.Checkbox(
            name="Show Pareto frontier", value=self.config.show_pareto
        )
        self._pareto_toggle.param.watch(self._on_axis, "value")

        # FR3.1: full dataset (including task-level breakdown) as CSV.
        self._download_csv = pn.widgets.FileDownload(
            label="⬇ Download data (CSV)",
            filename=f"{_slug(self.data.name)}.csv",
            callback=lambda: io.StringIO(self.data.to_csv()),
            button_type="success",
        )
        # FR3.2: interactive HTML export always works; PNG/SVG when a browser
        # driver is present.
        self._download_html = pn.widgets.FileDownload(
            label="⬇ Export chart (HTML)",
            filename=f"{_slug(self.data.name)}-chart.html",
            callback=self._html_callback,
        )
        # PNG/SVG export needs a headless browser. Offer the buttons only when
        # one is available, and always leave the interactive-HTML export and the
        # chart toolbar's in-browser "save" as guaranteed fallbacks.
        image_export_ok = export.webdriver_available()
        self._png_download = pn.widgets.FileDownload(
            label="⬇ PNG",
            filename=f"{_slug(self.data.name)}-chart.png",
            callback=lambda: io.BytesIO(
                export.to_png(charts.build_chart(self.data, self.config))
            ),
            width=110,
            visible=image_export_ok,
        )
        self._svg_download = pn.widgets.FileDownload(
            label="⬇ SVG",
            filename=f"{_slug(self.data.name)}-chart.svg",
            callback=lambda: io.StringIO(
                export.to_svg(charts.build_chart(self.data, self.config))
            ),
            width=110,
            visible=image_export_ok,
        )
        self._export_status = pn.pane.Markdown(
            ""
            if image_export_ok
            else "*Tip: use the chart toolbar's save button for a quick PNG, "
            "or the HTML export above. Install a headless browser for "
            "1-click PNG/SVG.*",
            styles={"font-size": "0.8em", "color": "gray"},
        )

    # --- callbacks ---------------------------------------------------------

    def _on_view(self, event) -> None:
        self.config.view = event.new
        self._sync_axis_visibility()
        self._render()

    def _on_axis(self, event) -> None:
        self.config.x = self._x_select.value
        self.config.y = self._y_select.value
        self.config.show_pareto = self._pareto_toggle.value
        self._render()

    def _sync_axis_visibility(self) -> None:
        """Only the scatter uses both axes + Pareto; bars uses Y as the metric."""
        is_scatter = self.config.view == "scatter"
        is_bars = self.config.view == "bars"
        self._x_select.visible = is_scatter
        self._pareto_toggle.visible = is_scatter
        self._y_select.visible = is_scatter or is_bars
        self._y_select.name = "Metric" if is_bars else "Y axis"

    def _render(self) -> None:
        self._chart_pane.object = charts.build_chart(self.data, self.config)
        self._deep_link.value = self.config.deep_link(self.base_url)

    def _html_callback(self) -> io.BytesIO:
        fig = charts.build_chart(self.data, self.config)
        return io.BytesIO(export.to_html(fig).encode("utf-8"))

    # --- standalone app ----------------------------------------------------

    def template(self) -> pn.template.BootstrapTemplate:
        """Wrap the dashboard in a full-page, Kaggle-branded app template.

        This is what turns the notebook component into a standalone web page:
        a titled header bar plus the dashboard in the main area, ready to be
        marked ``.servable()`` and served over HTTP by ``panel serve``.
        """
        template = pn.template.BootstrapTemplate(
            title="Kaggle Benchmarks",
            header_background=theme.KAGGLE_BLUE,
            theme="dark" if theme.get_palette().name == "dark" else "default",
        )
        template.main.append(self.__panel__())
        return template

    def servable(self) -> pn.viewable.Viewable:
        """Mark the app servable so ``panel serve`` picks it up.

        Usage::

            panel serve app.py            # where app.py calls dashboard(...).servable()
        """
        return self.template().servable()

    def serve(self, *, port: int = 5006, show: bool = False, **kwargs):
        """Launch a standalone web server for this dashboard and block.

        This is the "open the app" entry point when you are NOT in a notebook::

            from kaggle_benchmarks.ui import visualizations as viz
            viz.dashboard(viz.demo_data()).serve()   # -> http://localhost:5006

        Args:
            port: TCP port to bind.
            show: Open a browser tab automatically.
            **kwargs: Forwarded to :func:`panel.serve`.
        """
        return pn.serve(self.template(), port=port, show=show, **kwargs)

    # --- layout ------------------------------------------------------------

    def __panel__(self) -> pn.viewable.Viewable:
        controls = pn.Row(
            self._x_select,
            self._y_select,
            self._pareto_toggle,
            sizing_mode="stretch_width",
        )
        exports = pn.Row(
            self._download_csv,
            self._download_html,
            self._png_download,
            self._svg_download,
            sizing_mode="stretch_width",
        )
        return pn.Column(
            pn.pane.Markdown(f"## {self.data.name}"),
            self._view_chips,
            controls,
            self._chart_pane,
            pn.layout.Divider(),
            exports,
            self._export_status,
            self._deep_link,
            sizing_mode="stretch_width",
        )

    def _repr_mimebundle_(self, include=None, exclude=None):
        return self.__panel__()._repr_mimebundle_(include, exclude)


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")


def dashboard(
    data: LeaderboardData,
    config: ChartConfig | None = None,
    **kwargs,
) -> BenchmarkDashboard:
    """Create a :class:`BenchmarkDashboard` for ``data`` (convenience wrapper).

    In a notebook, the returned object renders itself; use ``.servable()`` on
    ``dashboard(...).__panel__()`` to serve it as a standalone app.
    """
    return BenchmarkDashboard(data, config, **kwargs)
