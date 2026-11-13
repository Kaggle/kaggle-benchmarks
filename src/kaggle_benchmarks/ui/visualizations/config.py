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

"""Chart configuration and deep-link serialization (PRD FR4.1).

The complete view state -- which visualization is active, which metrics are
mapped to the X/Y axes, and whether the Pareto frontier is shown -- is captured
in :class:`ChartConfig`. Serializing it to a URL query string is what makes a
chart shareable: paste the link and the recipient sees the exact chart, and the
same encoded state is what an OpenGraph image renderer would consume to produce
the social-media preview.
"""

from __future__ import annotations

import dataclasses
from urllib.parse import parse_qsl, urlencode

# Registry of visualization types. The key is the short token used in URLs; the
# value is the human label shown on the view chips.
VIEW_TYPES: dict[str, str] = {
    "bars": "Leaderboard",
    "scatter": "Trade-off",
    "heatmap": "Per-task",
    "winrate": "Win rate",
    "elo": "Elo",
    "passk": "pass@k",
}

DEFAULT_VIEW = "bars"


@dataclasses.dataclass
class ChartConfig:
    """Serializable view state for a benchmark chart.

    Attributes:
        view: Visualization type token, one of :data:`VIEW_TYPES`.
        x: Metric key mapped to the X axis (scatter only).
        y: Metric key mapped to the Y axis (scatter/bars).
        show_pareto: Whether to draw the Pareto frontier on the scatter.
        log_x / log_y: Force log scaling on the respective axis.
    """

    view: str = DEFAULT_VIEW
    x: str | None = None
    y: str | None = None
    show_pareto: bool = True
    log_x: bool = False
    log_y: bool = False

    def normalized(self) -> "ChartConfig":
        """Return a copy with an invalid view coerced to the default."""
        view = self.view if self.view in VIEW_TYPES else DEFAULT_VIEW
        return dataclasses.replace(self, view=view)

    def to_query(self) -> str:
        """Encode as a URL query string (stable key order).

        Only non-default fields are emitted to keep links short and readable.
        """
        params: list[tuple[str, str]] = [("view", self.view)]
        if self.x is not None:
            params.append(("x", self.x))
        if self.y is not None:
            params.append(("y", self.y))
        if not self.show_pareto:
            params.append(("pareto", "0"))
        if self.log_x:
            params.append(("logx", "1"))
        if self.log_y:
            params.append(("logy", "1"))
        return urlencode(params)

    def deep_link(self, base_url: str) -> str:
        """Full shareable URL: ``base_url`` with the config query appended."""
        sep = "&" if "?" in base_url else "?"
        return f"{base_url}{sep}{self.to_query()}"

    @classmethod
    def from_query(cls, query: str) -> "ChartConfig":
        """Parse a query string (with or without a leading ``?``) into config.

        Unknown / malformed values fall back to defaults so a hand-edited or
        truncated link never crashes the renderer.
        """
        query = query.lstrip("?")
        params = dict(parse_qsl(query, keep_blank_values=True))

        def flag(name: str, default: bool) -> bool:
            if name not in params:
                return default
            return params[name].strip().lower() in ("1", "true", "yes", "on")

        return cls(
            view=params.get("view", DEFAULT_VIEW),
            x=params.get("x") or None,
            y=params.get("y") or None,
            show_pareto=flag("pareto", True),
            log_x=flag("logx", False),
            log_y=flag("logy", False),
        ).normalized()
