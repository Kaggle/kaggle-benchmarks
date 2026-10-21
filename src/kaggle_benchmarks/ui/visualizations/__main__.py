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

"""See the benchmark visualizations, two ways.

Default -- write a self-contained ``index.html`` at the repo root you can just
open in a browser (chips, axis dropdowns, and Pareto toggle all work
client-side, no server). Writing ``index.html`` is what lets the VS Code
"Live Server" / web-preview button pick it up automatically; the Bokeh runtime
is inlined so it renders even with no network access::

    python -m kaggle_benchmarks.ui.visualizations
    python -m kaggle_benchmarks.ui.visualizations -o /tmp/frontier.html

Live server -- run the interactive Panel app over HTTP::

    python -m kaggle_benchmarks.ui.visualizations --serve --port 5006
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from kaggle_benchmarks.ui.visualizations.dashboard import dashboard
from kaggle_benchmarks.ui.visualizations.demo import demo_data
from kaggle_benchmarks.ui.visualizations.site import write_site


def _default_output() -> str:
    """Repo-root ``index.html`` so the Live Server preview serves it by default.

    Walk up from this file to the project root (the dir holding pyproject.toml)
    and target ``index.html`` there. Fall back to the CWD if not found.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return str(parent / "index.html")
    return "index.html"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="kaggle_benchmarks.ui.visualizations",
        description="Generate or serve the benchmark visualization dashboard.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run the live Panel web app instead of writing a static page.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Static page output path (default: <repo-root>/index.html).",
    )
    parser.add_argument(
        "--cdn",
        action="store_true",
        help="Load Bokeh from CDN (smaller file, needs internet) instead of "
        "inlining the runtime.",
    )
    parser.add_argument(
        "--port", type=int, default=5006, help="Port for --serve (default 5006)."
    )
    parser.add_argument(
        "--address",
        default="localhost",
        help="Bind address for --serve (use 0.0.0.0 to expose externally).",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open a browser automatically.",
    )
    args = parser.parse_args(argv)

    data = demo_data()

    if args.serve:
        app = dashboard(data)
        print(f"Serving benchmark dashboard on http://{args.address}:{args.port}")
        app.serve(port=args.port, address=args.address, show=not args.no_open)
        return

    output = args.output or _default_output()
    path = Path(write_site(data, output, inline=not args.cdn)).resolve()
    print(f"Wrote interactive benchmark page to {path}")
    print("Open it in the VS Code web preview (Live Server), or in a browser:")
    print(f"  file://{path}")
    if not args.no_open:
        try:
            webbrowser.open(f"file://{path}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
