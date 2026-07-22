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

Default -- write a self-contained web page you can just open in a browser
(chips, axis dropdowns, and Pareto toggle all work client-side, no server)::

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
        default="benchmark.html",
        help="Static page output path (default: benchmark.html).",
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

    path = Path(write_site(data, args.output)).resolve()
    print(f"Wrote interactive benchmark page to {path}")
    print(f"Open it in a browser:  file://{path}")
    if not args.no_open:
        try:
            webbrowser.open(f"file://{path}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
