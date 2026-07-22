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

"""Launch the benchmark visualization dashboard as a standalone web app.

    python -m kaggle_benchmarks.ui.visualizations            # demo data
    python -m kaggle_benchmarks.ui.visualizations --port 8080

Then open the printed http://localhost:<port> URL in a browser. This is the
"open the app" entry point for anyone not working inside a notebook.
"""

from __future__ import annotations

import argparse

from kaggle_benchmarks.ui.visualizations.dashboard import dashboard
from kaggle_benchmarks.ui.visualizations.demo import demo_data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="kaggle_benchmarks.ui.visualizations",
        description="Serve the benchmark visualization dashboard.",
    )
    parser.add_argument(
        "--port", type=int, default=5006, help="Port to serve on (default 5006)."
    )
    parser.add_argument(
        "--address",
        default="localhost",
        help="Address to bind (use 0.0.0.0 to expose externally).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not try to open a browser tab automatically.",
    )
    args = parser.parse_args(argv)

    app = dashboard(demo_data())
    print(f"Serving benchmark dashboard on http://{args.address}:{args.port}")
    app.serve(port=args.port, address=args.address, show=not args.no_show)


if __name__ == "__main__":
    main()
