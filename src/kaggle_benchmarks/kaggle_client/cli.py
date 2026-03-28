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
"""CLI entry points for kaggle-benchmarks."""

import argparse
import json
import sys
from pathlib import Path


def _get_client(base_dir: str):
    from kaggle_benchmarks.kaggle_client.notebook_api import BenchmarkNotebookClient
    return BenchmarkNotebookClient(base_dir=base_dir)


def run():
    """Entry point for the kaggle-bench CLI."""
    parser = argparse.ArgumentParser(
        prog="kaggle-bench",
        description="Manage Kaggle benchmark notebooks from the command line.",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory for benchmark workspaces (default: current directory).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run subcommand ---
    run_parser = subparsers.add_parser(
        "run",
        help="Publish and run a benchmark script on Kaggle.",
    )
    run_parser.add_argument(
        "notebook_slug",
        help="Short notebook name (e.g. 'my-benchmark').",
    )
    run_parser.add_argument(
        "--source-file",
        default=None,
        help="Path to a local .py file to use as the benchmark.",
    )
    run_parser.add_argument(
        "--dataset",
        action="append",
        dest="dataset_sources",
        metavar="OWNER/DATASET",
        help="Kaggle dataset slug to mount (can be repeated).",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Push even if a run is already in progress.",
    )
    run_parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for the run to finish and print results.",
    )
    run_parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between status checks when --wait is used (default: 60).",
    )
    run_parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Max seconds to wait when --wait is used. Default: no timeout.",
    )

    # --- fork subcommand ---
    fork_parser = subparsers.add_parser(
        "fork",
        help="Pull an existing Kaggle benchmark notebook for local editing.",
    )
    fork_parser.add_argument(
        "source_notebook_id",
        help="Full Kaggle notebook path (e.g. 'alice/riddle-benchmark').",
    )
    fork_parser.add_argument(
        "--dest",
        default=None,
        dest="dest_notebook_slug",
        help="Local workspace directory name. Defaults to the notebook slug.",
    )
    fork_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing local workspace.",
    )

    args = parser.parse_args()
    client = _get_client(args.base_dir)

    if args.command == "run":
        source = Path(args.source_file) if args.source_file else None
        print(f"Publishing '{args.notebook_slug}' to Kaggle...")
        tracking_url = client.publish_and_run(
            notebook_slug=args.notebook_slug,
            source_file=source,
            dataset_sources=args.dataset_sources,
            force=args.force,
        )
        print(f"Submitted. Tracking: {tracking_url}")
        if args.wait:
            print("Waiting for results (Ctrl+C to cancel)...")
            result = client.get_results(
                notebook_slug=args.notebook_slug,
                poll_interval=args.poll_interval,
                timeout=args.timeout,
                on_status=lambda s: print(f"  Status: {s}"),
            )
            print(f"\nFinal status: {result.status}")
            if result.tracking_url:
                print(f"Notebook: {result.tracking_url}")
            if result.output_dir:
                print(f"Output saved to: {result.output_dir}")
                for filename, data in result.iter_run_results():
                    print(f"\n--- {filename} ---")
                    print(json.dumps(data, indent=2))
            if result.error:
                print(f"Error: {result.error}", file=sys.stderr)
                sys.exit(1)

    elif args.command == "fork":
        print(f"Forking '{args.source_notebook_id}'...")
        workspace = client.fork(
            source_notebook_id=args.source_notebook_id,
            dest_notebook_slug=args.dest_notebook_slug,
            overwrite=args.overwrite,
        )
        print(f"Workspace ready: {workspace}")
