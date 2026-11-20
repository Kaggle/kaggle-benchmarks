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

# %% [markdown]
# # Private split: a manual end-to-end check
#
# The scenarios of the `private_split` golden tests in
# `test_cookbook_examples.py`, without pytest, against a real model. Each
# private row carries PRIVATE_TAG, which must reach `private/*.private.json`
# and nothing else: no other output file, no console output, no log record
# and no warning. Prints the scores, the results of the published run file,
# the files written, and a PASS/FAIL line per check.
#
# From the CLI (exits 1 if a check fails):
#
#     uv run python golden_tests/private_split_notebook.py --api openai
#
# As a notebook: open it in VS Code or Jupyter, or convert it with
# `jupytext --to ipynb golden_tests/private_split_notebook.py`.
#
# Needs MODEL_PROXY_URL and MODEL_PROXY_API_KEY, in the environment or in the
# `.env` that `kaggle b auth -y` writes. The model and API come from
# `--model`/`--api` or PRIVATE_SPLIT_MODEL/PRIVATE_SPLIT_API.

# %%
import argparse
import functools
import io
import json
import logging
import os
import sys
import tempfile
import traceback
import warnings
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import dotenv
import numpy as np
import pandas as pd

IN_NOTEBOOK = "ipykernel" in sys.modules
CASES = ("smoke", "rich", "dict", "stop")

parser = argparse.ArgumentParser(description="Checks a private split end to end.")
parser.add_argument(
    "--model",
    default=os.environ.get("PRIVATE_SPLIT_MODEL", "google/gemini-3-flash-preview"),
    help="Model Proxy slug (env PRIVATE_SPLIT_MODEL).",
)
parser.add_argument(
    "--api",
    default=os.environ.get("PRIVATE_SPLIT_API", "genai"),
    choices=["genai", "openai"],
    help="Model Proxy API (env PRIVATE_SPLIT_API).",
)
parser.add_argument(
    "--cases",
    default=os.environ.get("PRIVATE_SPLIT_CASES", ",".join(CASES)),
    help=f"Comma-separated cases to run, of {', '.join(CASES)} "
    "(env PRIVATE_SPLIT_CASES).",
)
parser.add_argument(
    "--output-dir",
    default=os.environ.get("PRIVATE_SPLIT_OUTPUT_DIR"),
    help="Where to write the run files; a new temp dir by default.",
)
# A notebook kernel passes arguments of its own.
args, _ = parser.parse_known_args()
cases = [case.strip() for case in args.cases.split(",") if case.strip()]
if unknown := set(cases) - set(CASES):
    raise SystemExit(f"Unknown cases {sorted(unknown)}; choose from {CASES}.")

# %%
# Checked before importing kaggle_benchmarks, which loads the models on import.
dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
if missing := [
    name
    for name in ("MODEL_PROXY_URL", "MODEL_PROXY_API_KEY")
    if not os.environ.get(name)
]:
    raise SystemExit(
        f"{' and '.join(missing)} not set. Export them, or run `kaggle b auth -y` "
        "to write a .env file, then run this again."
    )
# Read on import once Model Proxy is configured.
os.environ.setdefault("LLM_DEFAULT", args.model)

import kaggle_benchmarks as kbench  # noqa: E402
from kaggle_benchmarks import events  # noqa: E402
from kaggle_benchmarks.kaggle import serialization  # noqa: E402
from kaggle_benchmarks.tasks import NonRecoverableError  # noqa: E402

kbench.config.continue_with_exceptions = False
llm = kbench.kaggle.load_model(args.model, api=args.api)
output_root = Path(args.output_dir or tempfile.mkdtemp(prefix="private_split_"))
print(f"Model: {args.model} ({args.api}). Output: {output_root}")

# %%
# Running a split task and checking what it left behind.

PRIVATE_TAG = "ref-7f3a91"
PUBLIC_TAG = "pub-2c8e41"


@dataclass
class Outcome:
    name: str
    run: Any
    error: Exception | None
    directory: Path
    printed: str
    logs: list[str]
    warnings: list[str]
    checks: list[tuple[str, str, str]] = field(default_factory=list)

    def published_files(self) -> list[Path]:
        return [
            p
            for p in sorted(self.directory.rglob("*"))
            if p.is_file() and "private" not in p.relative_to(self.directory).parts
        ]

    def private_files(self) -> list[Path]:
        return sorted(self.directory.glob("private/*"))

    def private_file(self) -> dict:
        (path,) = self.directory.glob("private/*.private.json")
        return json.loads(path.read_text())

    def run_file(self) -> dict | None:
        if self.run is None:
            return None
        name = serialization.generate_run_filename(
            self.run.task.name, self.run.cache_id
        )
        path = self.directory / name
        return json.loads(path.read_text()) if path.exists() else None

    def check(self, label: str, fn: Callable[[], bool], known_issue: bool = False):
        """Records PASS/FAIL, or XFAIL/XPASS for a known issue pending a fix."""
        try:
            ok, detail = bool(fn()), ""
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        if known_issue:
            status = "XPASS" if ok else "XFAIL"
        else:
            status = "PASS" if ok else "FAIL"
        self.checks.append((status, label, detail))


class _RecordText(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.texts: list[str] = []

    def emit(self, record):
        self.texts.append(self.format(record))


def run_split(name: str, task, write_atif: bool = False) -> Outcome:
    """Runs `task` on the model into its own directory, recording the console
    output, every log record at DEBUG and above, and every warning."""
    directory = output_root / name
    directory.mkdir(parents=True, exist_ok=True)
    printed = io.StringIO()
    console = kbench.ui.console.ConsoleUI(output=printed)
    handler = _RecordText()
    root = logging.getLogger()
    run, error = None, None
    with ExitStack() as stack:
        # Each change is undone on exit, in reverse order.
        stack.callback(setattr, kbench, "client", kbench.client)
        kbench.client = kbench.kaggle.KaggleClient(directory)
        stack.callback(setattr, kbench.config, "write_atif", kbench.config.write_atif)
        kbench.config.write_atif = write_atif
        stack.callback(root.setLevel, root.level)
        root.setLevel(logging.DEBUG)
        root.addHandler(handler)
        stack.callback(root.removeHandler, handler)
        events.manager.bind(console)
        stack.callback(events.manager.unbind, console)
        caught = stack.enter_context(warnings.catch_warnings(record=True))
        warnings.simplefilter("always")
        try:
            run = task.run(llm)
        except Exception as e:
            error = e
    return Outcome(
        name=name,
        run=run,
        error=error,
        directory=directory,
        printed=printed.getvalue(),
        logs=handler.texts,
        warnings=[str(w.message) for w in caught],
    )


def check_leaks(outcome: Outcome):
    outcome.check(
        "tag absent from console output", lambda: PRIVATE_TAG not in outcome.printed
    )
    outcome.check(
        f"tag absent from {len(outcome.logs)} log records",
        lambda: not [t for t in outcome.logs if PRIVATE_TAG in t],
    )
    outcome.check(
        f"tag absent from {len(outcome.warnings)} warnings",
        lambda: not [t for t in outcome.warnings if PRIVATE_TAG in t],
    )
    published = outcome.published_files()
    leaked = [p.name for p in published if PRIVATE_TAG in p.read_text()]
    outcome.check(
        f"tag absent from {len(published)} published files"
        + (f" (found in {leaked})" if leaked else ""),
        lambda: published and not leaked,
    )
    outcome.check(
        "tag present in private/*.private.json",
        lambda: PRIVATE_TAG in json.dumps(outcome.private_file()),
    )


def check_results(outcome: Outcome) -> list[dict]:
    """Checks the published run file holds AGGREGATED, PUBLIC, PRIVATE, and
    returns those results."""
    results = (outcome.run_file() or {}).get("results", [])
    outcome.check(
        "results are AGGREGATED, PUBLIC, PRIVATE",
        lambda: [r["type"] for r in results] == ["AGGREGATED", "PUBLIC", "PRIVATE"],
    )
    return results


# %%
# The scenarios: the same tasks as the golden tests.

public_df = pd.DataFrame(
    [
        {"question": "What's the capital of Singapore", "answer": "Singapore"},
        {"question": "What's the capital of France", "answer": "Paris"},
    ]
)
private_df = pd.DataFrame(
    [
        {"question": f"What's the capital of Japan ({PRIVATE_TAG})", "answer": "Tokyo"},
        {"question": f"What's the capital of Italy ({PRIVATE_TAG})", "answer": "Rome"},
    ]
)


# Smoke: one private row fails while quoting its question.
@kbench.task(name="split_qa_task", store_task=False)
def split_qa_task(llm, question, answer) -> bool:
    response = llm.prompt(question)
    if answer == "Rome":
        raise ValueError(f"Intentional failure for sample: {question}")
    return answer.lower() in response.lower()


@kbench.task(name="private_split_eval")
def private_split_eval(llm) -> kbench.SplitScores:
    return split_qa_task.evaluate_splits(
        public=public_df, private=private_df, llm=[llm], on_failure="continue"
    )


# Rich: threads, a judge, a tool, a bare assert and a numpy (score, confidence)
# pair. Only the capital and assert rows are scored, so the scores don't depend
# on the judge or on the model calling the tool.
CAPITALS = {
    "australia": "Canberra",
    "canada": "Ottawa",
    "italy": "Rome",
    "japan": "Tokyo",
}
SCORED_KINDS = {"capital", "assert"}


def lookup_capital(country: str) -> str:
    """Returns the capital city of a country, given its English name."""
    return CAPITALS.get(country.strip().lower(), "unknown")


def _rows(tag: str, rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"kind": k, "question": f"{q} ({tag})", "answer": a} for k, q, a in rows]
    )


rich_public_df = _rows(
    PUBLIC_TAG,
    [
        ("capital", "What's the capital of France", "Paris"),
        ("capital", "What's the capital of Germany", "Berlin"),
        ("judge", "In one sentence, what's the capital of Canada", "Ottawa"),
        (
            "tool",
            "Use the lookup_capital tool to find the capital of Australia",
            "Canberra",
        ),
    ],
)
rich_private_df = _rows(
    PRIVATE_TAG,
    [
        ("capital", "What's the capital of Japan", "Tokyo"),
        ("capital", "What's the capital of Italy", "Rome"),
        ("assert", "What's the capital of Spain", "Madrid"),
        ("judge", "In one sentence, what's the capital of Japan", "Tokyo"),
        ("tool", "Use the lookup_capital tool to find the capital of Italy", "Rome"),
    ],
)


@kbench.task(name="rich_split_row", store_task=False)
def rich_split_row(llm, kind, question, answer) -> bool:
    if kind == "tool":
        response = llm.prompt(question, tools=[lookup_capital])
    else:
        response = llm.prompt(question)
    if kind == "judge":
        assessment = kbench.assertions.assess_response_with_judge(
            response_text=response,
            judge_llm=llm,
            criteria=[f"The response answers '{question}' with {answer}."],
        )
        kbench.assertions.assert_true(
            assessment is not None, expectation="The judge should assess."
        )
    if kind == "assert":
        assert False, f"Intentional bare assert for: {question}"
    return answer.lower() in response.lower()


def wilson(hits: Iterable[bool]) -> tuple[np.float32, np.float32]:
    """The accuracy and the half-width of its 95% Wilson interval."""
    values = np.array(list(hits), dtype=np.float32)
    n, p, z = values.size, values.mean(), np.float32(1.96)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return p, half


def wilson_accuracy(runs) -> tuple[np.float32, np.float32]:
    return wilson(
        run.result is True for run in runs if run.params["kind"] in SCORED_KINDS
    )


@kbench.task(name="private_split_rich")
def private_split_rich(llm) -> kbench.SplitScores:
    return rich_split_row.evaluate_splits(
        public=rich_public_df,
        private=rich_private_df,
        llm=[llm],
        n_jobs=2,
        on_failure="continue",
        score=wilson_accuracy,
    )


# Dict: score= returns a dict with fixed keys, which the platform makes into
# public leaf tasks.
def accuracy_and_count(runs) -> dict:
    hits = [run.result is True for run in runs]
    return {"accuracy": sum(hits) / len(hits), "n": len(hits)}


@kbench.task(name="split_plain_row", store_task=False)
def split_plain_row(llm, question, answer) -> bool:
    return answer.lower() in llm.prompt(question).lower()


@kbench.task(name="private_split_dict")
def private_split_dict(llm) -> kbench.SplitScores:
    return split_plain_row.evaluate_splits(
        public=public_df, private=private_df, llm=[llm], score=accuracy_and_count
    )


# Stop: a private row raises NonRecoverableError, which stops the task.
@kbench.task(name="split_stop_row", store_task=False)
def split_stop_row(llm, question, answer) -> bool:
    response = llm.prompt(question)
    if answer == "Rome":
        raise NonRecoverableError(f"Intentional stop at: {question}")
    return answer.lower() in response.lower()


@kbench.task(name="private_split_stop")
def private_split_stop(llm) -> kbench.SplitScores:
    # Not n_jobs=1: in notebook mode joblib 1.6 then replaces an error raised
    # after the first row with an AttributeError of its own, on any split.
    return split_stop_row.evaluate_splits(
        public=public_df, private=private_df, llm=[llm], n_jobs=2, on_failure="continue"
    )


# %%
# Each case: run it, then check it.


def smoke() -> Outcome:
    outcome = run_split("smoke", private_split_eval)
    outcome.check("no error", lambda: outcome.error is None)
    outcome.check(
        "scores are 0.75 / 1.0 / 0.5",
        lambda: outcome.run.result
        == kbench.SplitScores(overall=0.75, public=1.0, private=0.5),
    )
    outcome.check(
        "public rows are shown", lambda: "capital of France" in outcome.printed
    )
    check_results(outcome)
    check_leaks(outcome)
    return outcome


def rich() -> Outcome:
    outcome = run_split("rich", private_split_rich, write_atif=True)
    outcome.check("no error", lambda: outcome.error is None)
    outcome.check(
        "scores of the capital and assert rows",
        lambda: outcome.run.result
        == kbench.SplitScores(
            overall=wilson([True, True, True, True, False]),
            public=wilson([True, True]),
            private=wilson([True, True, False]),
        ),
    )
    outcome.check(
        "public rows of every kind are shown",
        lambda: outcome.printed.count(PUBLIC_TAG) >= len(rich_public_df),
    )
    results = check_results(outcome)
    outcome.check(
        "each result is a numericResult with confidenceInterval, as run.result",
        lambda: [r.get("numericResult") for r in results]
        == [
            {"value": value, "confidenceInterval": confidence}
            for value, confidence in (
                outcome.run.result.overall,
                outcome.run.result.public,
                outcome.run.result.private,
            )
        ],
    )
    check_leaks(outcome)

    @functools.cache
    def private_rows() -> dict[str, dict]:
        rows = {}
        for row in outcome.private_file()["subruns"]:
            (question,) = [q for q in rich_private_df.question if q in json.dumps(row)]
            rows[question] = row
        return rows

    by_kind = dict(zip(rich_private_df.kind, rich_private_df.question))
    outcome.check(
        "every private row is in the private file",
        lambda: set(private_rows()) == set(rich_private_df.question),
    )
    outcome.check(
        "the bare assert is a failed assertion of a completed row",
        lambda: (row := private_rows()[by_kind["assert"]])["state"]
        == "BENCHMARK_TASK_RUN_STATE_COMPLETED"
        and "AssertionError" in row["assertions"][0]["expectation"],
    )
    outcome.check(
        "the judge's side chat is in the private file",
        lambda: "impartial and strict technical evaluator"
        in json.dumps(private_rows()[by_kind["judge"]]),
    )

    def root_metadata() -> bool:
        private, published = outcome.private_file(), outcome.run_file()
        slug = private.get("modelVersion", {}).get("slug")
        return bool(
            slug
            and slug == published["modelVersion"]["slug"]
            and private.get("startTime")
            and private.get("endTime")
        )

    outcome.check(
        "private file has the root model slug, start and end time",
        root_metadata,
        known_issue=True,
    )
    outcome.check(
        "WRITE_ATIF=True writes private/<stem>.atif.json",
        lambda: len(list(outcome.directory.glob("private/*.atif.json"))) == 1,
        known_issue=True,
    )
    return outcome


def dict_case() -> Outcome:
    outcome = run_split("dict", private_split_dict)
    outcome.check("no error", lambda: outcome.error is None)
    outcome.check(
        "scores are accuracy 1.0 over 4 / 2 / 2 rows",
        lambda: outcome.run.result
        == kbench.SplitScores(
            overall={"accuracy": 1.0, "n": 4},
            public={"accuracy": 1.0, "n": 2},
            private={"accuracy": 1.0, "n": 2},
        ),
    )
    results = check_results(outcome)
    outcome.check(
        "each result is a dictResult with the keys accuracy, n",
        lambda: all(set(r["dictResult"]) == {"accuracy", "n"} for r in results),
    )
    outcome.check(
        "tag in no result key or value", lambda: PRIVATE_TAG not in json.dumps(results)
    )
    check_leaks(outcome)
    return outcome


def stop() -> Outcome:
    outcome = run_split("stop", private_split_stop)
    outcome.check(
        "the task stops with a withheld NonRecoverableError",
        lambda: isinstance(outcome.error, NonRecoverableError)
        and "withheld" in str(outcome.error),
    )
    outcome.check(
        "tag absent from the error's traceback",
        lambda: PRIVATE_TAG not in "".join(traceback.format_exception(outcome.error)),
    )
    check_leaks(outcome)
    outcome.check(
        "the private file keeps the original error",
        lambda: "Intentional stop at"
        in (error := outcome.private_file()["errorMessage"])
        and PRIVATE_TAG in error,
    )
    outcome.check(
        "private file is ERRORED",
        lambda: outcome.private_file()["state"] == "BENCHMARK_TASK_RUN_STATE_ERRORED",
        known_issue=True,
    )
    return outcome


RUNNERS = {"smoke": smoke, "rich": rich, "dict": dict_case, "stop": stop}

# %%
outcomes = [RUNNERS[case]() for case in cases]

# %%
for outcome in outcomes:
    print(f"\n=== {outcome.name} ===")
    if outcome.error is not None:
        print(f"Error: {type(outcome.error).__name__}: {outcome.error}")
    if outcome.run is not None:
        print(f"Scores: {outcome.run.result}")
    run_file = outcome.run_file()
    if run_file is not None:
        print("Published results:")
        for result in run_file.get("results", []):
            print(f"  {json.dumps(result)}")
    print("Wrote:")
    for path in outcome.published_files() + outcome.private_files():
        print(f"  {path}")
    print("Checks:")
    for status, label, detail in outcome.checks:
        print(f"  {status:5}  {label}" + (f"  [{detail}]" if detail else ""))

# %%
failed = [
    (outcome.name, label)
    for outcome in outcomes
    for status, label, _ in outcome.checks
    if status == "FAIL"
]
print(f"\n{'FAIL' if failed else 'PASS'}: {len(failed)} failed checks in {cases}.")
if failed and not IN_NOTEBOOK:
    sys.exit(1)
