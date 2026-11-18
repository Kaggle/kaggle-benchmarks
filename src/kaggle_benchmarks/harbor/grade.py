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

"""Runs an exported check inside a Harbor verifier and writes the reward.

`Task.export` copies this file into the task's `tests/` folder, so it may only
import the standard library and released `kaggle_benchmarks`.
"""

import dataclasses
import inspect
import json
import os
import pathlib
import pickle
import sys


def call_with(func, **available):
    names = inspect.signature(func).parameters
    return func(**{name: value for name, value in available.items() if name in names})


def answer_type(check):
    param = inspect.signature(check).parameters.get("answer")
    if param is None or param.annotation in (inspect.Parameter.empty, str):
        return str
    return restore_missing(param.annotation)


def restore_missing(cls):
    """Undoes cloudpickle copying `dataclasses.MISSING` in by-value dataclasses."""
    for field in getattr(cls, "__dataclass_fields__", {}).values():
        for name in ("default", "default_factory"):
            if type(getattr(field, name)).__name__ == "_MISSING_TYPE":
                setattr(field, name, dataclasses.MISSING)
    return cls


def use_verifier_models(judge_model: str):
    """Points kbench at the model proxy using the names Harbor gives verifiers."""
    if "MODEL_PROXY_URL" not in os.environ and "OPENAI_BASE_URL" in os.environ:
        os.environ["MODEL_PROXY_URL"] = os.environ["OPENAI_BASE_URL"]
        os.environ["MODEL_PROXY_API_KEY"] = os.environ["OPENAI_API_KEY"]
    os.environ.setdefault("LLM_DEFAULT", judge_model)
    os.environ["LLM_DEFAULT_JUDGE"] = judge_model


def grade(check, name: str, response: str, row: dict) -> dict:
    import kaggle_benchmarks as kbench

    def run_check():
        schema = answer_type(check)
        answer = response
        if schema is not str:
            try:
                answer = kbench.prompting.parse_response(
                    kbench.prompting.process_schema(schema), response
                )
            except (kbench.prompting.ResponseParsingError, TypeError, ValueError):
                kbench.assertions.assert_fail(
                    f"The answer should be {schema.__name__} JSON, got {response!r}."
                )
                return None
        return call_with(check, answer=answer, **row)

    task = kbench.tasks.Task(
        func=run_check, name=name, store_task=False, store_run=False
    )
    run = task.run()
    return {
        "reward": 1 if run.passed else 0,
        "answer": response,
        "assertions": [
            {"expectation": a.expectation, "passed": a.passed}
            for a in run.assertion_results
        ],
        "error": run.error_message,
    }


def main(tests_dir: pathlib.Path, response: pathlib.Path, out_dir: pathlib.Path):
    spec = json.loads((tests_dir / "task.json").read_text())
    if spec.get("judge_model"):
        use_verifier_models(spec["judge_model"])
    check = pickle.loads((tests_dir / "check.pkl").read_bytes())
    text = response.read_text() if response.exists() else ""

    report = grade(check, spec["name"], text, spec.get("row", {}))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grading.json").write_text(json.dumps(report, indent=2))
    (out_dir / "reward.txt").write_text(str(report["reward"]))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    tests_dir, response, out_dir = map(pathlib.Path, sys.argv[1:4])
    main(tests_dir, response, out_dir)
