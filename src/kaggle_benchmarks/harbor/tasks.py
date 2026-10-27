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

import contextlib
import dataclasses
import json
import pathlib
import re
import shlex
import sys
from collections.abc import Callable, Iterator
from typing import Any

import cloudpickle

import kaggle_benchmarks
from kaggle_benchmarks import actors, prompting, runs, tasks
from kaggle_benchmarks.harbor import grade
from kaggle_benchmarks.harbor.agents import (
    HARNESSES,
    ONESHOT_RESPONSE_PATH,
    Agent,
    Result,
    run_agent,
)

RESPONSE_PATH = ONESHOT_RESPONSE_PATH

ANSWER_NOTE = (
    "If you can write files, also save your final answer, and nothing else, "
    f"to `{RESPONSE_PATH}`."
)

TYPED_ANSWER_NOTE = (
    "If you can write files, also save your final answer as that JSON object, "
    f"and nothing else, to `{RESPONSE_PATH}`."
)

TASK_TOML = """\
schema_version = "1.3"

[task]
name = "kbench/{slug}"
description = {description}

[environment]
network_mode = "public"
build_timeout_sec = 600.0

[agent]
timeout_sec = {agent_timeout_sec}

[verifier]
timeout_sec = {verifier_timeout_sec}

[verifier.environment]
network_mode = "{verifier_network}"
build_timeout_sec = 900.0
"""

# Kaggle's oneshot player reads its answer path from here, wherever the task goes.
CONFIG_YAML = f"""\
agents:
  - name: {HARNESSES["oneshot"]}
    env:
      KAGGLE_HARBOR_ONESHOT_RESPONSE_PATH: {RESPONSE_PATH}
"""

ENVIRONMENT_DOCKERFILE = """\
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \\
    ca-certificates curl git \\
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
"""

TESTS_DOCKERFILE = """\
FROM python:{python}-slim
RUN pip install --no-cache-dir kaggle-benchmarks=={kbench}
COPY task.json check.pkl grade.py test.sh /tests/
RUN chmod +x /tests/test.sh
WORKDIR /tests
"""

TEST_SH = f"""\
#!/bin/bash
set -euo pipefail
python /tests/grade.py /tests {RESPONSE_PATH} /logs/verifier
"""


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def pickle_by_value(func: Callable) -> bytes:
    """Pickles `func` so the verifier can load it without the author's module."""
    module = sys.modules.get(func.__module__)
    if module is None or module.__name__ == "__main__":
        return cloudpickle.dumps(func)
    cloudpickle.register_pickle_by_value(module)
    try:
        return cloudpickle.dumps(func)
    finally:
        cloudpickle.unregister_pickle_by_value(module)


def as_records(rows) -> list[dict[str, Any]]:
    if hasattr(rows, "to_json"):
        return json.loads(rows.to_json(orient="records"))
    return [dict(row) for row in rows]


@contextlib.contextmanager
def judge(model: str | None) -> Iterator[None]:
    if model is None:
        yield
        return
    before = kaggle_benchmarks.judge_llm
    kaggle_benchmarks.judge_llm = kaggle_benchmarks.kaggle.load_model(model)
    try:
        yield
    finally:
        kaggle_benchmarks.judge_llm = before


@dataclasses.dataclass
class Task:
    """A benchmark task written in Python and exported as Harbor tasks.

    Example:
        task = Task("capital-of-france")

        @task.question
        def question():
            return "What is the capital of France?"

        @task.check
        def check(answer):
            kbench.assertions.assert_in("paris", answer.lower())

        task.run(kbench.llm)     # try it in the notebook
        task.export("out/")      # write the Harbor task folder

    With `rows`, each row becomes one Harbor task. `question` and `check` get
    row values by parameter name, and `check` also gets the player's `answer`,
    parsed into its type annotation if it has one.
    """

    name: str
    description: str = ""
    rows: Any = None
    judge_model: str | None = None
    agent_timeout_sec: float = 600.0
    verifier_timeout_sec: float = 300.0
    question_fn: Callable[..., str] | None = None
    check_fn: Callable[..., object] | None = None
    solution_fn: Callable[..., str] | None = None

    def question(self, func: Callable[..., str]) -> Callable[..., str]:
        self.question_fn = func
        return func

    def check(self, func: Callable[..., object]) -> Callable[..., object]:
        self.check_fn = func
        return func

    def solution(self, func: Callable[..., str]) -> Callable[..., str]:
        self.solution_fn = func
        return func

    @property
    def answer_type(self) -> type:
        return grade.answer_type(self.check_fn)

    def records(self) -> list[dict[str, Any]]:
        return [{}] if self.rows is None else as_records(self.rows)

    def instruction(self, row: dict[str, Any]) -> str:
        parts = [grade.call_with(self.question_fn, **row)]
        if self.answer_type is not str:
            schema_text, _ = next(prompting.process_schema(self.answer_type))
            parts += [schema_text, TYPED_ANSWER_NOTE]
        else:
            parts.append(ANSWER_NOTE)
        return "\n\n".join(parts) + "\n"

    def run(self, llm: actors.LLMChat) -> list[runs.Run]:
        def try_with(llm, row):
            question = grade.call_with(self.question_fn, **row)
            answer = llm.prompt(question, schema=self.answer_type)
            return grade.call_with(self.check_fn, answer=answer, **row)

        results = []
        with judge(self.judge_model):
            for i, row in enumerate(self.records()):
                name = self.name if self.rows is None else f"{self.name} #{i}"
                task = tasks.Task(
                    func=try_with, name=name, store_task=False, store_run=False
                )
                results.append(task.run(llm, row))
        if self.rows is not None:
            passed = sum(run.passed for run in results)
            print(f"{self.name}: {passed} of {len(results)} rows passed")
        return results

    def run_local(
        self,
        agents: list[Agent],
        workdir: str | pathlib.Path,
        runner: str | None = None,
        image: str | None = None,
    ) -> list[Result]:
        """Exports the task and runs it with each agent on Kaggle's Harbor executor.

        Needs a machine with Docker; this does not work inside a Kaggle notebook.
        """
        workdir = pathlib.Path(workdir)
        results = []
        for task_dir in self.export(workdir / "tasks"):
            for agent in agents:
                output_dir = workdir / "runs" / task_dir.name / slugify(agent.label)
                result = run_agent(task_dir, agent, output_dir, runner, image)
                print(
                    f"{task_dir.name} | {agent.label}: "
                    f"reward={result.reward} error={result.error}"
                )
                results.append(result)
        return results

    def export(self, path: str | pathlib.Path) -> list[pathlib.Path]:
        """Writes one Harbor task folder per row under `path` and returns them."""
        if self.question_fn is None or self.check_fn is None:
            raise ValueError("A task needs both a @question and a @check.")

        records = self.records()
        check = pickle_by_value(self.check_fn)
        task_dirs = []
        for i, row in enumerate(records):
            slug = slugify(self.name)
            if self.rows is not None:
                slug = f"{slug}-{i:04d}"
            task_dirs.append(self.export_one(pathlib.Path(path) / slug, row, check))
        return task_dirs

    def export_one(
        self, root: pathlib.Path, row: dict[str, Any], check: bytes
    ) -> pathlib.Path:
        python = f"{sys.version_info.major}.{sys.version_info.minor}"
        spec = {"name": self.name, "row": row, "judge_model": self.judge_model}
        files = {
            "task.toml": TASK_TOML.format(
                slug=root.name,
                description=json.dumps(self.description),
                agent_timeout_sec=self.agent_timeout_sec,
                verifier_timeout_sec=self.verifier_timeout_sec,
                verifier_network="public" if self.judge_model else "no-network",
            ),
            "config.yaml": CONFIG_YAML,
            "instruction.md": self.instruction(row),
            "environment/Dockerfile": ENVIRONMENT_DOCKERFILE,
            "tests/Dockerfile": TESTS_DOCKERFILE.format(
                python=python, kbench=kaggle_benchmarks.__version__
            ),
            "tests/test.sh": TEST_SH,
            "tests/grade.py": pathlib.Path(grade.__file__).read_text(),
            "tests/task.json": json.dumps(spec, indent=2),
        }
        if self.solution_fn is not None:
            answer = grade.call_with(self.solution_fn, **row)
            files["solution/solve.sh"] = (
                "#!/bin/bash\n"
                f"mkdir -p {pathlib.PurePosixPath(RESPONSE_PATH).parent}\n"
                f"printf '%s' {shlex.quote(answer)} > {RESPONSE_PATH}\n"
            )

        for name, text in files.items():
            file = root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(text)
        (root / "tests/check.pkl").write_bytes(check)
        for script in ("tests/test.sh", "solution/solve.sh"):
            if (root / script).exists():
                (root / script).chmod(0o755)
        return root
