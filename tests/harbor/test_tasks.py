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

import dataclasses
import json
import subprocess
import sys
import tomllib

import pytest

from kaggle_benchmarks import assertions
from kaggle_benchmarks import harbor as kh
from tests.mocks import MockedChat


@pytest.fixture()
def task():
    task = kh.Task("Capital of France")

    @task.question
    def question():
        return "What is the capital of France?"

    @task.check
    def check(answer):
        assertions.assert_in("paris", answer.lower())

    @task.solution
    def solution():
        return "Paris"

    return task


@dataclasses.dataclass
class City:
    name: str


@pytest.fixture()
def capitals():
    task = kh.Task(
        "capitals",
        rows=[
            {"country": "France", "capital": "Paris"},
            {"country": "Japan", "capital": "Tokyo"},
        ],
    )

    @task.question
    def question(country):
        return f"What is the capital of {country}?"

    @task.check
    def check(answer: City, capital):
        assertions.assert_equal(capital, answer.name)

    return task


def grade_exported(task_dir, tmp_path, response_text):
    tests_dir = task_dir / "tests"
    response = tmp_path / "response.txt"
    response.write_text(response_text)
    out_dir = tmp_path / "verifier"
    process = subprocess.run(
        [sys.executable, "-I", tests_dir / "grade.py", tests_dir, response, out_dir],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    return (out_dir / "reward.txt").read_text()


@pytest.mark.parametrize("reply, passed", [("Paris.", True), ("London", False)])
def test_run(task, reply, passed):
    llm = MockedChat.from_contents([reply])

    [run] = task.run(llm)

    assert run.passed is passed


def test_run_rows_with_typed_answer(capitals):
    llm = MockedChat.from_contents(['{"name": "Paris"}', '{"name": "Kyoto"}'])

    assert [run.passed for run in capitals.run(llm)] == [True, False]


def test_export(task, tmp_path):
    [root] = task.export(tmp_path)

    assert root.name == "capital-of-france"
    config = tomllib.loads((root / "task.toml").read_text())
    assert config["task"]["name"] == "kbench/capital-of-france"
    assert config["verifier"]["environment"]["network_mode"] == "no-network"
    assert "What is the capital of France?" in (root / "instruction.md").read_text()
    assert kh.tasks.RESPONSE_PATH in (root / "config.yaml").read_text()
    assert kh.tasks.RESPONSE_PATH in (root / "solution/solve.sh").read_text()


def test_export_rows_keeps_row_values_out_of_the_instruction(capitals, tmp_path):
    roots = capitals.export(tmp_path)

    assert [root.name for root in roots] == ["capitals-0000", "capitals-0001"]
    instruction = (roots[1] / "instruction.md").read_text()
    assert "Japan" in instruction
    assert "Tokyo" not in instruction
    assert "schema" in instruction
    spec = json.loads((roots[1] / "tests/task.json").read_text())
    assert spec["row"] == {"country": "Japan", "capital": "Tokyo"}


@pytest.mark.parametrize("answer, reward", [("Paris", "1"), ("Lyon", "0")])
def test_exported_check_grades_in_a_fresh_process(task, tmp_path, answer, reward):
    [root] = task.export(tmp_path / "out")

    assert grade_exported(root, tmp_path, answer) == reward


@pytest.mark.parametrize(
    "answer, reward",
    [
        ('{"name": "Tokyo"}', "1"),
        ('{"name": "Kyoto"}', "0"),
        ("Tokyo", "0"),
        ("1", "0"),
    ],
)
def test_exported_typed_check_parses_the_answer(capitals, tmp_path, answer, reward):
    root = capitals.export(tmp_path / "out")[1]

    assert grade_exported(root, tmp_path, answer) == reward


def test_agent_env():
    assert kh.Agent("claude-code", "google/gemini-3.6-flash", effort="high").env() == {
        "KAGGLE_AGENT_HARNESS": "claude-code",
        "KAGGLE_AGENT_LLM": "google/gemini-3.6-flash",
        "KAGGLE_AGENT_LLM_REASONING_EFFORT": "high",
    }
    oneshot = kh.Agent("oneshot", "openai/gpt-5.5-responses").env()
    assert oneshot["KAGGLE_AGENT_HARNESS"].endswith(":OneShotAgent")
