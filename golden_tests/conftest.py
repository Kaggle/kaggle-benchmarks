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

from pathlib import Path

import pytest
import yaml

import kaggle_benchmarks as kbench
from kaggle_benchmarks import ExecutionMode, clients, config

# Run as tests, not as a benchmark: the default client serializes every task and
# run to JSON in the working directory, which litters the tree (and the CI
# workspace) with hundreds of files. This has to happen at import time — conftest
# is imported before the test modules, and @kbench.task registers, and would
# store, each task as soon as its module is imported.
config.execution_mode = ExecutionMode.TESTING
kbench.client = clients.InMemoryClient()

module_reports = {}


def pytest_addoption(parser):
    """Adds the --generate-report command-line option."""
    parser.addoption(
        "--generate-report",
        action="store_true",
        default=False,
        help="Generate a YAML report for each test module.",
    )


@pytest.fixture(scope="module", autouse=True)
def module_report_fixture(request):
    """A module-scoped fixture to collect test results and generate a YAML report.

    The report is only generated if the --generate-report flag is provided.
    """
    if not request.config.getoption("--generate-report"):
        yield
        return

    module_name = request.module.__name__
    module_reports[module_name] = []

    yield

    # Teardown: Write report if the flag was used.
    report_data = module_reports.get(module_name, [])

    if report_data:
        report = {}
        for llm, api, test_result in report_data:
            model_report = report.setdefault(
                f"{api}://{llm.name}",
                {
                    "config": {"structured_output": llm.support_structured_outputs},
                    "tests": {},
                },
            )
            model_report["tests"][test_result.location[-1].split("[")[0]] = (
                test_result.outcome
            )

        base_name = Path(request.module.__file__).stem
        report_filename = f"{base_name}_report.yaml"
        report_path = Path(__file__).parent / report_filename

        with open(report_path, "w") as fp:
            yaml.dump(report, fp, sort_keys=True)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """pytest hook to capture test reports for YAML report generation."""
    # Only collect reports if the flag is enabled.
    if not item.config.getoption("--generate-report"):
        yield
        return

    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and "llm" in item.funcargs:
        module_name = item.module.__name__
        module_report = module_reports.setdefault(module_name, [])

        llm = item.funcargs["llm"]
        module_report.append((llm, _api_of(llm), report))


def _api_of(llm) -> str:
    """Names the backend an LLM talks to, for the report's `<api>://<model>` key."""
    return {"OpenAI": "openai", "GoogleGenAI": "genai"}.get(
        type(llm).__name__, type(llm).__name__.lower()
    )
