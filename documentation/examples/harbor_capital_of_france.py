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

# Write a task once, try it here, then run it on Harbor with any agent:
# a harness (oneshot for a bare model, claude-code, codex, ...) plus a model.

import os
import sys

import kaggle_benchmarks as kbench
from kaggle_benchmarks import harbor as kh

task = kh.Task("capital-of-france", description="Name the capital of France.")


@task.question
def question():
    return "What is the capital of France?"


@task.check
def check(answer):
    kbench.assertions.assert_in("paris", answer.lower())


# A known-good answer, so Harbor's `oracle` player can check the task itself.
@task.solution
def solution():
    return "Paris"


# Try it with a model, right here.
task.run(kbench.llm)

workdir = sys.argv[1] if len(sys.argv) > 1 else "harbor_tasks"
print(task.export(workdir))

# Run it on Kaggle's Harbor executor on this machine. Needs Docker, and
# KAGGLE_HARBOR_RUNNER pointing at Kaggle's run-local-datasets.sh.
if os.environ.get("KAGGLE_HARBOR_RUNNER"):
    task.run_local(
        [
            kh.Agent("oneshot", "google/gemini-3.6-flash"),
            kh.Agent("claude-code", "anthropic/claude-sonnet-5@default"),
            kh.Agent("claude-code", "google/gemini-3.6-flash"),
            kh.Agent("codex", "openai/gpt-5.5-responses"),
            kh.Agent("mini-swe-agent", "openai/gpt-5.5-responses"),
            kh.Agent("pi", "google/gemini-3.6-flash"),
        ],
        workdir=workdir,
    )
