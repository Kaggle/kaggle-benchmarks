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

# A dataset task with a typed answer. Each row becomes one Harbor task, and the
# correct option stays in the check: the player never sees it.

import dataclasses
import json
import os
import pathlib
import sys

import kagglehub
import pandas as pd

import kaggle_benchmarks as kbench
from kaggle_benchmarks import harbor as kh

path = pathlib.Path(kagglehub.dataset_download("tunguz/winogrande-dataset"))
df = pd.read_json(path / "train_s.jsonl", lines=True).head(5)
# `answer` is the name kbench gives the player's answer, so rename the column.
df = df.rename(columns={"answer": "correct"})


@dataclasses.dataclass
class Choice:
    option: int  # 1 or 2


task = kh.Task("winogrande", description="Fill in the blank.", rows=df)


@task.question
def question(sentence, option1, option2):
    return (
        "Fill in the blank (_) in this sentence with the right option.\n\n"
        f"{sentence}\n\n1: {option1}\n2: {option2}"
    )


@task.check
def check(answer: Choice, correct):
    kbench.assertions.assert_equal(correct, answer.option)


@task.solution
def solution(correct):
    return json.dumps({"option": correct})


task.run(kbench.llm)

workdir = sys.argv[1] if len(sys.argv) > 1 else "harbor_tasks"
print(task.export(workdir))

# Needs Docker, and KAGGLE_HARBOR_RUNNER pointing at Kaggle's run-local-datasets.sh.
if os.environ.get("KAGGLE_HARBOR_RUNNER"):
    task.run_local(
        [
            kh.Agent("oneshot", "google/gemini-3.6-flash"),
            kh.Agent("claude-code", "anthropic/claude-sonnet-5@default"),
        ],
        workdir=workdir,
    )
