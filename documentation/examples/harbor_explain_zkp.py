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

# An answer graded by a judge model. On Harbor the judge runs in the check's own
# box, so the player never sees the criteria. `judge_model` fixes the judge, so
# trying it here and running it on Harbor use the same one.

import os
import sys

import kaggle_benchmarks as kbench
from kaggle_benchmarks import harbor as kh

task = kh.Task(
    "explain-zero-knowledge-proofs",
    description="Explain zero-knowledge proofs, graded by a judge.",
    judge_model="google/gemini-3.6-flash",
)


@task.question
def question():
    return "Explain Zero-Knowledge Proofs (ZKPs)."


@task.check
def check(answer):
    report = kbench.assertions.assess_response_with_judge(
        criteria=[
            "States that the prover convinces the verifier without revealing the secret.",
            "Mentions the three properties: completeness, soundness and zero-knowledge.",
            "Distinguishes interactive from non-interactive proofs.",
            "Gives a conceptual analogy, such as Ali Baba's cave.",
        ],
        response_text=answer,
        judge_llm=kbench.judge_llm,
    )
    if report is None:
        kbench.assertions.assert_fail("The judge should return an assessment.")
        return
    for result in report.results:
        kbench.assertions.assert_true(result.passed, result.criterion)


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
