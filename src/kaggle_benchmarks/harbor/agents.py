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
import os
import pathlib
import subprocess
import tempfile

HARNESSES = {
    "oneshot": "harbor.agents.oneshot.one_shot:OneShotAgent",
}

ONESHOT_RESPONSE_PATH = "/logs/artifacts/response.txt"


@dataclasses.dataclass(frozen=True)
class Agent:
    """What plays a task: a harness driving a model.

    `harness` is a Harbor agent name ("claude-code", "codex", "mini-swe-agent",
    "pi", ...), "oneshot" for a bare model, or a "module:Class" import path.
    `model` is a Kaggle model proxy slug, such as "google/gemini-3.6-flash".
    """

    harness: str
    model: str
    effort: str | None = None
    version: str | None = None

    @property
    def label(self) -> str:
        return f"{self.harness} + {self.model}"

    def env(self) -> dict[str, str]:
        env = {
            "KAGGLE_AGENT_HARNESS": HARNESSES.get(self.harness, self.harness),
            "KAGGLE_AGENT_LLM": self.model,
        }
        if self.effort:
            env["KAGGLE_AGENT_LLM_REASONING_EFFORT"] = self.effort
        if self.version:
            env["KAGGLE_AGENT_HARNESS_VERSION"] = self.version
        return env


@dataclasses.dataclass
class Result:
    agent: Agent
    reward: float | None
    error: str | None
    trial_dir: pathlib.Path | None


def read_result(output_dir: pathlib.Path, agent: Agent) -> Result:
    trials = sorted(output_dir.glob("jobs/*/*/result.json"))
    if not trials:
        return Result(agent, None, "no trial result found", None)
    trial = json.loads(trials[-1].read_text())
    rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
    exception = trial.get("exception_info") or {}
    return Result(
        agent=agent,
        reward=rewards.get("reward"),
        error=exception.get("exception_message"),
        trial_dir=trials[-1].parent,
    )


def run_agent(
    task_dir: pathlib.Path,
    agent: Agent,
    output_dir: pathlib.Path,
    runner: str | None = None,
    image: str | None = None,
) -> Result:
    """Runs one exported task with one agent on Kaggle's Harbor executor, locally.

    `runner` is Kaggle's `run-local-datasets.sh` (default: $KAGGLE_HARBOR_RUNNER).
    Model proxy credentials come from this process, the same ones kbench uses.
    """
    runner = runner or os.environ["KAGGLE_HARBOR_RUNNER"]
    output_dir.mkdir(parents=True, exist_ok=True)
    proxy_url = os.environ.get("MODEL_PROXY_URL", "").removesuffix("/")
    proxy_env = {
        "KH_MODEL_PROXY_API_KEY": os.environ.get("MODEL_PROXY_API_KEY", ""),
        "KH_MODEL_PROXY_BASE_URL": proxy_url.removesuffix("/openapi"),
    }
    lines = [f"{name}={value}" for name, value in agent.env().items()]
    # Expanded by the runner when it sources the file, so no secret is written.
    lines += [
        'MODEL_PROXY_API_KEY="$KH_MODEL_PROXY_API_KEY"',
        'MODEL_PROXY_BASE_URL="$KH_MODEL_PROXY_BASE_URL"',
        "unset MODEL_PROXY_API_KEYS",
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as env_file:
        env_file.write("\n".join(lines) + "\n")
    env = os.environ | proxy_env | {"OUTPUT_DIR": str(output_dir.resolve())}
    if image:
        env["HARBOR_IMAGE"] = image
    try:
        process = subprocess.run(
            [
                runner,
                "--env-file",
                env_file.name,
                "-y",
                "--task-def",
                task_dir.resolve(),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        os.unlink(env_file.name)
    (output_dir / "runner.log").write_text(process.stdout + process.stderr)
    return read_result(output_dir, agent)
