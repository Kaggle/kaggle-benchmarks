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

import json

import pandas as pd
import pytest

import kaggle_benchmarks as kbench
from kaggle_benchmarks import kaggle, privacy, results, tasks
from kaggle_benchmarks.kaggle import atif
from kaggle_benchmarks.llm_messages import LLMMessage
from tests.mocks import MockedChat

# Only the private rows contain it, so any occurrence is a leak.
CANARY = "SECRET-row"

PUBLIC = pd.DataFrame({"q": ["public-a", "public-b"], "ok": [True, True]})
PRIVATE = pd.DataFrame({"q": [f"{CANARY}-1", f"{CANARY}-2"], "ok": [True, False]})


class FreshChat(MockedChat):
    """Returns a new message per call, as real backends do."""

    def invoke(self, messages, tools=None, **kwargs):
        return LLMMessage(sender=None, content="ok")


@pytest.fixture(autouse=True)
def warned(monkeypatch):
    """Silences the first-use warning of `evaluate_splits`."""
    monkeypatch.setattr(privacy, "_warned", True)


@pytest.fixture()
def llm():
    return FreshChat.from_contents(["ok"], cycle=True)


@pytest.fixture()
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "kaggle_benchmarks.client", kaggle.KaggleClient(directory=tmp_path)
    )
    return tmp_path


@tasks.task()
def row(llm, q: str, ok: bool) -> bool:
    llm.prompt(q)
    return ok


@tasks.task(name="scored")
def scored(llm) -> kbench.SplitScores:
    splits = row.evaluate_splits(public=PUBLIC, private=PRIVATE, llm=[llm])

    def mean(frame):
        return float(frame.result.mean())

    return kbench.SplitScores(
        overall=mean(splits.as_dataframe(include_hidden=True)),
        public=mean(splits["public"].as_dataframe()),
        private=(mean(splits["private"].as_dataframe()), 0.1),
    )


def latest_run_file(run_dir):
    return max(run_dir.glob("scored*.run.json"), key=lambda p: p.stat().st_mtime)


def test_annotation_infers_split_scores_result():
    assert scored.result_type is results.SplitScoresResult


def test_run_file_holds_one_result_per_split(run_dir, llm):
    scored.run(llm=llm)

    saved = json.loads(latest_run_file(run_dir).read_text())["results"]

    assert [entry["type"] for entry in saved] == ["AGGREGATED", "PUBLIC", "PRIVATE"]
    assert [entry["numericResult"]["value"] for entry in saved] == [0.75, 1.0, 0.5]
    assert saved[2]["numericResult"]["confidenceInterval"] == 0.1
    assert "confidenceInterval" not in saved[0]["numericResult"]


def test_format_lists_every_score(llm):
    run = scored.run(llm=llm)

    assert results.SplitScoresResult.format(run) == (
        "overall 0.75, public 1.0, private (0.5, 0.1)"
    )


def test_choose_leaves_scores_but_no_private_rows(run_dir, llm):
    scored.run(llm=llm)

    # What %choose keeps: the main task file, its latest run and its trajectory.
    kept = latest_run_file(run_dir)
    keep = {run_dir / "scored.task.json", kept, *atif.paths_beside(kept)}
    for path in run_dir.glob("*.json"):
        if path not in keep:
            path.unlink()

    assert not any(CANARY in path.read_text() for path in run_dir.glob("*"))
    assert len(json.loads(kept.read_text())["results"]) == 3
