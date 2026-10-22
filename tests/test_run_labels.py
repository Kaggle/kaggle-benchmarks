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

import pandas as pd
import pytest

from kaggle_benchmarks import kaggle, tasks

FIRST = pd.DataFrame({"q": ["a", "b"]})
SECOND = pd.DataFrame({"q": ["c", "d"]})  # Same 0..1 index as FIRST.


@tasks.task()
def echo(llm, q: str) -> bool:
    llm.prompt(q)
    return True


@pytest.fixture()
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "kaggle_benchmarks.client", kaggle.KaggleClient(directory=tmp_path)
    )
    return tmp_path


def test_label_keeps_run_files_apart(run_dir, duck):
    echo.evaluate(evaluation_data=FIRST, label="first", llm=[duck])
    echo.evaluate(evaluation_data=SECOND, label="second", llm=[duck])

    names = [p.name for p in run_dir.glob("*.run.json")]
    assert len(names) == 4, names


def test_label_leaves_param_id_unchanged(run_dir, duck):
    runs_ = echo.evaluate(evaluation_data=FIRST, label="first", llm=[duck])

    assert list(runs_.as_dataframe()["id"]) == [0, 1]


def test_unlabelled_collision_warns(run_dir, duck, caplog):
    echo.evaluate(evaluation_data=FIRST, llm=[duck])
    echo.evaluate(evaluation_data=SECOND, llm=[duck])

    assert "Overwriting" in caplog.text
    assert len(list(run_dir.glob("*.run.json"))) == 2


def test_rerun_of_the_same_rows_does_not_warn(run_dir, duck, caplog):
    echo.evaluate(evaluation_data=FIRST, llm=[duck])
    caplog.clear()
    echo.evaluate(evaluation_data=FIRST, llm=[duck])

    assert "Overwriting" not in caplog.text


def test_label_cannot_escape_the_run_directory(run_dir, duck):
    echo.evaluate(evaluation_data=FIRST, label="../../escape", llm=[duck])

    assert list(run_dir.glob("*.run.json"))
    assert not list(run_dir.parent.glob("*.run.json"))


def test_cache_reads_the_file_of_the_same_label(tmp_path, monkeypatch, duck):
    monkeypatch.setattr(
        "kaggle_benchmarks.client",
        kaggle.KaggleClient(directory=tmp_path, use_cache=True),
    )
    echo.evaluate(evaluation_data=FIRST, label="first", llm=[duck])

    same = echo.evaluate(evaluation_data=FIRST, label="first", llm=[duck])
    other = echo.evaluate(evaluation_data=SECOND, label="second", llm=[duck])

    assert all(run.cached for run in same)
    assert not any(run.cached for run in other)
