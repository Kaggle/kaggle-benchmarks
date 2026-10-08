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

import io
import json

import pandas as pd
import pytest

import kaggle_benchmarks as kbench
from kaggle_benchmarks import assertions, contexts, events, kaggle, privacy, runs, tasks
from kaggle_benchmarks.kaggle import serialization
from kaggle_benchmarks.llm_messages import LLMMessage
from kaggle_benchmarks.ui import panel as ui_panel
from kaggle_benchmarks.ui.console import ConsoleUI
from kaggle_benchmarks.ui.panel import PanelUI
from tests.mocks import MockedChat

# Only the private rows contain it, so any occurrence is a leak.
CANARY = "SECRET-row"

PUBLIC = pd.DataFrame({"q": ["public-a", "public-b"]})
PRIVATE = pd.DataFrame({"q": [f"{CANARY}-1", f"{CANARY}-2"]})


class FreshChat(MockedChat):
    """Returns a new message per call, as real backends do."""

    def invoke(self, messages, tools=None, **kwargs):
        return LLMMessage(sender=None, content="ok")


@pytest.fixture(autouse=True)
def warned(monkeypatch):
    """Silences the first-use warning; `test_first_call_warns_once` covers it."""
    monkeypatch.setattr(privacy, "_warned", True)


@pytest.fixture()
def llm():
    return FreshChat.from_contents(["ok"], cycle=True)


@pytest.fixture()
def run_dir(tmp_path, monkeypatch, cfg):
    """The published directory. `config.private_directory` is outside it."""
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.setattr(
        "kaggle_benchmarks.client", kaggle.KaggleClient(directory=working)
    )
    cfg.private_directory = tmp_path / "private"
    return working


@tasks.task()
def echo(llm, q: str) -> bool:
    llm.prompt(q)
    return True


@tasks.task(name="main")
def main(llm) -> float:
    splits = echo.evaluate_splits(public=PUBLIC, private=PRIVATE, llm=[llm])
    return float(splits["private"].as_dataframe().result.mean())


def rendered(viewable) -> str:
    """Returns the HTML a notebook saves for `viewable`."""
    buffer = io.StringIO()
    viewable.save(buffer, embed=True, resources="inline")
    return buffer.getvalue()


def bound(ui, func):
    """Calls `func` with `ui` bound to the event manager."""
    events.manager.bind(ui)
    try:
        return func()
    finally:
        events.manager.unbind(ui)


def console_output(func) -> str:
    captured = io.StringIO()
    bound(ConsoleUI(output=captured), func)
    return captured.getvalue()


def test_each_split_is_a_runs(llm):
    splits = echo.evaluate_splits(public=PUBLIC, private=PRIVATE, llm=[llm])

    assert set(splits) == {"public", "private"}
    assert isinstance(splits["private"], runs.Runs)
    assert splits["private"].as_dataframe().result.mean() == 1.0


def test_row_labels_are_unchanged(llm):
    splits = echo.evaluate_splits(public=PUBLIC, private=PRIVATE, llm=[llm])

    assert list(splits["public"].as_dataframe()["id"]) == [0, 1]
    assert list(splits["private"].as_dataframe()["id"]) == [0, 1]


@pytest.mark.parametrize("n_jobs", [1, 4])
def test_private_rows_write_no_files(run_dir, llm, cfg, n_jobs):
    echo.evaluate_splits(public=PUBLIC, private=PRIVATE, llm=[llm], n_jobs=n_jobs)

    assert len(list(run_dir.glob("*.run.json"))) == len(PUBLIC)
    assert not any(CANARY in path.read_text() for path in run_dir.iterdir())
    assert not cfg.private_directory.exists()


def test_private_rows_are_not_cached(run_dir, llm):
    from kaggle_benchmarks import client

    with client.enable_cache():
        echo.evaluate_splits(public=PUBLIC, private=PRIVATE, llm=[llm])
        splits = echo.evaluate_splits(public=PUBLIC, private=PRIVATE, llm=[llm])

    assert all(run.cached for run in splits["public"])
    assert not any(run.cached for run in splits["private"])


def test_as_dataframe_excludes_hidden_by_default(llm):
    splits = echo.evaluate_splits(public=PUBLIC, private=PRIVATE, llm=[llm])

    assert set(splits.as_dataframe().split) == {"public"}
    assert CANARY not in splits.as_dataframe().to_string()
    assert set(splits.as_dataframe(include_hidden=True).split) == {"public", "private"}


def test_as_dataframe_rejects_a_split_parameter(llm):
    @tasks.task()
    def with_split(llm, q: str, split: str) -> bool:
        return True

    frame = PUBLIC.assign(split="train")
    splits = with_split.evaluate_splits(public=frame, private=frame, llm=[llm])

    with pytest.raises(ValueError, match="already have a `split` column"):
        splits.as_dataframe()


def test_summary_and_repr_hide_private_rows(llm):
    splits = echo.evaluate_splits(public=PUBLIC, private=PRIVATE, llm=[llm])

    assert CANARY not in repr(splits)
    html = rendered(ui_panel.render_splits(splits))
    assert CANARY not in html
    assert "public-a" in html


def test_named_hidden_split_is_displayed(llm):
    splits = echo.evaluate_splits(public=PUBLIC, private=PRIVATE, llm=[llm])

    assert CANARY in rendered(splits["private"].__panel__())


@pytest.mark.parametrize("n_jobs", [1, 4])
def test_console_prints_nothing_for_private_rows(llm, n_jobs):
    # n_jobs=4 checks that the hidden scope reaches worker threads.
    printed = console_output(
        lambda: echo.evaluate_splits(
            public=PUBLIC, private=PRIVATE, llm=[llm], n_jobs=n_jobs
        )
    )

    assert CANARY not in printed
    assert "public-a" in printed


def test_console_hides_private_assertions_and_results(llm):
    @tasks.task()
    def checked(llm, q: str) -> dict:
        reply = str(llm.prompt(q))
        assertions.assert_contains_regex(
            "no-such-text", reply, expectation=f"{q!r} should match"
        )
        return {"echo": q}

    printed = console_output(
        lambda: checked.evaluate_splits(
            public=PUBLIC, private=PRIVATE, llm=[llm], on_failure="continue"
        )
    )

    assert CANARY not in printed
    assert "public-a" in printed


def test_panel_hides_private_runs(llm):
    ui = PanelUI()
    bound(ui, lambda: echo.evaluate_splits(public=PUBLIC, private=PRIVATE, llm=[llm]))

    assert CANARY not in rendered(ui.feed)


def test_parent_run_leaves_out_private_rows(llm):
    run = main.run(llm=llm)

    assert run.result == 1.0
    assert len(run.subruns) == len(PUBLIC)
    assert CANARY not in repr(run)
    assert CANARY not in rendered(ui_panel.render_run(run))


def test_run_directory_never_holds_private_rows(run_dir, llm):
    main.run(llm=llm)

    assert not any(CANARY in path.read_text() for path in run_dir.glob("*"))


def test_private_rows_are_written_outside_the_run_directory(run_dir, llm, cfg):
    run = main.run(llm=llm)

    path = serialization.private_run_path(
        kaggle.KaggleClient(directory=run_dir)._get_run_filepath(run)
    )
    assert path.parent == cfg.private_directory
    # Never under the directory a notebook publishes.
    assert not path.is_relative_to(run_dir)

    content = path.read_text()
    assert len(json.loads(content)["subruns"]) == len(PRIVATE)
    assert CANARY in content
    assert "public-a" not in content


def test_private_file_holds_the_whole_private_half(run_dir, llm, cfg):
    main.run(llm=llm)

    [path] = cfg.private_directory.iterdir()
    subruns = json.loads(path.read_text())["subruns"]
    assert len(subruns) == len(PRIVATE)
    assert all(subrun["conversations"] for subrun in subruns)


def test_private_file_is_written_when_the_main_task_fails(run_dir, llm, cfg):
    @tasks.task(name="failing_main")
    def failing_main(llm) -> float:
        echo.evaluate_splits(public=PUBLIC, private=PRIVATE, llm=[llm])
        raise RuntimeError("after the splits")

    with pytest.raises(RuntimeError):
        failing_main.run(llm=llm)

    [path] = cfg.private_directory.iterdir()
    assert len(json.loads(path.read_text())["subruns"]) == len(PRIVATE)


def test_private_file_is_not_a_run_file(run_dir, llm, cfg):
    """The platform takes the first output path that ends in `run.json`.

    A second one would race with the real run file, so the private file must
    not end in that suffix.
    """
    main.run(llm=llm)

    written = list(cfg.private_directory.iterdir())
    assert written, "expected a private file"
    for path in written:
        assert path.name.endswith(serialization.PRIVATE_FILE_SUFFIX)
        assert not path.name.endswith(serialization.RUN_FILE_SUFFIX)


def test_nothing_written_without_a_calling_run(run_dir, llm, cfg):
    echo.evaluate_splits(public=PUBLIC, private=PRIVATE, llm=[llm])

    assert not cfg.private_directory.exists()


def test_reveal_hidden_does_not_reattach_private_rows(cfg, llm):
    with kbench.reveal_hidden():
        run = main.run(llm=llm)

    assert len(run.subruns) == len(PUBLIC)


def test_private_failure_hides_the_row():
    @tasks.task()
    def boom(q: str) -> bool:
        raise ValueError(f"failed on: {q}")

    with contexts.hidden():
        with pytest.raises(privacy.HiddenRunError) as excinfo:
            boom.run(q=f"{CANARY}-x")

    assert CANARY not in str(excinfo.value)
    assert excinfo.value.__context__ is None


def test_hiding_ends_with_the_call(llm):
    echo.evaluate_splits(public=PUBLIC, private=PRIVATE, llm=[llm])

    assert not privacy.output_hidden()


def test_reveal_hidden(cfg):
    with contexts.hidden():
        assert privacy.output_hidden()
        with kbench.reveal_hidden():
            assert not privacy.output_hidden()
        assert privacy.output_hidden()


def test_kwargs_reach_evaluate(llm):
    splits = echo.evaluate_splits(public=PUBLIC, private=PRIVATE, llm=[llm], n_jobs=2)

    assert len(splits["public"]) == 2


def test_invalid_input_fails_before_either_set_runs(run_dir, llm):
    with pytest.raises(TypeError, match="goes in grid"):
        echo.evaluate_splits(public=["a"], private=PRIVATE, llm=[llm])
    with pytest.raises(TypeError) as excinfo:
        echo.evaluate_splits(public={"q": "a"}, private=PRIVATE, llm=[llm])
    assert "grid" not in str(excinfo.value)
    with pytest.raises(TypeError, match="sets \\['label'\\] itself"):
        echo.evaluate_splits(public=PUBLIC, private=PRIVATE, label="x", llm=[llm])
    with pytest.raises(ValueError, match="same columns"):
        echo.evaluate_splits(
            public=PUBLIC, private=pd.DataFrame({"other": ["x"]}), llm=[llm]
        )
    with pytest.raises(ValueError, match="empty"):
        echo.evaluate_splits(public=PUBLIC, private=PRIVATE.head(0), llm=[llm])

    assert not list(run_dir.glob("*.run.json"))


def test_unknown_hidden_split_is_rejected():
    with pytest.raises(ValueError, match="Unknown hidden"):
        kbench.Splits(splits={"a": runs.Runs()}, hidden=frozenset({"typo"}))


def test_hand_built_splits_hide_only_the_display(llm):
    first = echo.evaluate(evaluation_data=PUBLIC, label="a", llm=[llm])
    second = echo.evaluate(evaluation_data=PRIVATE, label="b", llm=[llm])

    splits = kbench.Splits(splits={"a": first, "b": second}, hidden=frozenset({"b"}))

    assert set(splits.as_dataframe().split) == {"a"}
    assert CANARY not in repr(splits)


def test_first_call_warns_once(cfg, llm, monkeypatch):
    monkeypatch.setattr(privacy, "_warned", False)

    with pytest.warns(UserWarning, match="private_directory") as caught:
        echo.evaluate_splits(public=PUBLIC, private=PRIVATE, llm=[llm])
        echo.evaluate_splits(public=PUBLIC, private=PRIVATE, llm=[llm])

    first_use = [w for w in caught if w.message.args[0] == privacy.FIRST_USE_WARNING]
    assert len(first_use) == 1
    assert first_use[0].filename == __file__
