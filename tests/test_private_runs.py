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

"""A run marked private must not appear in anything drawn or printed.

One test per way out: the repr, the Panel widgets, the console, the
dataframe. Plus the marking itself, a nested run inheriting it, and a
traceback that would quote the row.
"""

import io

import pandas as pd
import pytest

from kaggle_benchmarks import events, privacy, tasks, ui
from kaggle_benchmarks.runs import Split
from kaggle_benchmarks.ui import panel as ui_panel

SECRET = "hunter2-classified"


@tasks.task(name="Talker", store_task=False, store_run=False)
def talker(llm, message: str = "") -> float:
    llm.prompt(message)
    return float(len(message))


@pytest.fixture
def private_run(duck):
    return talker.run(llm=duck, message=SECRET, _split=Split.PRIVATE)


@pytest.fixture
def public_run(duck):
    return talker.run(llm=duck, message="public-question")


def rendered(viewable) -> str:
    with io.StringIO() as file:
        viewable.servable().save(file)
        file.seek(0)
        return file.read()


# -- Marking ---------------------------------------------------------------


def test_a_run_is_marked_by_the_caller_or_not_at_all(duck):
    assert talker.run(llm=duck, _split=Split.PRIVATE).split is Split.PRIVATE
    assert talker.run(llm=duck).split is None


def test_a_nested_run_inherits_the_split(duck):
    """Its conversation holds the same data as the row that started it."""

    @tasks.task(name="Inner", store_task=False, store_run=False)
    def inner(llm) -> float:
        return 1.0

    @tasks.task(name="Outer", store_task=False, store_run=False)
    def outer(llm) -> float:
        return inner.run(llm=llm).result

    run = outer.run(llm=duck, _split=Split.PRIVATE)

    assert [sub.split for sub in run.subruns] == [Split.PRIVATE]


# -- The ways out ----------------------------------------------------------


def test_repr_says_nothing(private_run, public_run):
    """The fallback whenever no renderer is available."""
    assert SECRET not in repr(private_run)
    assert SECRET not in str(private_run)
    assert SECRET not in f"{private_run}"
    assert "public-question" in repr(public_run)


def test_the_panel_view_says_nothing(private_run, public_run):
    assert SECRET not in rendered(ui_panel.render_run(private_run))
    assert "public-question" in rendered(ui_panel.render_run(public_run))


def test_a_listing_hides_the_row_and_says_how_many(private_run, public_run):
    from kaggle_benchmarks import runs as runs_module

    html = rendered(ui_panel.render_runs(runs_module.Runs([public_run, private_run])))

    assert SECRET not in html
    assert "public-question" in html
    assert "1 private row is hidden" in html


def test_the_console_says_nothing(duck):
    output = io.StringIO()
    events.manager.bind(ui.console.ConsoleUI(color=False, output=output))

    talker.run(llm=duck, message=SECRET, _split=Split.PRIVATE)

    assert SECRET not in output.getvalue()
    assert privacy.HIDDEN_LABEL in output.getvalue()


def test_the_dataframe_masks_the_row_and_stays_numeric(private_run, public_run):
    from kaggle_benchmarks import runs as runs_module

    frame = runs_module.Runs([public_run, private_run]).as_dataframe()

    assert SECRET not in frame.to_string()
    assert frame.loc[private_run.id, "message"] == privacy.MASK
    assert frame.loc[private_run.id, "split"] == "private"
    # Blank rather than marked, so arithmetic over the column still works.
    assert pd.isna(frame.loc[private_run.id, "result"])
    assert frame["result"].dtype == "float64"


def test_a_failed_private_run_does_not_raise_its_own_error(duck):
    @tasks.task(name="Exploder", store_task=False, store_run=False)
    def exploder(llm, message: str = "") -> float:
        raise ValueError(f"boom with {message}")

    with pytest.raises(privacy.PrivateRunError) as excinfo:
        exploder.run(llm=duck, message=SECRET, _split=Split.PRIVATE)

    assert SECRET not in str(excinfo.value)
    # Not attached as context either, where a traceback would walk to it.
    assert excinfo.value.__context__ is None


def test_the_original_error_is_kept_on_the_run(duck):
    """Withheld from the traceback, not discarded."""

    @tasks.task(name="Exploder2", store_task=False, store_run=False)
    def exploder(llm, message: str = "") -> float:
        raise ValueError(f"boom with {message}")

    @tasks.task(name="Catcher", store_task=False, store_run=False)
    def catcher(llm) -> float:
        with pytest.raises(privacy.PrivateRunError):
            exploder.run(llm=llm, message=SECRET)
        return 1.0

    # The inner run inherits private, and is linked to its parent before the
    # stand-in is raised, so we can still get at it.
    failed = catcher.run(llm=duck, _split=Split.PRIVATE).subruns[0]

    assert SECRET in failed.error_message


# -- Revealing -------------------------------------------------------------


def test_revealing_shows_the_row_again_and_then_stops(private_run, cfg):
    with privacy.reveal_private():
        assert SECRET in repr(private_run)
        assert SECRET in rendered(ui_panel.render_run(private_run))

    assert SECRET not in repr(private_run)


def test_a_public_run_is_never_affected(public_run, cfg):
    before = repr(public_run)
    with privacy.reveal_private():
        assert repr(public_run) == before
