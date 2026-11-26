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

"""Canaries for `contexts.hidden()`: a token that only private rows carry must
reach no file the client writes, no UI and no log."""

import contextlib
import io

import pandas as pd
import pytest

from kaggle_benchmarks import assertions, contexts, events, kaggle, task, tasks, utils
from kaggle_benchmarks.ui.console import ConsoleUI
from tests.mocks import MockedChat

CANARY = "s3cr3t-canary"
# Each route is one way the token can get into a run file. All but "direct" go
# through a row; "direct" is the parent task itself prompting and asserting.
ROUTES = ["prompt", "exception", "nested_run", "direct"]
# Each failure is one way an error carrying the token can leave the scope.
FAILURES = ["row", "raise", "assert"]


@pytest.fixture
def client(monkeypatch, tmp_path, cfg):
    # On, so the trajectory files are checked along with the run files.
    cfg.write_atif = True
    kaggle_client = kaggle.KaggleClient(directory=tmp_path)
    monkeypatch.setattr("kaggle_benchmarks.client", kaggle_client)
    return kaggle_client


@pytest.fixture
def llm():
    return MockedChat.from_contents(["an answer"], cycle=True)


@task()
def nested(label: str):
    assertions.assert_equal(label, "an answer")


@task()
def row(llm, question: str, label: str, route: str):
    llm.prompt(question)
    if route == "exception":
        {}[label]
    elif route == "nested_run":
        nested.run(label=label)


def frame(route: str, token: str) -> pd.DataFrame:
    question = f"What is {token}?" if route == "prompt" else "What is it?"
    return pd.DataFrame([{"question": question, "label": token, "route": route}])


@pytest.fixture
def exposed(client, caplog):
    """Reads all that a notebook shows or keeps: files, UI output and the log."""
    printed = io.StringIO()
    ui = ConsoleUI(output=printed)
    events.manager.bind(ui)

    def read() -> str:
        files = (p.read_text() for p in client.directory.rglob("*") if p.is_file())
        return "".join(files) + printed.getvalue() + caplog.text

    yield read
    events.manager.unbind(ui)


def evaluate(llm, route: str, hidden: bool):
    """Runs a parent task over one public and one private row."""

    @task()
    def parent(llm) -> int:
        public = row.evaluate(
            evaluation_data=frame(route, "public"), llm=[llm], on_failure="continue"
        )
        with contexts.hidden() if hidden else contextlib.nullcontext():
            # n_jobs=2: the row runs on a worker thread, which must inherit the scope.
            private = row.evaluate(
                evaluation_data=frame(route, CANARY),
                llm=[llm],
                on_failure="continue",
                n_jobs=2,
            )
            if route == "direct":
                llm.prompt(f"What is {CANARY}?")
                assertions.assert_equal(CANARY, "an answer")
        return len(public) + len(private)

    return parent.run(llm=llm)


@pytest.mark.parametrize("route", ROUTES)
def test_private_row_is_not_exposed(client, exposed, llm, route):
    run = evaluate(llm, route, hidden=True)

    assert run.result == 2  # Both rows ran.
    assert len(run.subruns) == 1  # Only the public row is linked.
    assert CANARY not in exposed()
    assert "What is" in exposed()  # The public row is still shown.
    # The parent and its public row are still written.
    assert len(list(client.directory.glob("Parent-*.run.json"))) == 1
    assert len(list(client.directory.glob("Row-*.run.json"))) == 1
    # Of the two nested runs, only the public row's.
    assert len(list(client.directory.glob("Nested-*.run.json"))) == (
        route == "nested_run"
    )


@pytest.mark.parametrize("route", ROUTES)
def test_canary_is_live(exposed, llm, route):
    """Without the scope the same row does leak, so the test above means something."""
    evaluate(llm, route, hidden=False)

    assert CANARY in exposed()


def fail_in_scope(llm, route: str, hidden: bool):
    """Runs a parent task that an error carrying the token leaves the scope of."""

    @task()
    def parent(llm):
        with contexts.hidden() if hidden else contextlib.nullcontext():
            if route == "row":  # on_failure="raise": the row's error leaves evaluate().
                row.evaluate(evaluation_data=frame("exception", CANARY), llm=[llm])
            elif route == "raise":
                raise KeyError(CANARY)
            elif route == "assert":
                assert False, CANARY

    return parent.run(llm=llm)


@pytest.mark.parametrize("route", ["row", "raise"])
def test_error_leaving_the_scope_is_withheld(exposed, llm, route):
    with pytest.raises(RuntimeError, match="withheld"):
        fail_in_scope(llm, route, hidden=True)

    assert CANARY not in exposed()
    # The parent's file is written, and says that it failed.
    assert "BENCHMARK_TASK_RUN_STATE_ERRORED" in exposed()


def test_withheld_error_is_kept_on_the_sink(llm):
    seen = {}

    @task()
    def parent(llm):
        with contexts.hidden() as scope:
            seen["sink"] = scope.run
            raise KeyError(CANARY)

    with pytest.raises(RuntimeError, match="withheld"):
        parent.run(llm=llm)

    assert CANARY in seen["sink"].error_message


def test_assert_leaving_the_scope_is_a_failed_assertion(exposed, llm):
    """As for a bare assert anywhere in a task: it does not fail the run."""
    run = fail_in_scope(llm, "assert", hidden=True)

    assert run.status == utils.Status.SUCCESS
    assert not run.passed
    (assertion,) = run.assertion_results
    assert "withheld" in assertion.expectation
    assert CANARY not in exposed()


@pytest.mark.parametrize("route", FAILURES)
def test_failure_canary_is_live(exposed, llm, route):
    # A bare assert is recorded as a failed assertion, not raised.
    with contextlib.suppress(KeyError):
        fail_in_scope(llm, route, hidden=False)

    assert CANARY in exposed()


def test_non_recoverable_error_keeps_its_type(client, llm):
    """Its message is withheld too: a tool call turns any error into a message."""

    @task()
    def parent(llm):
        with contexts.hidden():
            raise tasks.NonRecoverableError(CANARY)

    with pytest.raises(tasks.NonRecoverableError, match="withheld") as error:
        parent.run(llm=llm)

    assert CANARY not in str(error.value)
    assert not list(client.directory.glob("*.run.json"))  # Still stops the task.


def test_hidden_row_ignores_the_cache(client, llm):
    """A public and a private row share a cache_id, so a cache hit would score
    the private row with the public row's result, and never run it."""
    calls = []

    @task()
    def counted(llm, label: str) -> bool:
        calls.append(label)
        return label == "public"

    @task()
    def parent(llm) -> bool:
        with client.enable_cache():
            counted.evaluate(
                evaluation_data=pd.DataFrame([{"label": "public"}]), llm=[llm]
            )
            with contexts.hidden():
                private = counted.evaluate(
                    evaluation_data=pd.DataFrame([{"label": CANARY}]), llm=[llm]
                )
        return private[0].result

    run = parent.run(llm=llm)

    assert calls == ["public", CANARY]
    assert run.result is False


def test_hidden_rows_leave_visible_run_files_alone(client, llm):
    """A hidden row has no file, but the visible row of the same index does,
    under the same name."""

    @task()
    def parent(llm) -> bool:
        row.evaluate(evaluation_data=frame("prompt", "public"), llm=[llm])
        with contexts.hidden():
            row.evaluate(
                evaluation_data=frame("prompt", CANARY),
                llm=[llm],
                remove_run_files=True,
            )
        return True

    parent.run(llm=llm)

    assert len(list(client.directory.glob("Row-*.run.json"))) == 1


def test_runs_in_scope_are_hidden_and_link_to_the_sink(llm):
    seen = {}

    @task()
    def parent(llm) -> bool:
        seen["before"] = nested.run(label="an answer")
        with contexts.hidden() as scope:
            seen["sink"] = scope.run
            # n_jobs=2: the row runs on a worker thread, which must inherit the scope.
            seen["inside"] = row.evaluate(
                evaluation_data=frame("nested_run", "an answer"), llm=[llm], n_jobs=2
            )[0]
        seen["after"] = nested.run(label="an answer")
        return True

    run = parent.run(llm=llm)
    sink, inside = seen["sink"], seen["inside"]

    assert not run.hidden
    assert list(run.subruns) == [seen["before"], seen["after"]]
    assert not seen["before"].hidden and not seen["after"].hidden
    # The sink stands in for the parent, without taking a run number of its own.
    assert sink.hidden
    assert sink is not run
    assert sink.id == run.id
    assert list(sink.subruns) == [inside]
    assert inside.hidden and inside.subruns[0].hidden


def test_hidden_needs_a_run():
    with pytest.raises(RuntimeError, match="inside a task"), contexts.hidden():
        pass
