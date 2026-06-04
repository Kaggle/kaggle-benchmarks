# Copyright 2025 Kaggle Inc.
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
import datetime
import threading
import traceback
from collections import abc
from typing import Any, Generic, Literal, Self, TypeVar

import pandas as pd

from kaggle_benchmarks import actors, assertions, chats, results, tasks, utils
from kaggle_benchmarks.actors.llms import LLMChat

T = TypeVar("T")

_run_counters_lock = threading.Lock()
_run_counters: dict[int, int] = {}


@dataclasses.dataclass
class Run(Generic[T]):
    task: tasks.Task[T]
    result: T | results.Unknown = results.PENDING
    chat: chats.Chat | None = None
    status: utils.Status = utils.Status.PENDING
    params: dict[str, Any] = dataclasses.field(default_factory=dict)
    id: str = ""
    param_id: Any = None
    subruns: "Runs" = dataclasses.field(default_factory=lambda: Runs())
    assertion_results: list[assertions.AssertionResult] = dataclasses.field(
        default_factory=list
    )
    start_time: datetime.datetime | None = None
    end_time: datetime.datetime | None = None
    # Whether the result of this run was retrieved from a cache.
    cached: bool = False
    # Details for status=FAILED
    error_message: str | None = None

    def __post_init__(self):
        if self.id == "":
            with _run_counters_lock:
                _run_counters.setdefault(id(self.task), 0)
                _run_counters[id(self.task)] += 1
                self.id = f"Run #{_run_counters[id(self.task)]}"

    def handle_assertion_exception(self, e: AssertionError):
        """Helper to handle assertion errors during a task run."""
        assertion_result = assertions.AssertionResult(
            passed=False,
            expectation=f"Model expected to return response in correct format but got {type(e).__name__}: {e}",
        )
        actors.assertion.send(assertion_result, is_visible_to_llm=False)
        self.assertion_results.append(assertion_result)
        self.result = results.FAILED

    def fail_with_message(self, msg: str | None = None):
        """Sets the run status to FAILED and records the error message."""
        self.status = utils.Status.FAILED
        self.error_message = msg
        self.result = results.FAILED

    def handle_general_exception(self, e: Exception):
        """Helper to handle general exceptions during a task run.

        It will mark the run's status as FAILED.
        """

        summary = traceback.format_exc()
        self.fail_with_message(summary)

        raise e

    @property
    def cache_id(self) -> str:
        """Gets a unique ID for a run, primarily for caching purposes.

        This function constructs the ID using the run's `param_id` (prioritized, typically
        corresponding to the evaluation data's index) and falls back to the run's
        sequential `id` if `param_id` is not set.

        Additionally, if an `evaluated_subject` is present, its name is appended
        as a suffix to distinguish results across different subjects or actors.
        """
        actor_model = None
        if param := self.evaluated_subject:
            actor_model = getattr(param, "model", None) or param.name

        actor_param_suffix = f"_{actor_model}" if actor_model else ""

        return (
            f"run_param_id_{self.param_id}{actor_param_suffix}"
            if self.param_id is not None
            else f"run_id_{self.id}{actor_param_suffix}"
        )

    @property
    def name(self) -> str:
        return ", ".join(
            f"{k}={getattr(v, 'name', v)}" for k, v in self.params.items()
        ) + (f"/{self.id}" if self.id else "")

    @property
    def parent(self) -> Self | None:
        from kaggle_benchmarks import contexts

        context = contexts.get_current()
        while context.parent:
            context = context.parent
            if context.run is not self:
                return context.run

    @property
    def passed(self):
        # Short-circuit only for cached SUCCESS runs. A cached FAILED run
        # (possible once Task.run() persists failures) should report False.
        if self.cached and self.status == utils.Status.SUCCESS:
            return True
        # A run that hit a general exception in the task body is FAILED.
        # Most result types' `passed(value)` returns True regardless of the
        # value passed, so without this guard a failed run with no failing
        # assertions would incorrectly report passed=True.
        if self.status == utils.Status.FAILED:
            return False
        return (
            self.task.result_type.passed(self.result)
            and all(result.passed for result in self.assertion_results)
            and all(run.passed for run in self.subruns)
        )

    @property
    def evaluated_subject(self) -> LLMChat | None:
        """Retrieves the primary benchmark candidate for this task.

        Scans the parameters and returns the first `LLMChat` instance found.
        Assumes the first match is the subject of the benchmark.
        """
        # TODO (b/419612137): This implementation is currently hardcoded for LLMChat.
        return next(
            (p_val for p_val in self.params.values() if isinstance(p_val, LLMChat)),
            None,
        )

    def format_result(self):
        if self.cached:
            return "✅ (cached)"
        if self.result is results.PENDING:
            return "..."
        elif self.result is results.FAILED:
            return "❌"

        return self.task.result_type.format(self)

    def __panel__(self):
        from kaggle_benchmarks.ui import panel

        return panel.render_run(self)

    def _repr_mimebundle_(self, include=None, exclude=None):
        return self.__panel__()._repr_mimebundle_(include, exclude)


@dataclasses.dataclass
class Runs(Generic[T], abc.MutableSequence):
    runs: list[Run[T]] = dataclasses.field(default_factory=list)

    def __setitem__(self, index, value):
        self.runs[index] = value

    def __getitem__(self, index):
        return self.runs[index]

    def __delitem__(self, index):
        del self.runs[index]

    def insert(self, index, value: Run[T]):
        self.runs.insert(index, value)

    def __len__(self):
        return len(self.runs)

    @property
    def completed_runs(self) -> "Runs[T]":
        """Runs whose execution completed (status=SUCCESS).

        Note: "completed" is not the same as "passed". A completed run
        may still have failed assertions or scored 0. Use `run.passed`
        to check that.
        """
        return Runs([r for r in self.runs if r.status == utils.Status.SUCCESS])

    @property
    def errored_runs(self) -> "Runs[T]":
        """Runs that hit an infrastructure error (status=FAILED).

        These appear when `evaluate(..., on_failure="continue")` is used.
        They're the candidates that `max_attempts > 1` will retry on the
        next attempt (with `enable_cache()` enabled).
        """
        return Runs([r for r in self.runs if r.status == utils.Status.FAILED])

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                t.params
                | {"run_id": t.id, "result": t.result}
                | ({"id": t.param_id} if t.param_id is not None else {})
                for t in self.runs
            ]
        ).set_index("run_id")

    def group_by(self, by="llm"):
        from kaggle_benchmarks.ui import panel

        runs = []
        for run in self.runs:
            if run.subruns:
                runs.extend(run.subruns)
            else:
                runs.append(run)

        groups = {}
        for run in runs:
            params = ", ".join(f"{k} = {v}" for k, v in run.params.items() if k != by)
            key = f"{run.task.name}\n{params}"
            groups.setdefault(key, {})[str(run.params[by])] = run

        return panel.render_groups(groups=groups)

    def pivot(self, by: str = "llm", mode: Literal["tabs", "columns"] = "columns"):
        from kaggle_benchmarks.ui import panel

        groups: dict[Any, dict[str, Run]] = {}
        for run in self.runs:
            groups.setdefault(run.params[by], {})[run.param_id] = run

        return panel.render_pivot(groups, mode=mode)

    def __panel__(self):
        from kaggle_benchmarks.ui import panel

        return panel.render_runs(self)

    def _repr_mimebundle_(self, include=None, exclude=None):
        return self.__panel__()._repr_mimebundle_(include, exclude)
