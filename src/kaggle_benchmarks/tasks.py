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
import datetime  # For Run start_time and end_time
import functools
import inspect
import logging
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    Self,
    TypeVar,
)

import pandas as pd

from kaggle_benchmarks import chats, events, results, utils

if TYPE_CHECKING:
    from kaggle_benchmarks import runs

T = TypeVar("T")

logger = logging.getLogger(__name__)


class NonRecoverableError(Exception):
    """Custom exception for non-recoverable errors during task running."""

    pass


@dataclasses.dataclass(frozen=True)
class Task(Generic[T]):
    func: Callable[..., T]
    name: str
    description: str = ""
    result_type: type[results.Result] = results.PassFail
    version: int = 1
    store_task: bool = True
    store_run: bool = True

    def __post_init__(self):
        from kaggle_benchmarks import client
        from kaggle_benchmarks._config import config

        name_limit = config.task_name_max_length
        if name_limit and len(self.name) > name_limit:
            raise ValueError(
                f"Task name is {len(self.name)} characters; the maximum "
                f"allowed is {name_limit}. Please shorten the 'name' "
                f"argument to @kbench.task(...)."
            )

        description_limit = config.task_description_max_length
        if description_limit and len(self.description) > description_limit:
            raise ValueError(
                f"Task description is {len(self.description)} characters; "
                f"the maximum allowed is {description_limit}. Please shorten "
                f"the 'description' argument to @kbench.task(...)."
            )

        client.register_task(self)
        events.manager.dispatch("new_task", self)

    def __call__(self, *args, **kwargs) -> T:
        with chats.new(self.name):
            return self.func(*args, **kwargs)

    def _handle_cached_run(self, run: "runs.Run[T]", ctx) -> "runs.Run[T] | None":
        from kaggle_benchmarks import client

        if client.skips_cached_run(run):
            try:
                logger.info(
                    f"Skipping run {run.id} for task {self.name} as its output file already exists."
                )
                run.cached = True
                run.status = utils.Status.SUCCESS
                run.result = client.load_run_result(run)
                if ctx.parent and ctx.parent.run:
                    ctx.parent.run.subruns.append(run)
                return run
            except Exception as e:
                logger.warning(f"Reading cached run failed: {e}")
        return None

    def _finalize_and_persist(self, run: "runs.Run[T]", ctx) -> None:
        """Stamp end_time, link to parent, and write the run file.

        Called after `contexts.enter` has exited, so `run.status` is
        already set to its final value (SUCCESS or FAILED) by
        `contexts.enter`'s own finally block. Skipped for cached runs,
        which are fully handled by `_handle_cached_run`.
        """
        from kaggle_benchmarks import client

        if run.cached:
            return

        run.end_time = datetime.datetime.now(datetime.timezone.utc)

        if ctx.parent and ctx.parent.run and run not in ctx.parent.run.subruns:
            ctx.parent.run.subruns.append(run)

        if self.store_run:
            try:
                client.store_run(run)
            except Exception as store_exc:
                # Persistence failure must not mask the original task
                # outcome (success or failure).
                logger.warning(f"Failed to store run {run.id}: {store_exc}")

    def run(self, *args, _id=None, **kwargs) -> "runs.Run[T]":
        from kaggle_benchmarks import contexts, runs

        # Internal flag set only by Task._evaluate_once() when
        # on_failure="continue". Popped from kwargs (not added to the public
        # signature) to keep it out of inspect.signature() and IDE autocomplete.
        _suppress_raise = kwargs.pop("_suppress_raise", False)

        signature = inspect.signature(self.func)
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        params = bound_args.arguments

        for param in signature.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                params.update(params.pop(param.name))
                break

        run = runs.Run(
            task=self,
            result=results.PENDING,
            params=params,
            param_id=_id,
        )

        try:
            with contexts.enter(run=run) as ctx:
                cached_run = self._handle_cached_run(run, ctx)
                if cached_run is not None:
                    # Cached path is fully handled by _handle_cached_run
                    # (parent linkage included). Early return skips both
                    # the except/else clauses below.
                    return cached_run

                run.start_time = datetime.datetime.now(datetime.timezone.utc)

                with chats.new(self.name, orphan=True) as chat:
                    try:
                        run.chat = chat
                        events.manager.dispatch("run_update", run)
                        run.result = self.func(*args, **kwargs)
                        if not self.result_type.check_value(run.result):
                            logger.warning(
                                f"Wrong return type {type(run.result)}. Expected {self.result_type._type}. This may need to lead to unexpected task behavior."
                            )
                    # Always make AssertionError non-blocking.
                    # This allows users to write/track native Python asserts within a task.
                    except AssertionError as e:
                        run.handle_assertion_exception(e)
                    # Let KeyboardInterrupt propagate to stop execution.
                    except (NonRecoverableError, KeyboardInterrupt):
                        raise
                    # Handle all other exceptions.
                    # Always re-raise — the outer except below decides
                    # whether to propagate further.
                    except Exception as e:
                        run.handle_general_exception(e)
        except (NonRecoverableError, KeyboardInterrupt):
            # Don't persist runs interrupted by Ctrl-C or unrecoverable
            # errors. contexts.enter's KbdInt handler skips setting the
            # FAILED status (see contexts.py:75-76), so a persisted run
            # would end up with state=COMPLETED and pollute the cache.
            raise
        except Exception:
            # contexts.enter has exited and set run.status=FAILED. Persist
            # the failed run for debugging and for the retry loop, then
            # decide whether to propagate.
            self._finalize_and_persist(run, ctx)
            if not _suppress_raise:
                raise
        else:
            # No exception reached us: either the task body succeeded
            # (status=SUCCESS) or contexts.enter swallowed a failure at
            # root with continue_with_exceptions=True (status=FAILED).
            # Either way, contexts.enter's finally has set the right
            # status — persist now.
            self._finalize_and_persist(run, ctx)

        return run

    def evaluate(
        self,
        grid: dict[str, Iterable[Any]] | None = None,
        evaluation_data: pd.DataFrame | None = None,
        n_jobs: int = 1,
        timeout: float | None = None,
        stop_condition: Callable[["runs.Runs[T]"], bool] | None = None,
        max_attempts: int = 1,
        retry_delay: int = 1,
        remove_run_files: bool = False,
        on_failure: Literal["raise", "continue"] = "raise",
        **kwargs: Iterable[Any],
    ) -> "runs.Runs[T]":
        """Evaluates the function over a grid of parameters, with optional retries.

        This method runs evaluations, optionally in parallel, across different
        parameter combinations and/or evaluation data rows. It can retry failed
        attempts.

        Args:
            grid: A dictionary defining the parameter grid to search.
                Keys are parameter names and values are iterables of the
                values to test.
            evaluation_data: An optional pandas DataFrame where each row
                             represents a separate evaluation to run.
            n_jobs: The number of jobs to run in parallel.
                    - If `n_jobs = 1` (default), runs sequentially in the main thread.
                    - If `n_jobs > 1`, runs in parallel using that many threads.
                    - If `n_jobs = -1`, uses all available CPU cores.
                    The underlying backend is "threading".
            timeout: The timeout in seconds for each job. If a job takes longer,
                     it is marked as a failure.
                     Note: With the "threading" backend, timed-out threads are
                     not killed and will run to completion. However this gives
                     a stuck job (e.g., due to timeout) a chance to restart.
            stop_condition: A function that takes a `runs.Runs` object and
                            returns `True` if the evaluation should stop.
                            If provided, the evaluation will stop when this
                            condition is met.
            max_attempts: The maximum number of evaluation attempts.
                          Defaults to 1 (no retries). When combined with
                          `on_failure="continue"` and `enable_cache()`,
                          subsequent attempts re-run only the previously
                          failed samples (cached successes are skipped);
                          results from all attempts are merged into the
                          returned `Runs`. The loop also exits early when
                          no failed runs remain.
                          Note: Nested task evaluations always run with
                          `max_attempts=1` regardless of the value passed;
                          a warning is logged if a different value is provided.
            retry_delay: The delay in seconds between retry attempts.
            remove_run_files: Remove generated run files when done.
            on_failure: How to handle per-sample task failures.
                - `"raise"` (default): if any sample fails, raise after the
                  current attempt finishes. In dev mode (no
                  `continue_with_exceptions`), a failure raises immediately
                  via the joblib worker. In Kaggle batch mode, the eval
                  finishes all parallel workers and then raises a summary —
                  so you still get a hard failure instead of a silent drop.
                - `"continue"`: catch per-sample failures, include the
                  failed runs in the returned `Runs`, and keep going.
                  Use `results.completed_runs` / `results.errored_runs` to
                  split them. Pair with `enable_cache()` + `max_attempts > 1`
                  to retry only the failures.
            **kwargs: Alternative way to specify the parameter grid.

        Returns:
            A `runs.Runs` object. With `on_failure="raise"` (default) it
            contains only successful runs. With `on_failure="continue"` it
            contains both successful and failed runs.
        """
        from kaggle_benchmarks import contexts, orchestration, runs
        from kaggle_benchmarks.kaggle import serialization

        ctx = contexts.get_current()
        if ctx.parent and ctx.parent.run and max_attempts > 1:
            logger.warning(
                "`max_attempts` must be 1 for nested task evaluations; coercing from %d to 1.",
                max_attempts,
            )
            max_attempts = 1

        if grid is None:
            grid = kwargs
        else:
            grid |= kwargs

        if isinstance(evaluation_data, pd.DataFrame):
            evaluation_data = evaluation_data.reset_index(names=["_id"])

        def _evaluate_once():
            # In "continue" mode, tell Task.run() not to re-raise so the
            # failed Run object comes back to us instead of crashing the
            # joblib worker. In "raise" mode, leave Task.run()'s behavior
            # untouched (it raises in dev, returns the failed Run in batch).
            runner = functools.partial(
                self.run, _suppress_raise=(on_failure == "continue")
            )
            all_runs = runs.Runs(
                [
                    run
                    for _, _, run in orchestration.evaluate_function(
                        runner,
                        grid=grid,
                        evaluation_data=evaluation_data,
                        n_jobs=n_jobs,
                        timeout=timeout,
                    )
                    if run is not None
                ]
            )

            # In "raise" mode, surface any failed runs that slipped through
            # because contexts.enter swallowed them (Kaggle batch). In dev
            # mode, the worker already raised and we never reach here.
            if on_failure == "raise":
                failures = [r for r in all_runs if r.status == utils.Status.FAILED]
                if failures:
                    first = failures[0]
                    msg = (
                        f"Task {self.name!r} run {first.id} failed:\n"
                        f"{first.error_message}"
                    )
                    if len(failures) > 1:
                        msg += (
                            f"\n\n({len(failures)} of {len(all_runs)} runs "
                            "failed. Pass on_failure='continue' to collect "
                            "failures instead of raising.)"
                        )
                    raise RuntimeError(msg)

            return all_runs

        # Accumulator across attempts, keyed by positional index within the
        # attempt. Position is stable across attempts because joblib returns
        # results in input order regardless of n_jobs or cache hits, and
        # because the grid/evaluation_data don't change between attempts.
        #
        # We use position rather than run.cache_id because cache_id falls
        # back to the per-run sequential `id` when param_id is None
        # (no evaluation_data), which would generate a fresh key on every
        # attempt and prevent the merge from ever overwriting.
        #
        # Python dicts preserve insertion order, and reassigning an existing
        # key updates the value without moving its iteration position — so
        # output order matches the original attempt-1 order.
        runs_by_position: dict[int, "runs.Run[T]"] = {}
        all_runs = runs.Runs()
        attempt = 0
        try:
            while attempt < max_attempts:
                attempt += 1
                if max_attempts > 1:
                    logging.info(f"Attempt {attempt}/{max_attempts}.")

                try:
                    attempt_runs = _evaluate_once()
                except Exception as e:
                    logging.warning(
                        f"An error occurred during evaluation attempt {attempt}: {e}"
                    )
                    # Always re-raise exception because the `continue_with_exceptions``
                    # setting should only apply to the subruns level.
                    raise e

                # Merge: new successes append at their position; retried
                # failures overwrite in place at the same position.
                for position, run in enumerate(attempt_runs):
                    runs_by_position[position] = run
                all_runs = runs.Runs(list(runs_by_position.values()))

                # Nothing failed — no point trying again.
                if not any(r.status == utils.Status.FAILED for r in all_runs):
                    break

                if stop_condition and stop_condition(all_runs):
                    logging.info("Stop condition met.")
                    break

                if attempt < max_attempts:
                    logging.info(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)

            if (
                attempt == max_attempts
                and stop_condition
                and not stop_condition(all_runs)
            ):
                logging.warning(
                    f"Maximum number of attempts ({max_attempts}) reached, but stop "
                    "condition not met. Exiting."
                )

            return all_runs
        finally:
            if remove_run_files:
                serialization.remove_runs_files(all_runs)

    def bind_dataframe(self, df: pd.DataFrame, **kwargs) -> Self:
        def func(**kwargs):
            result_df = self.evaluate(
                evaluation_data=df,
                grid={k: [v] for k, v in kwargs.items()},
            ).as_dataframe()

            return int(result_df.result.sum()), len(result_df)

        kwargs = (
            dataclasses.asdict(self)
            | kwargs
            | dict(func=func, result_type=results.PassCount)
        )
        return Task(**kwargs)

    def _repr_html_(self):
        return f"""
        <div class="task-info">
            <h3>{self.name}</h3>
            <pre>version: {self.version}</pre>
            <pre>description: {self.description}</pre>
            <pre>returns: {self.result_type.__name__}</pre>
        </div>
        """

    def partial(self, params: dict[str, Any], **kwargs) -> Self:
        return dataclasses.replace(
            self, func=functools.partial(self.func, **params), **kwargs
        )


def _infer_result_type(func: Callable[..., Any]) -> type:
    """Infers the result type for a task function based on its return annotation."""
    return_annotation = func.__annotations__.get("return")

    # No return type annotation, default to PassFail.
    if return_annotation is None:
        return results.PassFail
    # A return type annotation is a known type.
    # Note this doesn't work with parameterirzed generic type like dict[Any, float].
    elif return_annotation in results.types:
        return results.types[return_annotation]
    # The annotated type is not a known type, raise an error.
    else:
        supported_types_display = []
        for t_key in results.types.keys():
            supported_types_display.append(getattr(t_key, "__name__", str(t_key)))
        supported_types_str = ", ".join(
            f"'{s}'" for s in sorted(supported_types_display)
        )

        raise TypeError(
            f"Return type annotation '{getattr(return_annotation, '__name__', return_annotation)}' "
            f"for function '{func.__name__}' is not a registered result type for automatic inference. "
            f"Supported auto-inferred types are: [{supported_types_str}]. "
            f"If this is a custom type, ensure it's a subclass of 'benchmarks.results.Result' "
            f"and is correctly registered."
        )


def task(
    name: str | None = None,
    *,
    description: str | None = None,
    version: int = 1,
    store_task: bool = True,
    store_run: bool = True,
) -> Callable[..., Task]:
    def decorator(func: Callable[..., T]) -> Task[T]:
        rt = _infer_result_type(func)
        return Task(
            func=func,
            name=name or func.__name__.title().replace("_", " "),
            version=version,
            description=description or inspect.cleandoc(func.__doc__ or ""),
            result_type=rt,
            store_task=store_task,
            store_run=store_run,
        )

    return decorator


def benchmark(*args, **kwargs) -> Callable[..., Task]:
    return task(*args, **kwargs)
