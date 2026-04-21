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
from typing import TYPE_CHECKING, Any, Callable, Generic, Iterable, Self, TypeVar

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

        if (
            config.task_name_max_length > 0
            and len(self.name) > config.task_name_max_length
        ):
            raise ValueError(
                f"Task name is {len(self.name)} characters; the maximum "
                f"allowed is {config.task_name_max_length}. Please shorten "
                f"the 'name' argument to @kbench.task(...)."
            )

        if (
            config.task_description_max_length > 0
            and len(self.description) > config.task_description_max_length
        ):
            raise ValueError(
                f"Task description is {len(self.description)} characters; "
                f"the maximum allowed is {config.task_description_max_length}. "
                f"Please shorten the 'description' argument to "
                f"@kbench.task(...)."
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

    def run(self, *args, _id=None, **kwargs) -> "runs.Run[T]":
        from kaggle_benchmarks import client, contexts, runs

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

        with contexts.enter(run=run) as ctx:
            cached_run = self._handle_cached_run(run, ctx)
            if cached_run is not None:
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
                # Always re-raise the exception because it will be handled in contexts.enter.
                except Exception as e:
                    run.handle_general_exception(e)

        run.end_time = datetime.datetime.now(datetime.timezone.utc)

        if ctx.parent and ctx.parent.run:
            ctx.parent.run.subruns.append(run)

        if self.store_run:
            client.store_run(run)

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
                          Defaults to 1 (no retries).
                          Note: Nested task evaluations always run with
                          `max_attempts=1` regardless of the value passed;
                          a warning is logged if a different value is provided.
            retry_delay: The delay in seconds between retry attempts.
            remove_run_files: Remove generated run files when done.
            **kwargs: Alternative way to specify the parameter grid.

        Returns:
            A `runs.Runs` object containing the results of all successful
            evaluations.
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
            return runs.Runs(
                [
                    run
                    for _, _, run in orchestration.evaluate_function(
                        self.run,
                        grid=grid,
                        evaluation_data=evaluation_data,
                        n_jobs=n_jobs,
                        timeout=timeout,
                    )
                    if run is not None and run.status == utils.Status.SUCCESS
                ]
            )

        all_runs = runs.Runs()
        attempt = 0
        try:
            while attempt < max_attempts:
                attempt += 1
                if max_attempts > 1:
                    logging.info(f"Attempt {attempt}/{max_attempts}.")

                try:
                    all_runs = _evaluate_once()
                except Exception as e:
                    logging.warning(
                        f"An error occurred during evaluation attempt {attempt}: {e}"
                    )
                    # Always re-raise exception because the `continue_with_exceptions``
                    # setting should only apply to the subruns level.
                    raise e

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
