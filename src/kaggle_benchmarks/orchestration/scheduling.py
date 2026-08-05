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

import itertools
from typing import Any, Iterable, Iterator, TypeVar

import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from kaggle_benchmarks import contexts
from kaggle_benchmarks._config import config
from kaggle_benchmarks.orchestration import task_queue

T = TypeVar("T")


def sample_combinations(grid: dict[str, Iterable[T]]) -> Iterator[dict[str, T]]:
    """Samples all possible combinations of items from a dictionary of key to list mappings."""

    combinations = itertools.product(*grid.values())
    return (dict(zip(grid, combination)) for combination in combinations)


def prepare_tasks(
    func,
    grid: dict[str, Iterable[Any]],
    evaluation_data: pd.DataFrame | None = None,
    id_format: str = "{id}",
) -> Iterable[task_queue.Task]:
    if evaluation_data is None:
        evaluation_data = pd.DataFrame([()])

    for params in sample_combinations(grid=grid):
        for id, row in evaluation_data.iterrows():
            kwargs = {**params, **row}
            yield task_queue.Task(
                id=id_format.format(id=id, **kwargs),
                params=kwargs,
                func=func,
            )


def evaluate_function(
    func,
    grid: dict[str, Iterable[Any]],
    evaluation_data: pd.DataFrame | None = None,
    tqdm=tqdm,
    n_jobs: int = 1,
    timeout: float | None = None,
):
    parameters = list(sample_combinations(grid=grid))

    for params in tqdm(
        parameters,
        desc="Parameters",
        colour="#777777",
        disable=config.disable_tqdm() or len(parameters) == 1,
        leave=False,
    ):
        if evaluation_data is None:
            yield None, params, func(**params)

        else:
            main_context = contexts.get_current()

            def run_on_row(id_and_row, context):
                id, row = id_and_row
                call_params = {**params, **row.to_dict()}
                contexts.set_from_context(context)
                result = func(**call_params)
                return id, call_params, result

            with Parallel(
                n_jobs=n_jobs,
                backend="threading",
                verbose=7 if not config.disable_tqdm() else 0,
                timeout=timeout,
            ) as parallel:
                tasks = (
                    delayed(run_on_row)(item, main_context)
                    for item in evaluation_data.iterrows()
                )

                results_iterator = parallel(tasks)

                yield from tqdm(
                    results_iterator,
                    total=len(evaluation_data),
                    desc="Running subtasks",
                    colour="#888888",
                    disable=config.disable_tqdm() or len(evaluation_data) == 1,
                    leave=False,
                )
