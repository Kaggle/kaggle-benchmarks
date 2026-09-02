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

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from google.protobuf import json_format

from kaggle_benchmarks import clients, runs, tasks
from kaggle_benchmarks._config import ExecutionMode, config
from kaggle_benchmarks.kaggle import atif, serialization


def save_proto(message: Any, path: Path):
    with open(path, "wt") as file:
        file.write(json_format.MessageToJson(message))


def _source(run: runs.Run) -> str | None:
    """The group this trial belongs to, which is how harbor collects a dataset.

    A run with rows is the group and names itself; a row belongs to its
    parent's. Both sides must agree, so neither reads its own task name.
    """
    if run.subruns:
        return run.task.name
    return parent.task.name if (parent := run.parent) else None


class KaggleClient(clients.Client):
    def __init__(
        self,
        directory: str | Path = ".",
        format: str = "json",
        use_cache: bool = False,
    ):
        self.directory = Path(directory)
        self.registry: dict[str, tasks.Task] = {}
        self.format = format
        self.use_cache = use_cache

    def run_task(
        self, task: tasks.Task, grid: dict[str, Any], evaluation_data
    ) -> runs.Runs:
        return task.evaluate(grid=grid, evaluation_data=evaluation_data)

    def register_task(self, task: tasks.Task):
        self.registry[task.name] = task
        if task.store_task:
            self.store_task(task)

    def store_task(self, task: tasks.Task):
        self.directory.mkdir(parents=True, exist_ok=True)

        filepath = self.directory / serialization.generate_task_filename(task.name)

        try:
            proto_message = serialization.dump_task(task)
        except json_format.ParseError as e:
            logging.error(
                f"Could not parse dictionary to BenchmarkTaskVersion for {filepath}. Details: {e}"
            )
            raise e

        save_proto(proto_message, filepath)

    def store_run(self, run: runs.Run):
        self.directory.mkdir(parents=True, exist_ok=True)

        json_file_path = self._get_run_filepath(run)

        try:
            proto_message = serialization.dump_run(run)
        except json_format.ParseError as e:
            logging.error(
                f"Could not parse dictionary to BenchmarkTaskRun for {json_file_path}. Details: {e}"
            )
            raise e

        save_proto(proto_message, json_file_path)

        # After the run.json, and never allowed to fail: a converter bug must
        # not cost someone a finished run.
        atif.write_beside(
            json_format.MessageToDict(proto_message),
            json_file_path,
            subrun_paths=self._subrun_paths(run),
            source=_source(run),
        )

    def _subrun_paths(self, run: runs.Run) -> dict[str, str]:
        """Maps each row of a dataset eval to its own trajectory file.

        Only the live Run knows this: a row's filename comes from its cache_id,
        which holds the dataset's index label, and the parent's run.json records
        only the sequential pyRunId. Keyed by that id, never by position --
        subruns are appended in completion order, so under n_jobs>1 the array
        and the files disagree.

        A row whose file is gone is left out, so `remove_run_files=True` gets a
        warning rather than a ref to nothing. The parent still summarises it.
        """
        paths = {}
        for subrun in run.subruns:
            trajectory, _ = atif.paths_beside(self._get_run_filepath(subrun))
            if trajectory.exists():
                paths[f"{subrun.task.name}-{subrun.id}"] = trajectory.name
        return paths

    def skips_cached_run(self, run: runs.Run) -> bool:
        """Determines whether to skip a run by checking for the config and a cached run file."""
        if not self.use_cache:
            return False
        filepath = self._get_run_filepath(run)
        if not filepath.exists():
            return False
        with open(filepath, "r") as f:
            run_data = json.load(f)
            if run_data.get("state") != "BENCHMARK_TASK_RUN_STATE_COMPLETED":
                return False
        return True

    def _get_run_filepath(self, run: runs.Run) -> Path:
        """Gets the filepath for a given run's output file."""
        return self.directory / serialization.generate_run_filename(
            run.task.name, run.cache_id
        )

    def load_run_result(self, run: runs.Run) -> Any:
        """Loads the result from a cached run file."""
        filepath = self._get_run_filepath(run)
        with open(filepath, "r") as f:
            run_data = json.load(f)

        # Assume a single value.
        result_data = run_data["results"][0]

        match result_data:
            case {"booleanResult": value}:
                return value

            # proto3 JSON drops default-valued fields, so a numeric result
            # of 0.0 round-trips to `{}`, and (0.0, ci) to `{"confidenceInterval": ci}`.
            # Extract by key with defaults instead of matching by shape.
            # Known limitation: a tuple `(v, 0.0)` collapses to scalar `v`
            # because zero CI is indistinguishable from "no CI" on disk.
            case {"numericResult": nr}:
                value = nr.get("value", 0.0)
                if "confidenceInterval" in nr:
                    return (value, nr["confidenceInterval"])
                return value

            case {"dictResult": value}:
                return value

            case _:
                raise ValueError(
                    f"Could not determine result from cached run file: {filepath}. "
                    "No known result key found in 'results' field."
                )

    @classmethod
    def can_initialize(cls) -> bool:
        return config.execution_mode not in (ExecutionMode.DOC, ExecutionMode.TESTING)

    @contextlib.contextmanager
    def enable_cache(self):
        """
        Context manager to temporarily enable the cache state of the kbench.client.

        Resets to use_cache=False upon exit, matching the original code's logic.
        """
        try:
            self.use_cache = True
            yield
        finally:
            self.use_cache = False
