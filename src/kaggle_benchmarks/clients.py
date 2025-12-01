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
from typing import Any, Self

from kaggle_benchmarks import runs, tasks


class Client:
    _subclasses: list[type[Self]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._subclasses.append(cls)

    def register_task(self, task: tasks.Task): ...

    def store_task(self, task: tasks.Task): ...

    def store_run(self, run: runs.Run): ...

    def skips_cached_run(self, run: runs.Run) -> bool:
        """Determines whether to skip a run because its results are already cached."""
        ...

    def load_run_result(self, run: runs.Run) -> Any:
        """Loads the result of a cached run."""

    @classmethod
    def can_initialize(cls) -> bool:
        """Subclasses should override this method to indicate if they can run in the current environment."""
        return False

    @contextlib.contextmanager
    def enable_cache(self):
        """Context manager to temporarily enable caching. Base implementation is a no-op that yields."""
        yield


class InMemoryClient(Client):
    def __init__(self):
        self.registry = {}

    def register_task(self, task: tasks.Task):
        self.registry[task.name] = task

    def store_task(self, task: tasks.Task): ...

    def store_run(self, run: runs.Run): ...

    def skips_cached_run(self, run: runs.Run) -> bool:
        """In-memory client does not support caching."""
        return False

    def load_run_result(self, run: runs.Run) -> Any:
        """In-memory client does not support loading from cache."""
        return None

    @classmethod
    def can_initialize(cls) -> bool:
        return True


def resolve_client():
    # todo: find a more convenient way
    for cls in reversed(Client._subclasses):
        if cls.can_initialize():
            return cls()
    return InMemoryClient()
