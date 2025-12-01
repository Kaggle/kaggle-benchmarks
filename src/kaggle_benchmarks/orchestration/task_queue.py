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

import asyncio
import dataclasses
import inspect
from typing import Any, Callable, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")

pending = object()


@dataclasses.dataclass
class Task:
    id: str
    func: Callable[..., Any]
    params: dict[str, Any]
    result: Any = pending

    async def run(self):
        if inspect.iscoroutinefunction(self.func):
            self.result = await self.func(**self.params)

        else:
            self.result = await asyncio.to_thread(self.func, **self.params)

        return self.result


async def worker(queue: asyncio.Queue[Task], pbar: tqdm):
    while True:
        task = await queue.get()

        await task.run()
        queue.task_done()

        pbar.update(1)


async def arun_tasks(tasks: list[Task], n_jobs: int = 4):
    queue: asyncio.Queue[Task] = asyncio.Queue()

    for task in tasks:
        queue.put_nowait(task)

    with tqdm(total=len(tasks)) as pbar:
        workers = [asyncio.create_task(worker(queue, pbar)) for _ in range(n_jobs)]

        await queue.join()

        for w in workers:
            w.cancel()

    await asyncio.gather(*workers, return_exceptions=True)


def run_tasks(tasks: list[Task], n_jobs: int = 1, debug: bool = False):
    if n_jobs > 1:
        asyncio.run(arun_tasks(tasks, n_jobs=n_jobs), debug=debug)
        return {task.id: task.result for task in tasks}

    else:
        return {task.id: task.func(**task.params) for task in tqdm(tasks)}
