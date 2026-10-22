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

"""Runs of one task, grouped by named sets of rows.

Each split is a regular `Runs`. Hidden splits are left out of the default
display and of `as_dataframe()`; `splits[name]` returns them in full.

`hidden` only changes how the container displays. Output while a split runs
is hidden only by `Task.evaluate_splits`.
"""

import dataclasses
from typing import TYPE_CHECKING, Any, Iterator

import pandas as pd

if TYPE_CHECKING:
    from kaggle_benchmarks import runs


@dataclasses.dataclass
class Splits:
    splits: dict[str, "runs.Runs[Any]"] = dataclasses.field(default_factory=dict)
    hidden: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if unknown := sorted(set(self.hidden) - set(self.splits)):
            raise ValueError(
                f"Unknown hidden splits: {unknown}. Available: {sorted(self.splits)}."
            )
        self.hidden = frozenset(self.hidden)

    def __getitem__(self, name: str) -> "runs.Runs[Any]":
        return self.splits[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.splits)

    def __len__(self) -> int:
        return len(self.splits)

    def is_hidden(self, name: str) -> bool:
        return name in self.hidden

    def as_dataframe(self, include_hidden: bool = False) -> pd.DataFrame:
        """One row per run, with a `split` column.

        Hidden splits are excluded unless `include_hidden=True`.
        """
        frames = []
        for name, runs_ in self.splits.items():
            if not len(runs_) or (self.is_hidden(name) and not include_hidden):
                continue
            frame = runs_.as_dataframe()
            if "split" in frame.columns:
                raise ValueError(
                    "The runs already have a `split` column, from a task "
                    "parameter of that name. Use `splits[name].as_dataframe()` "
                    "for each split instead."
                )
            frames.append(frame.assign(split=name))
        if not frames:
            return pd.DataFrame().rename_axis("run_id")
        return pd.concat(frames)

    def __repr__(self) -> str:
        # The generated dataclass repr would print every hidden row.
        parts = [
            f"{name}={len(runs_)} runs" + (" (hidden)" if self.is_hidden(name) else "")
            for name, runs_ in self.splits.items()
        ]
        return f"{type(self).__name__}({', '.join(parts)})"

    def __panel__(self):
        from kaggle_benchmarks.ui import panel

        return panel.render_splits(self)

    def _repr_mimebundle_(self, include=None, exclude=None):
        return self.__panel__()._repr_mimebundle_(include, exclude)
