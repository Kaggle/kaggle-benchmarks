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
import json
import math
import numbers
import pprint
from typing import TYPE_CHECKING, Any, Generic, TypeVar, get_args

import numpy as np
from google.protobuf import struct_pb2

T = TypeVar("T")

if TYPE_CHECKING:
    from kaggle_benchmarks import runs

    T = TypeVar("T", bound="runs.Run")

types = {}


class Result(Generic[T]):
    _type: type[T] = None  # type: ignore

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._type = get_args(cls.__orig_bases__[0])[0]
        types[cls._type] = cls

    @classmethod
    def format(cls, run: T) -> str:
        return f"{run.result!s:>10}"

    @classmethod
    def passed(cls, value: T) -> bool:
        return True

    @classmethod
    def check_value(cls, value: Any) -> bool:
        if hasattr(cls._type, "__origin__"):
            # less robust check for generic types (like list[int])
            return isinstance(value, cls._type.__origin__)
        return isinstance(value, cls._type)


class Unknown(Result[object]):
    """Special type for a result that has not yet been computed."""

    def __bool__(self):
        return False


PENDING = Unknown()
FAILED = Unknown()


class PassFail(Result[type(None) | Unknown]):
    @classmethod
    def format(cls, run: "runs.Run[None]") -> str:
        if run.assertion_results:
            return f"{sum(result.passed for result in run.assertion_results)} out of {len(run.assertion_results)} assertions passed."
        return "✅" if run.result is None else "❌"

    @classmethod
    def passed(cls, value: None | Unknown) -> bool:
        return value is None

    @classmethod
    def check_value(cls, value: Any) -> bool:
        return value is None


class Boolean(Result[bool]):
    @classmethod
    def format(cls, run: "runs.Run[bool]") -> str:
        return "✅" if run.result else "❌"

    @classmethod
    def passed(cls, value: bool) -> bool:
        return value


class Numerical(Result[int]): ...


class Score(Result[float]): ...


class PassCount(Result[tuple[int, int]]):
    """Note the backend doesn't support PassCount yet."""

    @classmethod
    def format(cls, run) -> str:
        return f"{run.result[0]} out of {run.result[1]}"


class MetricWithCI(Result[tuple[float, float]]):
    """Metric with its confidence interval."""

    @classmethod
    def format(cls, run) -> str:
        return f"{run.result[0]}(±{run.result[1]})"


SplitValue = float | bool | tuple[float, float] | dict[str, Any]


def split_value(value: Any, name: str) -> SplitValue:
    """Returns `value` as a bool, a float, a pair of floats or a dict, the forms
    a run file holds, or raises `TypeError` if it cannot hold it."""

    def is_number(item: Any) -> bool:
        return (
            isinstance(item, numbers.Real)
            and not isinstance(item, (bool, np.bool_))
            and math.isfinite(item)
        )

    try:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if is_number(value):
            return float(value)
        if isinstance(value, tuple) and len(value) == 2 and all(map(is_number, value)):
            return (float(value[0]), float(value[1]))
        if isinstance(value, dict):
            # As the run file is written: a Struct, then JSON without NaN.
            struct_pb2.Struct().update(value)
            json.dumps(value, allow_nan=False)
            return value
    except (TypeError, ValueError, OverflowError):
        pass
    raise TypeError(
        f"The {name} score must be a bool, a finite number, a (score, confidence) "
        f"pair or a dict of JSON values, not {value!r:.50}."
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class SplitScores:
    """Scores of a task over a public and a private split of its rows.

    Each is a bool, a finite number, a `(score, confidence)` pair or a dict of
    JSON values, and all three are published. The task is ranked on `overall`.
    """

    overall: SplitValue
    public: SplitValue
    private: SplitValue

    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = split_value(getattr(self, field.name), field.name)
            object.__setattr__(self, field.name, value)


class Splits(Result[SplitScores]): ...


class Dictionary(Result[dict]):
    @classmethod
    def format(cls, run: "runs.Run[dict]") -> str:
        formatted_dict = pprint.pformat(run.result)

        if len(formatted_dict) > 1000:
            # If it's too long, default to rendering all keys.
            keys_list = list(run.result.keys())
            return f"Dict with {len(keys_list)} keys: {keys_list}"
        else:
            return formatted_dict
