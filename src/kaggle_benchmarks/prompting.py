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

"""
Module for defining handlers and building prompts for structured data extraction.

This module provides a generic and customizable way to handle structured output from language models.
It allows defining handlers that process responses and convert them into specific data types.
The `PromptBuilder` class uses these handlers to generate prompts based on a given schema.

Default handlers are provided for common types (str, bool, dataclass),
but these can be overwritten or extended with custom handlers to support more complex
data structures or specific formatting requirements.

Handlers are processed in reverse registration order, so later registered handlers can take precedence.

Supported types currently include several build-in types, Pydantic models, and dataclasses.
"""

import dataclasses
import datetime
import inspect
import json
from typing import Generator, TypeVar, overload

import pydantic

from kaggle_benchmarks import utils

handlers = []
T = TypeVar("T")


def parse_response(handler: Generator[str | tuple[str, T], str, T], value: str) -> T:
    next(handler)
    try:
        handler.send(value)
        raise ValueError("Handler hasn't returned a value.")
    except StopIteration as e:
        return e.value


def handler(criterion=None, types=None):
    if criterion is None and types is None:
        raise ValueError("Either criterion or types must be specified.")

    def checker(x):
        if criterion is not None:
            return criterion(x)
        elif types is not None:
            if inspect.isclass(x):
                return issubclass(x, types)
            return False

    def decorator(func):
        handlers.append((checker, func))
        return func

    return decorator


@handler(types=str)
def string(_):
    value = yield None
    return value


@handler(types=float)
def float_handler(_):
    value = yield "Provide a float."
    try:
        return float(utils.extract_code_block(value))
    except ValueError:
        raise ValueError("Invalid float provided.")


@handler(types=int)
def integer(_):
    value = yield "Provide an integer."
    try:
        return int(utils.extract_code_block(value))
    except ValueError:
        raise ValueError("Invalid integer provided.")


@handler(types=datetime.datetime)
def datetime_handler(_):
    value = yield "Provide a datetime in ISO 8601 format (e.g., 2024-10-27T10:00:00Z)."
    try:
        return datetime.datetime.fromisoformat(
            utils.extract_code_block(value).replace("Z", "+00:00")
        )
    except ValueError:
        raise ValueError("Invalid datetime format.")


@handler(types=bool)
def boolean(_):
    value = yield "Start your answer with `True.` or `False.`."
    value = value.lower()
    assert ("true" in value) + ("false" in value) == 1, (
        f"Boolean value of {value} is unclear."
    )
    return "true" in value.lower()


@handler(criterion=dataclasses.is_dataclass)
def dataclass(cls):
    fields = "\n".join(
        f"{field.name}: {field.type.__name__}" for field in dataclasses.fields(cls)
    )

    value = yield f"Write a JSON with the following keys: {fields}"
    value = utils.extract_code_block(value, name="json", greedy=False)
    try:
        return cls(**json.loads(value))
    except json.JSONDecodeError:
        raise AssertionError(f"Invalid JSON `{value}`")


@handler(criterion=lambda x: isinstance(x, dict))
def typed_dict(attrs: dict[str, type]):
    class BaseModel(pydantic.BaseModel):
        def _repr_markdown_(self):
            return "\n".join(
                f"## {key.title().replace('_', '')}\n{value}"
                for key, value in self.model_dump().items()
            )

    return pyndantic_like(
        pydantic.create_model(
            "Response",
            **{key: (value, ...) for key, value in attrs.items()},
            __base__=BaseModel,
        )
    )


@handler(types=pydantic.BaseModel)
def pyndantic_like(model: pydantic.BaseModel):
    response = yield (
        f"Output JSON using this schema: {json.dumps(model.model_json_schema())}",
        model,
    )

    return model.model_validate_json(utils.extract_code_block(response, name="json"))


@overload
def process_schema(cls: type[T]) -> Generator[str, str, T]: ...


@overload
def process_schema(cls: dict[str, type]) -> Generator[str, str, pydantic.BaseModel]: ...


def process_schema(cls):
    for criterion, handler in reversed(handlers):
        if criterion(cls):
            return handler(cls)

    raise ValueError(f"Unsupported type: {cls}")
