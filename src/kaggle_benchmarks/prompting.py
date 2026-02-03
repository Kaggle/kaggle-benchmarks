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

import datetime
import inspect
import json
from typing import Generator, Generic, TypeVar, overload

import pydantic

from kaggle_benchmarks import utils

handlers = []
T = TypeVar("T")


class BaseModel(pydantic.BaseModel):
    """Base pydantic model that renderable in markdown."""

    def _repr_markdown_(self):
        return "\n".join(
            f"## {key.title().replace('_', '')}\n{value}"
            for key, value in self.model_dump().items()
        )


class TypedResponse(BaseModel, Generic[T]):
    """
        A generic container for wrapping a typed value.

        This is particularly useful for APIs that require a JSON object as the
        root of a response schema. It allows for defining a response format
    for
        primitive or generic types on the fly.

        For example:
        - `TypedResponse[int]` will expect a JSON object like `{"value": 123}`.
        - `TypedResponse[list[int]]` will expect `{"value": [1, 2, 3]}`.
        - `TypedResponse[tuple[int, str]]` will expect `{"value": [1, "hello"]}`.
    """

    value: T


class ResponseParsingError(ValueError):
    def __init__(self, value=None, message=None, schema=None, *args: object) -> None:
        self.value = value
        self.message = message
        self.schema = schema
        super().__init__(*args)

    def __str__(self) -> str:
        return f"ResponseParsingError(value={self.value}, message={self.message}, schema={self.schema})"


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


@handler(criterion=lambda x: isinstance(x, dict))
def typed_dict(attrs: dict[str, type]):
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

    try:
        return model.model_validate_json(
            utils.extract_code_block(response, name="json")
        )
    except pydantic.ValidationError as e:
        raise ResponseParsingError(response, str(e), model) from e


def can_be_root_model(cls):
    try:
        _ = pydantic.RootModel[cls]
        _.model_json_schema()
        return True
    except Exception:
        return False


@handler(criterion=can_be_root_model)
def root_model_handler(cls):
    model_cls = pydantic.RootModel[cls]
    response = yield (
        f"Output JSON using this schema: {json.dumps(model_cls.model_json_schema())}",
        model_cls,
    )

    try:
        validated_model = model_cls.model_validate_json(
            utils.extract_code_block(response, name="json")
        )
        return validated_model.root
    except pydantic.ValidationError as e:
        raise ResponseParsingError(response, str(e), model_cls) from e


@handler(types=(float, int, datetime.datetime, bool))
def primitive_type_handler(cls):
    model = TypedResponse[cls]
    response = yield (
        f"Output JSON using this schema: {json.dumps(model.model_json_schema())}",
        model,
    )
    try:
        return model.model_validate_json(
            utils.extract_code_block(response, name="json")
        ).value
    except pydantic.ValidationError as e:
        raise ResponseParsingError(response, str(e), model) from e


@handler(types=str)
def string(_):
    value = yield None
    return value


@overload
def process_schema(cls: type[T]) -> Generator[str, str, T]: ...


@overload
def process_schema(cls: dict[str, type]) -> Generator[str, str, pydantic.BaseModel]: ...


def process_schema(cls):
    for criterion, handler in reversed(handlers):
        if criterion(cls):
            return handler(cls)

    raise ValueError(f"Unsupported type: {cls}")
