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

Supported types currently include several built-in types, Pydantic models, and dataclasses.
"""

import dataclasses
import datetime
import inspect
import json
from typing import Generator, Generic, TypeVar, overload

import pydantic

from kaggle_benchmarks import utils

handlers = []
T = TypeVar("T")


class RenderablePydanticModel(pydantic.BaseModel):
    """Base pydantic model that renderable in markdown."""

    def _repr_markdown_(self):
        return "\n".join(
            f"## {key.title().replace('_', '')}\n{value}"
            for key, value in self.model_dump().items()
        )

    def get_payload(self):
        return self.model_dump_json()


# A generic shortcut for wrapping a typed response.
# For example:
# - `TypedResponse[int]` will expect a JSON object like `{"value": 123}`.
# - `TypedResponse[list[int]]` will expect `{"value": [1, 2, 3]}`.
# - `TypedResponse[tuple[int, str]]` will expect `{"value": [1, "hello"]}`.
class TypedResponse(RenderablePydanticModel, Generic[T]):
    value: T

    model_config = pydantic.ConfigDict(
        title="Response",
        extra="forbid",
        arbitrary_types_allowed=False,
    )

    __name__ = "Response"


class ResponseParsingError(ValueError):
    """Error raised when a model response cannot be parsed into the desired schema."""

    def __init__(self, *args: object, value, error, schema) -> None:
        self.value = value
        self.error = error
        self.schema = schema
        super().__init__("Failed to parse model response.", *args)

    def __str__(self) -> str:
        if issubclass(self.schema, pydantic.BaseModel):
            schema_str = json.dumps(self.schema.model_json_schema(), indent=2)
        else:
            schema_str = str(self.schema)

        return (
            f"Response parsing failed.\n"
            f"Input Value:\n---\n{self.value}\n---\n"
            f"Target Schema:\n---\n{schema_str}\n---\n"
            f"Parsing Error:\n---\n{self.error}\n---"
        )


class SchemaError(TypeError):
    """Covers errors related to wrong schema or handler."""


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
            __base__=RenderablePydanticModel,
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
        raise ResponseParsingError(
            value=response, error=e.errors(), schema=model
        ) from e


@handler(criterion=dataclasses.is_dataclass)
def root_model_handler(cls):
    schema = pydantic.TypeAdapter(cls).json_schema()
    model_cls = pydantic.create_model(
        cls.__name__,
        __base__=(RenderablePydanticModel, cls),
        **{field.name: (field.type, ...) for field in dataclasses.fields(cls)},
    )
    response = yield (
        f"Output JSON using this schema: {json.dumps(schema)}",
        model_cls,
    )

    try:
        value = json.loads(utils.extract_code_block(response, name="json"))
        return cls(**value)
    except json.JSONDecodeError as e:
        raise ResponseParsingError(
            value=response, error=f"Invalid JSON {e}", schema=model_cls
        ) from e

    except pydantic.ValidationError as e:
        raise ResponseParsingError(
            value=response, error=e.errors(), schema=model_cls
        ) from e


@handler(types=(float, int, datetime.datetime, bool))
def primitive_type_handler(cls):
    model = TypedResponse[cls]
    model.__name__ = "Response"
    response = yield (
        f"Output JSON using this schema: {json.dumps(model.model_json_schema())}",
        model,
    )
    try:
        return model.model_validate_json(
            utils.extract_code_block(response, name="json")
        ).value
    except pydantic.ValidationError as e:
        raise ResponseParsingError(
            value=response, error=e.errors(), schema=model
        ) from e


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
