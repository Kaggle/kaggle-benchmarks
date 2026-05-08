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

import inspect
from typing import Any, Callable

import pydantic
from google.genai import types


class ToolSchemaError(Exception):
    """Raised when a function schema cannot be generated."""


def get_function_schema(func: Callable) -> dict:
    """Generates a JSON schema for a function's parameters using Pydantic."""
    sig = inspect.signature(func)
    fields = {}

    for name, param in sig.parameters.items():
        annotation = (
            param.annotation if param.annotation != inspect.Parameter.empty else Any
        )
        default = param.default if param.default != inspect.Parameter.empty else ...

        fields[name] = (annotation, default)

    try:
        DynamicModel = pydantic.create_model(f"{func.__name__}", **fields)
        return DynamicModel.model_json_schema()
    except pydantic.PydanticSchemaGenerationError as e:
        raise ToolSchemaError(
            "Unable to generate json schema for function {func.__name__} arguments", e
        )


def function_to_openai_tool(func: Callable) -> dict:
    """Converts a Python function into an OpenAI-compatible tool definition."""
    schema = get_function_schema(func)

    # Omit 'parameters' entirely for parameterless functions.
    # GPT-5.5 rejects empty properties objects (`{"properties": {}}`)
    # with 400: "object schema missing properties".
    properties = schema.get("properties", {})
    if properties:
        parameters: dict = {
            "type": "object",
            "properties": properties,
            "required": schema.get("required", []),
        }
        if "$defs" in schema:
            parameters["$defs"] = schema["$defs"]
    else:
        parameters = None

    tool_def = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
        },
    }
    if parameters:
        tool_def["function"]["parameters"] = parameters
    return tool_def


def function_to_genai_tool(
    tool: Callable,
) -> types.FunctionDeclaration:
    """Converts a Python function to a Google GenAI FunctionDeclaration."""
    return types.FunctionDeclaration.from_callable_with_api_option(callable=tool)
