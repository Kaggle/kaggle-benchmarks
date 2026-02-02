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
from typing import Any, Callable, Union

from google.genai import types
from pydantic import create_model


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

    DynamicModel = create_model(f"{func.__name__}", **fields)

    return DynamicModel.model_json_schema()


def function_to_openai_tool(func: Callable) -> dict:
    """Converts a Python function into an OpenAI-compatible tool definition."""
    schema = get_function_schema(func)

    parameters = {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
    }

    return {
        "type": "function",
        "name": func.__name__,
        "description": (func.__doc__ or "").strip(),
        "parameters": parameters,
    }


def function_to_genai_tool(
    tool: Union[Callable, dict],
) -> types.FunctionDeclaration:
    """Converts a Python function or an OpenAI-style tool dictionary into a Google GenAI FunctionDeclaration."""
    if isinstance(tool, Callable):
        return types.FunctionDeclaration(
            name=tool.__name__,
            description=tool.__doc__,
            parameters=get_function_schema(tool),
        )

    elif isinstance(tool, dict):
        # map from openai style
        return types.FunctionDeclaration(
            name=tool.get("name"),
            description=tool.get("description"),
            parameters=tool.get("parameters"),
        )
    else:
        raise ValueError("Unknown tool type")
