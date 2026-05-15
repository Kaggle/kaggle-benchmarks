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

from typing import Callable, Literal, TypeVar, Union

import pydantic
from typing_extensions import TypedDict

from kaggle_benchmarks import actors, chats, usage
from kaggle_benchmarks.llm_messages import LLMMessage
from kaggle_benchmarks.tools import base, functions

T = TypeVar("T")


class ToolInvocationLimitExhausted(Exception):
    pass


def build_response_model(tools: list[Callable], output_schema: type):
    """Creates a pydantic model that can be used as response format for LLM that provides option for llm to invoke tools."""
    return base.ModelResponse[
        output_schema,
        Union[
            *(
                base.ToolCallModel[
                    Literal[tool.__name__],
                    TypedDict(
                        tool.__name__,
                        {
                            field: annotation
                            for field, (
                                annotation,
                                _,
                            ) in functions._get_function_arguments(tool).items()
                        },
                    ),
                ]
                for tool in tools
            )
        ],
    ]


def simulate_respond_with_tools(
    llm: actors.LLMChat,
    tools: list[Callable],
    output_schema: type[T],
    system: str | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> LLMMessage[T]:
    """Simulates tool calling for models that do not support it natively."""
    if not tools:
        return llm.respond(
            system=system,
            schema=output_schema,
            temperature=temperature,
            seed=seed,
            tools=None,
        )

    chat = chats.get_current_chat()

    previous_invocations = list(base.iter_invocations(chat))

    if previous_invocations:
        invocation_history = "\n".join(i.describe() for i in previous_invocations)

        history_prompt = f"""You have already invocated the following tools:
{invocation_history}"""
    else:
        history_prompt = ""

    instructions = f"""You can invoke the following tools:
{base.describe_tools(tools)}

{history_prompt}

If you decided to invoke a tool, fill the `tools` attribute with the invocation details, like `[{{"name": "function_name", "arguments": {{...}}}}]`.
If you have enough information from previous tool calls or you decide not to use any tools, leave the `tools` field blank and fill the `message` field with your response.
Only one of `tools` or `message`, should be filled with a value.
"""

    with chats.fork(orphan=True) as subchat:
        actors.user.send(instructions)

        try:
            wrapped_schema = build_response_model(tools, output_schema)
        except pydantic.PydanticSchemaGenerationError as e:
            raise ValueError(
                f"Unable to generate JSON schema for response format {output_schema}."
            ) from e

        response = llm.respond(
            system=system,
            schema=wrapped_schema,
            temperature=temperature,
            seed=seed,
            tools=None,
        )

    value = response.content
    if value.tools:
        response.tool_calls = [
            base.invoke_tool(
                base.ToolInvocation(
                    name=call.name,
                    arguments=call.arguments,
                    call_id=f"call_{call.name}",
                ),
                tools,
            )
            for call in value.tools
        ]
        response.content = value.message
    elif value.message is not None:
        response.content = value.message
    else:
        # some models will not produce anything
        # so we ask them once more without tools
        response = llm.respond(
            system=system,
            schema=output_schema,
            temperature=temperature,
            seed=seed,
            tools=None,
        )
    response.chat = subchat
    return response


def simulate_agent(
    llm: actors.LLMChat,
    tools: list[Callable],
    output_schema: type[T] = str,
    max_iterations: int = 10,
    system: str | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> LLMMessage[T | None]:
    """Simulates an agent using tools over multiple iterations."""
    final_response = LLMMessage(
        sender=llm, content=None, usage=usage.Usage(0, 0), tool_calls=[]
    )

    for _ in range(max_iterations):
        response = simulate_respond_with_tools(
            llm,
            tools,
            output_schema=output_schema,
            system=system,
            temperature=temperature,
            seed=seed,
        )

        final_response.content = response.content
        final_response.usage += response.usage

        if tools and response.tool_calls:
            for call in response.tool_calls:
                result = base.invoke_tool(call=call, tools=tools)
                final_response.tool_calls.append(result)
                actors.Tool(name=call.name).send(result)
        else:
            break
    else:
        raise ToolInvocationLimitExhausted()

    return final_response
