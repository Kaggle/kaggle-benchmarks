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

import pytest

from kaggle_benchmarks import actors, llm_messages, tools, utils


def test_llm_message_creation():
    msg = llm_messages.LLMMessage(content="Hello", sender=actors.LLMChat())
    assert msg.content == "Hello"
    assert msg.status == utils.Status.RUNNING


def test_usage_add():
    usage1 = llm_messages.Usage(input_tokens=10, output_tokens=20)
    usage2 = llm_messages.Usage(input_tokens=5, output_tokens=15)
    usage3 = usage1 + usage2
    assert usage3.input_tokens == 15
    assert usage3.output_tokens == 35


def test_usage_add_with_none():
    usage1 = llm_messages.Usage(input_tokens=10, output_tokens=20)
    usage3 = usage1 + None
    assert usage3.input_tokens == 10
    assert usage3.output_tokens == 20


@pytest.mark.parametrize(
    "chunks, kwargs, expected_content, expected_tool_calls, expected_usage",
    [
        (["Hello", ", ", "world!"], {}, "Hello, world!", [], None),
        (
            [tools.ToolInvocation(name="test_tool", arguments={"arg1": "val1"})],
            {},
            "",
            [tools.ToolInvocation(name="test_tool", arguments={"arg1": "val1"})],
            None,
        ),
        (
            [
                tools.ToolInvocationResult(
                    name="test_tool", arguments={}, output="Success"
                )
            ],
            {},
            "",
            [
                tools.ToolInvocationResult(
                    name="test_tool", arguments={}, output="Success"
                )
            ],
            None,
        ),
        (
            [
                "Hello",
                tools.ToolInvocation(name="test_tool", arguments={"arg1": "val1"}),
                ", ",
                "world!",
                tools.ToolInvocationResult(
                    name="test_tool", arguments={}, output="Success"
                ),
            ],
            {},
            "Hello, world!",
            [
                tools.ToolInvocation(name="test_tool", arguments={"arg1": "val1"}),
                tools.ToolInvocationResult(
                    name="test_tool", arguments={}, output="Success"
                ),
            ],
            None,
        ),
        ([], {}, "", [], None),
        (
            [
                "Hello",
                llm_messages.Usage(input_tokens=1, output_tokens=2),
                ", world!",
                llm_messages.Usage(input_tokens=3, output_tokens=4),
            ],
            {},
            "Hello, world!",
            [],
            llm_messages.Usage(input_tokens=4, output_tokens=6),
        ),
        (
            ["Hello", llm_messages.Usage(input_tokens=1, output_tokens=2)],
            {"usage": llm_messages.Usage(input_tokens=10, output_tokens=20)},
            "Hello",
            [],
            llm_messages.Usage(input_tokens=11, output_tokens=22),
        ),
        (
            [
                tools.ToolInvocation(name="test_tool", arguments={}, call_id="123"),
                tools.ToolInvocationResult(
                    name="test_tool", arguments={}, output="Success", call_id="123"
                ),
            ],
            {},
            "",
            [
                tools.ToolInvocationResult(
                    name="test_tool", arguments={}, output="Success", call_id="123"
                )
            ],
            None,
        ),
    ],
    ids=[
        "strings",
        "tool_invocations",
        "tool_invocation_results",
        "mixed_content",
        "no_chunks",
        "with_usage",
        "with_initial_usage",
        "tool_invocation_with_result",
    ],
)
def test_from_chunks(
    chunks, kwargs, expected_content, expected_tool_calls, expected_usage
):
    msg = llm_messages.LLMMessage.from_chunks(chunks, sender=actors.LLMChat(), **kwargs)

    assert msg.content == expected_content
    assert msg.tool_calls == expected_tool_calls
    assert msg.usage == expected_usage
    assert msg.status == utils.Status.SUCCESS
