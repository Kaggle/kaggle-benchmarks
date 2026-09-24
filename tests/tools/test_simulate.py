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


import pydantic
import pytest

from kaggle_benchmarks import chats
from kaggle_benchmarks.tools import simulate
from tests.mocks import MockedChat


def dummy_tool(x: int, y: str = "default") -> str:
    """A dummy tool."""
    return f"{x}-{y}"


def dummy_tool_2(z: float) -> float:
    return z


def tool_no_arguments():
    pass


def error_tool() -> str:
    raise ValueError("Simulated tool failure")


def test_build_response_model():
    """Tests the creation of a Pydantic response model for tool invocation."""
    model = simulate.build_response_model(
        [dummy_tool, dummy_tool_2, tool_no_arguments], str
    )
    assert issubclass(model, pydantic.BaseModel)


@pytest.mark.parametrize(
    "tools_payload",
    [
        [{"name": "dummy_tool", "arguments": {"x": 1, "y": "test"}}],
        [{"name": "dummy_tool_2", "arguments": {"z": 3.14}}],
        [
            {
                "name": "dummy_tool",
                "arguments": {"x": 1, "y": "test", "extra_arg": "ignored"},
            }
        ],
    ],
)
def test_build_response_model_valid(tools_payload):
    """Tests the creation of a Pydantic response model for tool invocation."""
    model = simulate.build_response_model([dummy_tool, dummy_tool_2], str)
    assert issubclass(model, pydantic.BaseModel)

    instance = model(tools=tools_payload, message=None)
    assert instance.tools is not None
    assert len(instance.tools) == 1
    assert instance.tools[0].name == tools_payload[0]["name"]
    for k, v in instance.tools[0].arguments.items():
        assert v == tools_payload[0]["arguments"][k]


@pytest.mark.parametrize(
    "tools_payload,expected_error",
    [
        (
            [{"name": "non_existent_tool", "arguments": {"x": 1, "y": "test"}}],
            "Input should be",
        ),
        (
            [{"name": "dummy_tool", "arguments": {"x": "not-an-int", "y": "test"}}],
            "Input should be a valid integer",
        ),
        (
            [{"name": "dummy_tool", "arguments": {}}],
            "Field required",
        ),
    ],
)
def test_build_response_model_invalid(tools_payload, expected_error):
    """Tests that the response model rejects invalid tool calls."""
    model = simulate.build_response_model([dummy_tool, dummy_tool_2], str)

    with pytest.raises(pydantic.ValidationError) as exc_info:
        model(tools=tools_payload, message=None)

    assert expected_error in str(exc_info.value)


def test_simulate_tool_calling_with_tools():
    llm = MockedChat.from_contents_data(
        [
            dict(
                tools=[
                    {
                        "name": "dummy_tool",
                        "arguments": {"x": 42, "y": "default"},
                    }
                ],
                message=None,
            )
        ]
    )

    response = simulate.simulate_respond_with_tools(
        llm=llm, tools=[dummy_tool, dummy_tool_2], output_schema=str
    )

    tool_calls = response.tool_calls or []
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "dummy_tool"
    assert tool_calls[0].arguments == {"x": 42, "y": "default"}
    assert response.content is None


def test_simulate_tool_calling_with_message():
    llm = MockedChat.from_contents_data(
        [{"tools": None, "message": "Here is the final answer."}]
    )

    response = simulate.simulate_respond_with_tools(
        llm=llm, tools=[dummy_tool, dummy_tool_2], output_schema=str
    )
    tool_calls = response.tool_calls or []
    assert not tool_calls
    assert response.content == "Here is the final answer."


def test_simulate_agent_success():
    llm = MockedChat.from_contents_data(
        [
            {
                "tools": [
                    {
                        "name": "dummy_tool",
                        "arguments": {"x": 10, "y": "test"},
                    }
                ],
                "message": None,
            },
            {"tools": None, "message": "Done!"},
        ]
    )

    response = simulate.simulate_agent(
        llm=llm, tools=[dummy_tool, dummy_tool_2], output_schema=str
    )

    assert response.content == "Done!"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "dummy_tool"
    assert response.tool_calls[0].output == "10-test"


def test_simulate_agent_limit_exhausted():
    llm = MockedChat.from_contents_data(
        [
            {
                "tools": [
                    {
                        "name": "dummy_tool",
                        "arguments": {"x": i, "y": "test"},
                    }
                ],
                "message": None,
            }
            for i in range(5)
        ]
    )

    with pytest.raises(simulate.ToolInvocationLimitExhausted):
        simulate.simulate_agent(
            llm=llm, tools=[dummy_tool], output_schema=str, max_iterations=2
        )


def test_simulate_agent_without_tools():
    llm = MockedChat.from_contents(["No tools used!"])
    with chats.new("test_chat"):
        response = simulate.simulate_agent(llm=llm, tools=[], output_schema=str)
        assert response.content == "No tools used!"
        assert not response.tool_calls


def test_simulate_agent_tool_error_recovery():
    # 1st turn: calls the tool that will fail
    # 2nd turn: acknowledges the error and provides a final answer
    llm = MockedChat.from_contents_data(
        [
            {
                "tools": [{"name": "error_tool", "arguments": {}}],
                "message": None,
            },
            {"tools": None, "message": "I recovered from the error!"},
        ]
    )

    with chats.new("test_chat"):
        response = simulate.simulate_agent(
            llm=llm, tools=[error_tool], output_schema=str
        )

        assert response.content == "I recovered from the error!"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "error_tool"
        assert response.tool_calls[0].error is not None
        assert "Simulated tool failure" in response.tool_calls[0].error
