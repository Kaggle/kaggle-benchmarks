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

import pytest
from google.genai import types
from pydantic import BaseModel

from kaggle_benchmarks import actors, chats
from kaggle_benchmarks.actors.llms import GoogleGenAI, LLMResponse
from kaggle_benchmarks.content_types.images import ImageBase64


class MockedGoogleGenAI(GoogleGenAI):
    """A mock of the GoogleGenAI class that records inputs and returns fixed outputs."""

    def __init__(self, **kwargs):
        super().__init__(client=None, model="mocked-gemini", **kwargs)
        self.support_temperature = True
        self.support_structured_outputs = True

    def _call_api(
        self, contents: list[types.Content], config: types.GenerateContentConfig
    ):
        self.contents = contents
        self.config = config

        if config.response_schema:
            mock_json_response = (
                '{"name": "Mock Recipe", "ingredients": ["water", "flour"]}'
            )
            return LLMResponse(
                content=mock_json_response,
                meta={"input_tokens": 10, "output_tokens": 10},
            )

        if self.stream_responses:

            def stream_generator():
                yield LLMResponse(content="Streaming", meta={"input_tokens": 15})
                yield LLMResponse(
                    content=" response",
                    meta={"input_tokens": 15, "output_tokens": 8},
                )

            return stream_generator()
        else:
            return LLMResponse(
                content="Non-streaming response",
                meta={"input_tokens": 10, "output_tokens": 4},
            )


def test_invoke_basic():
    """Tests that a simple user prompt is formatted correctly."""
    llm = MockedGoogleGenAI()
    llm.prompt("Hello")

    assert len(llm.contents) == 1
    assert llm.contents[0].role == "user"
    assert llm.contents[0].parts[0].text == "Hello"
    # Ensure no system instruction is passed by default
    assert llm.config.system_instruction is None


def test_invoke_with_system_instruction():
    """Tests that system instructions are placed correctly in the config."""
    llm = MockedGoogleGenAI()
    llm.respond(system="You are a helpful assistant.")

    assert llm.config.system_instruction == "You are a helpful assistant."


def test_invoke_with_config_params():
    """Tests that temperature and seed are passed correctly into the config."""
    llm = MockedGoogleGenAI()
    llm.prompt("Be creative", temperature=0.9, seed=42)

    assert llm.config.temperature == 0.9


@pytest.mark.parametrize("streaming", [True, False])
def test_streaming_and_non_streaming_responses(streaming):
    """Tests both streaming and non-streaming modes and checks metadata."""
    llm = MockedGoogleGenAI()
    llm.stream_responses = streaming

    with chats.new("Test GenAI Tokens") as t:
        response_content = llm.prompt("Tell me a story.")

        last_message = t.messages[-1]
        assert last_message.sender is llm

        if streaming:
            assert response_content == "Streaming response"
            assert last_message._meta["input_tokens"] == 15
            assert last_message._meta["output_tokens"] == 8
        else:
            assert response_content == "Non-streaming response"
            assert last_message._meta["input_tokens"] == 10
            assert last_message._meta["output_tokens"] == 4


def test_invoke_with_tools():
    """Tests that tools are correctly passed into the config."""
    llm = MockedGoogleGenAI()

    def multiply(a: int, b: int) -> int:
        return a * b

    llm.respond(tools=[multiply])

    assert llm.config.tools is not None
    assert len(llm.config.tools) == 1
    assert llm.config.tools[0] == multiply


def test_invoke_with_structured_output():
    """Tests that a schema correctly configures the response format."""
    llm = MockedGoogleGenAI()

    class Recipe(BaseModel):
        name: str
        ingredients: list[str]

    response = llm.prompt("Give me a recipe.", schema=Recipe)

    assert isinstance(response, Recipe)
    assert response.name == "Mock Recipe"
    assert llm.config.response_schema == Recipe


def test_invoke_with_image_input():
    """Tests that image payloads are correctly formatted as inline data."""
    llm = MockedGoogleGenAI()

    mock_image = ImageBase64(
        b64_string="R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",
        mime_type="image/jpeg",
    )

    with chats.new("Image Test Chat"):
        actors.user.send(mock_image)
        llm.prompt("What is in this image?")

    assert len(llm.contents) == 2

    image_part = llm.contents[0].parts[0]
    assert image_part.text is None or image_part.text == ""
    assert image_part.inline_data is not None
    assert image_part.inline_data.mime_type == "image/jpeg"
    assert llm.contents[1].parts[0].text == "What is in this image?"
