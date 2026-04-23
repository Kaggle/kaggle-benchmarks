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


def test_prompt_reasoning_maps_to_thinking_config():
    """Tests that reasoning='low' maps to thinking_level='LOW'."""
    llm = MockedGoogleGenAI()
    llm.prompt("Think hard", reasoning="low")

    assert llm.config.thinking_config.thinking_level == "LOW"


def test_prompt_reasoning_high():
    """Tests that reasoning='high' maps to thinking_level='HIGH'."""
    llm = MockedGoogleGenAI()
    llm.prompt("Think hard", reasoning="high")

    assert llm.config.thinking_config.thinking_level == "HIGH"


def test_prompt_include_thoughts():
    """Tests that include_thoughts is passed into thinking_config."""
    llm = MockedGoogleGenAI()
    llm.prompt("Think hard", include_thoughts=True)

    assert llm.config.thinking_config.include_thoughts is True


def test_prompt_reasoning_with_include_thoughts():
    """Tests that reasoning and include_thoughts are merged into one thinking_config."""
    llm = MockedGoogleGenAI()
    llm.prompt("Think hard", reasoning="high", include_thoughts=True)

    assert llm.config.thinking_config.thinking_level == "HIGH"
    assert llm.config.thinking_config.include_thoughts is True


def test_extract_text_excludes_thought_parts():
    """Tests that _extract_text excludes thought parts from content."""
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[
                        types.Part(text="I need to count...", thought=True),
                        types.Part(text="There are 3 r's."),
                    ]
                )
            )
        ]
    )
    text = GoogleGenAI._extract_text(response)
    assert text == "There are 3 r's."
    assert "I need to count..." not in text
    assert "<think>" not in text


def test_extract_thinking_returns_thought_text():
    """Tests that _extract_thinking extracts thought parts separately."""
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[
                        types.Part(text="I need to count...", thought=True),
                        types.Part(text="There are 3 r's."),
                    ]
                )
            )
        ]
    )
    thinking = GoogleGenAI._extract_thinking(response)
    assert thinking == "I need to count..."


def test_extract_thinking_returns_none_without_thoughts():
    """Tests that _extract_thinking returns None when no thought parts exist."""
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[types.Part(text="Hello world")]
                )
            )
        ]
    )
    assert GoogleGenAI._extract_thinking(response) is None


def test_extract_text_excludes_thoughts_from_structured_content():
    """Thought parts should not corrupt structured output."""
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[
                        types.Part(text="Let me think...", thought=True),
                        types.Part(text='{"answer": 42}'),
                    ]
                )
            )
        ]
    )
    text = GoogleGenAI._extract_text(response)
    assert text == '{"answer": 42}'


def test_extract_text_no_thought_parts():
    """Tests that _extract_text returns plain text when no thought parts."""
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[types.Part(text="Hello world")]
                )
            )
        ]
    )
    text = GoogleGenAI._extract_text(response)
    assert text == "Hello world"
    assert "<think>" not in text


def test_extract_text_multi_part_matches_response_text():
    """_extract_text should match response.text for non-thought multi-part responses."""
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[types.Part(text="Hello"), types.Part(text=" world")]
                )
            )
        ]
    )
    assert GoogleGenAI._extract_text(response) == response.text


def test_streaming_thought_output_matches_non_streaming():
    """Streaming and non-streaming should produce the same text for identical content."""
    parts = [
        types.Part(text="step 1", thought=True),
        types.Part(text="step 2", thought=True),
        types.Part(text="Final answer"),
    ]

    # Non-streaming: all parts in one response
    full_response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(parts=parts))
        ]
    )
    non_streaming_text = GoogleGenAI._extract_text(full_response)

    # Streaming: one part per chunk
    chunks = [
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(content=types.Content(parts=[p]))
            ]
        )
        for p in parts
    ]
    streaming_text = "".join(GoogleGenAI._extract_text(c) for c in chunks)

    assert streaming_text == non_streaming_text


def test_thinking_captured_in_response():
    """Tests that thought parts from GenAI response reach the message as thinking."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[
                        types.Part(text="Let me count the letters...", thought=True),
                        types.Part(text="There are 3 r's."),
                    ]
                )
            )
        ],
        usage_metadata=types.UsageMetadata(
            prompt_token_count=10, response_token_count=5
        ),
    )
    mock_client.models.generate_content.return_value = mock_response

    llm = GoogleGenAI(client=mock_client, model="test-gemini")
    llm.stream_responses = False

    with chats.new("Test GenAI thinking") as t:
        response = llm.prompt("How many r's in strawberry?")

    assert response == "There are 3 r's."
    last_message = t.messages[-1]
    assert last_message.thinking == "Let me count the letters..."


def test_prompt_rejects_invalid_reasoning():
    """Tests that an invalid reasoning level raises ValueError."""
    llm = MockedGoogleGenAI()
    with pytest.raises(ValueError, match="Invalid reasoning level"):
        llm.prompt("Think hard", reasoning="hgih")


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

    assert len(llm.contents) == 1
    assert len(llm.contents[0].parts) == 2

    image_part = llm.contents[0].parts[0]
    assert image_part.text is None or image_part.text == ""
    assert image_part.inline_data is not None
    assert image_part.inline_data.mime_type == "image/jpeg"
    assert llm.contents[0].parts[1].text == "What is in this image?"
