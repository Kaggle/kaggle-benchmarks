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


@pytest.mark.parametrize(
    "reasoning, expected_level",
    [
        ("low", "LOW"),
        ("medium", "MEDIUM"),
        ("high", "HIGH"),
    ],
)
def test_prompt_thinking_config(reasoning, expected_level):
    """Tests that reasoning maps to the correct ThinkingConfig."""
    llm = MockedGoogleGenAI()
    llm.prompt("Think hard", reasoning=reasoning)

    tc = llm.config.thinking_config
    assert tc.thinking_level == expected_level


def test_prompt_forwards_api_params():
    """Tests that api_params from prompt() reach the config."""
    llm = MockedGoogleGenAI()
    llm.prompt("Think hard", api_params={"top_p": 0.95, "max_output_tokens": 500})

    assert llm.config.top_p == 0.95
    assert llm.config.max_output_tokens == 500


def test_split_response_separates_content_and_thinking():
    """Tests that _split_response separates content from thought parts."""
    llm = MockedGoogleGenAI()
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
    content, thinking = llm._split_response(response)
    assert content == "There are 3 r's."
    assert thinking == "I need to count..."


def test_split_response_returns_none_thinking_without_thoughts():
    """Tests that _split_response returns None thinking when no thought parts exist."""
    llm = MockedGoogleGenAI()
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(parts=[types.Part(text="Hello world")])
            )
        ]
    )
    content, thinking = llm._split_response(response)
    assert content == "Hello world"
    assert thinking is None


def test_extract_text_multi_part_matches_response_text():
    """_extract_text should match response.text for non-thought multi-part responses."""
    llm = MockedGoogleGenAI()
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[types.Part(text="Hello"), types.Part(text=" world")]
                )
            )
        ]
    )
    assert llm._extract_text(response) == response.text


def test_streaming_thought_output_matches_non_streaming():
    """Streaming and non-streaming should produce the same content for identical input."""
    llm = MockedGoogleGenAI()
    parts = [
        types.Part(text="step 1", thought=True),
        types.Part(text="step 2", thought=True),
        types.Part(text="Final answer"),
    ]

    # Non-streaming: all parts in one response
    full_response = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=parts))]
    )
    non_streaming_text = llm._extract_text(full_response)

    # Streaming: one part per chunk
    chunks = [
        types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=[p]))]
        )
        for p in parts
    ]
    streaming_text = "".join(llm._extract_text(c) for c in chunks)

    assert streaming_text == non_streaming_text


def test_thinking_captured_in_response(mocker):
    """Tests that thought parts from GenAI response are captured as reasoning traces."""
    mock_client = mocker.MagicMock()
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
    assert last_message.reasoning_traces == "Let me count the letters..."


def test_prompt_reasoning_none():
    """Tests that reasoning='none' maps to thinking_budget=0."""
    llm = MockedGoogleGenAI()
    llm.prompt("Think hard", reasoning="none")

    assert llm.config.thinking_config.thinking_budget == 0


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


def test_prompt_rejects_invalid_reasoning():
    """Tests that an invalid reasoning level raises ValueError."""
    llm = MockedGoogleGenAI()
    with pytest.raises(ValueError, match="Invalid reasoning level"):
        llm.prompt("Think hard", reasoning="hgih")


def test_prompt_rejects_thinking_config_in_api_params():
    """Tests that thinking_config in api_params raises an error."""
    llm = MockedGoogleGenAI()
    with pytest.raises(ValueError, match="thinking_config.*not allowed in api_params"):
        llm.prompt(
            "Think hard",
            api_params={"thinking_config": types.ThinkingConfig(thinking_level="LOW")},
        )


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
