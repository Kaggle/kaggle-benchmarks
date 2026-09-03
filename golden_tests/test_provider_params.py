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

"""Provider-parameter forwarding tasks, and their golden tests (live-only).

Each task is followed by its test. These prove that ``extra_api_params``
actually reach the underlying provider — capping output tokens, or forwarding
image ``detail`` / ``media_resolution``. A fake model never touches a provider,
so it would verify nothing here: every test is parametrized over a live,
api-pinned pool only and nothing is scripted. Without a configured provider the
pools are empty and pytest skips these tests cleanly.
"""

import pytest
from models import GENAI_PROBE_MODELS, OPENAI_PROBE_MODELS

import kaggle_benchmarks as kbench

_RED_DOT_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


@kbench.task(name="extra_api_params_max_tokens_genai")
def extra_api_params_max_tokens_genai(llm):
    """Proves extra_api_params reach GenAI by capping max_output_tokens."""
    response = llm.prompt(
        "Write a 500-word essay about the history of computing.",
        extra_api_params={"max_output_tokens": 10},
    )
    word_count = len(response.split())
    kbench.assertions.assert_true(
        word_count < 30,
        expectation=f"Response should be truncated (got {word_count} words). "
        "If extra_api_params were ignored, response would be ~500 words.",
    )


@pytest.mark.parametrize("llm", GENAI_PROBE_MODELS)
def test_extra_api_params_max_tokens_genai(llm):
    assert extra_api_params_max_tokens_genai.run(llm).passed


@kbench.task(name="extra_api_params_max_tokens_openai")
def extra_api_params_max_tokens_openai(llm):
    """Proves extra_api_params reach OpenAI by capping max_completion_tokens."""
    response = llm.prompt(
        "Write a 500-word essay about the history of computing.",
        extra_api_params={"max_completion_tokens": 50},
    )
    word_count = len(response.split())
    kbench.assertions.assert_true(
        word_count < 80,
        expectation=f"Response should be truncated (got {word_count} words). "
        "If extra_api_params were ignored, response would be ~500 words.",
    )


@pytest.mark.parametrize("llm", OPENAI_PROBE_MODELS)
def test_extra_api_params_max_tokens_openai(llm):
    assert extra_api_params_max_tokens_openai.run(llm).passed


@kbench.task(name="image_with_detail")
def image_with_detail(llm):
    """Tests that detail api_param on images is forwarded via OpenAI."""
    image = kbench.content_types.images.from_base64(
        _RED_DOT_B64,
        format="png",
        extra_api_params={"detail": "low"},
    )

    response = llm.prompt("What color is this image?", image=image)

    kbench.assertions.assert_contains_regex(
        r"(?i)red|pink|salmon|coral",
        response,
        expectation="LLM should identify the color red.",
    )


@pytest.mark.parametrize("llm", OPENAI_PROBE_MODELS)
def test_image_with_detail(llm):
    assert image_with_detail.run(llm).passed


@kbench.task(name="image_with_media_resolution")
def image_with_media_resolution(llm):
    """Tests that media_resolution api_param on images is forwarded via GenAI."""
    image = kbench.content_types.images.from_base64(
        _RED_DOT_B64,
        format="png",
        extra_api_params={"media_resolution": {"level": "MEDIA_RESOLUTION_LOW"}},
    )

    response = llm.prompt("What color is this image?", image=image)

    kbench.assertions.assert_contains_regex(
        r"(?i)red|pink|salmon|coral",
        response,
        expectation="LLM should identify the color red.",
    )


@pytest.mark.parametrize("llm", GENAI_PROBE_MODELS)
def test_image_with_media_resolution(llm):
    assert image_with_media_resolution.run(llm).passed
