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
Golden tests of cookbook examples for backward compatibility and end-to-end verification.

These tests validate core library features across a diverse set of LLMs. They are
excluded from the standard CI/CD pipeline due to requirements for manual
configuration (e.g., API keys, model access) and oversight.

Usage:
    - Run all tests:
        `uv run pytest golden_tests/test_cookbook_examples.py`
    - Run a specific test case:
        `uv run pytest golden_tests/test_cookbook_examples.py::test_extract_int`
    - Run tests for a specific API (e.g., genai/openai):
        `uv run pytest golden_tests/test_cookbook_examples.py -k "genai"`
    - Run tests for a specific feature (e.g., tool/audio/image):
        `uv run pytest golden_tests/test_cookbook_examples.py -k "tool"`
    - Run tests and update the report:
        `uv run pytest golden_tests/test_cookbook_examples.py --generate-report`
"""

# %%
import base64
import dataclasses
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import httpx
import pandas as pd
import pytest
from pydantic import BaseModel, Field

import kaggle_benchmarks as kbench
from kaggle_benchmarks.content_types import (
    audios,
    images,
    videos,
)

# Models to be tested as the primary subject.
TEST_LLM_NAMES = {
    "google/gemini-2.0-flash",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-3-flash-preview",
    "google/gemma-4-31b",
    "qwen/qwen3-235b-a22b-instruct-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
    "anthropic/claude-sonnet-4-6@default",
    "anthropic/claude-opus-4-7@default",
    "openai/gpt-5.5-2026-04-23",
    "deepseek-ai/deepseek-r1-0528",
    "deepseek-ai/deepseek-v3.2",
    # xai/grok-4.20 excluded: genai→xAI tool/reasoning routing is broken in MP.
    "google/gemini-3.1-flash-lite-preview",
}

# Models to be used as judges for evaluation.
JUDGE_LLM_NAMES = {
    "google/gemini-2.5-flash",
}

# Models that support audio input.
AUDIO_LLM_NAMES = {
    "google/gemini-2.0-flash",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-flash-lite-preview",
}


# %%
# Helper functions to wrap a benchmark task to a test function.


def benchmark_test(
    include: Iterable[str] = TEST_LLM_NAMES,
    exclude: Iterable[str] | None = None,
    verify_fn: Callable[[Any], None] | None = None,
    **task_kwargs,
):
    def decorator(task):
        models = set(include)
        if exclude:
            models -= set(exclude)

        # Pre-load the LLM and API as fixtures to make model attributes
        # (e.g., `support_structured_outputs`) and the API name available
        # to the reporting hooks in `conftest.py`.
        llms = []
        for api in ["openai", "genai"]:
            llms += [
                pytest.param(
                    kbench.kaggle.load_model(key, api=api), api, id=f"{api}-{key}"
                )
                for key in sorted(models)
            ]

        @pytest.mark.parametrize("llm, api", llms)
        def test_func(llm, api):
            run = task.run(llm, **task_kwargs)
            assert run.passed
            if verify_fn:
                verify_fn(run)

        return test_func

    return decorator


@contextmanager
def download_temp_image(url: str, suffix: str):
    """Downloads an image to a temporary file and cleans it up afterward."""
    response = httpx.get(url)
    response.raise_for_status()

    fd, temp_path = tempfile.mkstemp(suffix=suffix)

    try:
        with os.fdopen(fd, "wb") as f:
            f.write(response.content)
        yield temp_path

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@pytest.fixture(scope="session", autouse=True)
def configure():
    # Ensure tests fail immediately on exceptions rather than continuing.
    kbench.config.continue_with_exceptions = False


# %%
# --- Test Case: Assessment with a Judge ---


@kbench.task()
def assess_with_judge_task(llm, judge_llm) -> None:
    """
    A task where the LLM answers a question, and a Judge LLM evaluates the answer.
    """
    response: str = llm.prompt("What is Kaggle?")
    kbench.assertions.assert_in("platform", response.lower())

    assessment = kbench.assertions.assess_response_with_judge(
        response_text=response,
        judge_llm=judge_llm,
        criteria=[
            "The answer must mention data science or machine learning.",
            "The answer should mention competitions.",
        ],
    )

    for result in assessment.results:
        kbench.assertions.assert_true(
            result.passed,
            expectation=f"Judge Criterion '{result.criterion}' should pass: {result.reason}",
        )


# We fix the test LLM to one reliable model to focus on testing the judges.
@pytest.mark.parametrize("llm_name", ["google/gemini-2.5-flash"])
@pytest.mark.parametrize("judge_llm_name", JUDGE_LLM_NAMES)
def test_assess_with_judge(llm_name, judge_llm_name):
    llm = kbench.kaggle.load_model(llm_name)
    judge_llm = kbench.kaggle.load_model(judge_llm_name)
    run = assess_with_judge_task.run(llm, judge_llm)
    assert run.passed


# %%
# --- Test Case: Structured Output (Integer Extraction) ---


# Known failures (genai): gpt-5.5 — MP sends empty json_schema.name.
# Known failures (openai): deepseek-r1 — nondeterministic on structured int extraction.
@benchmark_test()
@kbench.task()
def test_extract_int(llm):
    text = "The Apollo 11 mission landed on the Moon in 1969."
    year = llm.prompt(f"Extract the year from this text: '{text}'", schema=int)

    kbench.assertions.assert_equal(
        1969, year, expectation="Extracted year should be 1969."
    )


# %%
# --- Test Case: Structured Output (Bool Extraction) ---


# Known failures (genai): gpt-5.5 — MP sends empty json_schema.name.
@benchmark_test()
@kbench.task()
def test_extract_bool(llm):
    text = "I absolutely loved this movie! It was fantastic."
    is_positive = llm.prompt(f"Is this review positive? '{text}'", schema=bool)

    kbench.assertions.assert_true(
        is_positive, expectation="Sentiment should be positive."
    )


# %%
# --- Test Case: Structured Output (Dict Extraction) ---


# Known failures (genai): gpt-5.5 — MP sends empty json_schema.name.
@benchmark_test()
@kbench.task()
def test_extract_dict(llm):
    text = "Contact info: John Doe, age 42, works as a Software Engineer."

    person_schema = {"name": str, "age": int, "occupation": str}

    person = llm.prompt(f"Extract person details from: '{text}'", schema=person_schema)

    kbench.assertions.assert_equal(
        "John Doe", person.name, expectation="Name should be John Doe."
    )
    kbench.assertions.assert_equal(42, person.age, expectation="Age should be 42.")
    kbench.assertions.assert_contains_regex(
        r"(?i)software engineer",
        person.occupation,
        expectation="Occupation should match.",
    )


# %%
# --- Test Case: Structured Output (dataclass Extraction) ---


@dataclass
class RPGCharacter:
    name: str
    class_type: str
    level: int
    inventory: str


# Known failures (genai): gpt-5.5 — MP sends empty json_schema.name.
@benchmark_test()
@kbench.task()
def test_extract_dataclass(llm):
    character = llm.prompt(
        "Generate a level 5 wizard character for a fantasy game.", schema=RPGCharacter
    )

    kbench.assertions.assert_true(
        len(character.name) > 0, expectation="Character should have a name."
    )
    kbench.assertions.assert_contains_regex(
        r"(?i)wizard", character.class_type, expectation="Class should be Wizard."
    )
    kbench.assertions.assert_equal(5, character.level, expectation="Level should be 5.")
    kbench.assertions.assert_true(
        len(character.inventory) > 0, expectation="Inventory should not be empty."
    )


# %%
# --- Test Case: Structured Output (pydantic Extraction) ---


class Planet(BaseModel):
    name: str
    mass_earth_masses: float = Field(description="Mass relative to Earth")
    has_life: bool = Field(description="Whether the planet is known to have life")
    moons: list[str] = Field(default_factory=list, description="List of major moons")


# Known failures (genai): gpt-5.5 — MP sends empty json_schema.name.
@benchmark_test()
@kbench.task()
def test_extract_pydantic(llm):
    planet = llm.prompt("Provide information about the planet Jupiter.", schema=Planet)

    kbench.assertions.assert_contains_regex(
        r"(?i)jupiter", planet.name, expectation="Planet name should be Jupiter."
    )
    kbench.assertions.assert_true(
        planet.mass_earth_masses > 300,
        expectation="Jupiter mass should be > 300 Earths.",
    )
    kbench.assertions.assert_true(
        len(planet.moons) > 0, expectation="Jupiter should have moons."
    )


# %%
# --- Test Case: Structured Output (composite pydantic Extraction) ---


class FriendsActor(BaseModel):
    actor_name: str
    role_name: str


class Casting(BaseModel):
    actors: list[FriendsActor]


# Known failures (genai): gpt-5.5 — MP sends empty json_schema.name.
@benchmark_test()
@kbench.task()
def test_extract_composite_pydantic(llm):
    casting = llm.prompt("List the 6 main characters of Friends.", schema=Casting)

    kbench.assertions.assert_equal(len(casting.actors), 6)

    names = ", ".join([actor.actor_name for actor in casting.actors])
    role_names = ", ".join([actor.role_name for actor in casting.actors])

    kbench.assertions.assert_in("Jennifer", names)
    kbench.assertions.assert_in("Ross", role_names)


# %%
# --- Test Case: Dataset evaluation ---

df = pd.DataFrame(
    [
        {
            "question": "What's the capital of Singapore",
            "answer": "Singapore",
        },
        {
            "question": "What's the capital of France",
            "answer": "Paris",
        },
    ]
)


@kbench.task(name="single_qa_task", store_task=False)
def single_qa_task(llm, question, answer) -> dict:
    response = llm.prompt(question)
    return {
        "question": question,
        "gold_target": answer,
        "predicted_answer": response,
        "is_correct": answer.lower() in response.lower(),
    }


def assert_multi_qa_result(run):
    assert len(run.result) == 2
    assert run.result[0] == pytest.approx(1.0)
    assert run.result[1] == pytest.approx(0.0)


@benchmark_test(
    df=df,
    verify_fn=assert_multi_qa_result,
)
@kbench.task()
def test_dataset_eval(llm, df) -> tuple[float, float]:
    with kbench.client.enable_cache():
        runs = single_qa_task.evaluate(
            llm=[llm],
            evaluation_data=df,
            n_jobs=2,
            remove_run_files=True,
        )

    eval_df = runs.as_dataframe()

    accuracy = float(eval_df.result.str.get("is_correct").mean())
    std = float(eval_df.result.str.get("is_correct").std())

    return accuracy, std


# %%
# --- Test Case: Dataset evaluation with failure tolerance (production pattern) ---
# Demonstrates the recommended call shape for large evals where transient
# per-sample failures (timeouts, 5xx, network blips) are expected. Pair
# on_failure="continue" with max_attempts > 1 and enable_cache() so that:
#   - failed samples are retried on subsequent attempts
#   - successful samples are skipped (cache hit) on retry
#   - results from all attempts are merged into one Runs object


def assert_resilient_qa_result(run):
    completed, errored = run.result
    # All 2 simple QA samples should complete on the first attempt.
    assert completed == 2
    assert errored == 0


@benchmark_test(
    df=df,
    verify_fn=assert_resilient_qa_result,
)
@kbench.task()
def test_dataset_eval_resilient(llm, df) -> tuple[int, int]:
    with kbench.client.enable_cache():
        results = single_qa_task.evaluate(
            llm=[llm],
            evaluation_data=df,
            n_jobs=2,
            on_failure="continue",  # collect failures instead of raising
            max_attempts=3,  # retry transient failures up to twice
            remove_run_files=True,
        )

    # Split results: completed vs errored. Always filter to .completed_runs
    # before aggregating numeric columns — .result for errored runs is the
    # `results.FAILED` sentinel, which would break .mean() / .sum() / etc.
    return len(results.completed_runs), len(results.errored_runs)


# %%
# --- Test Case: Image inputs (URL) ---


# Excluded: text-only models that don't support image inputs via Model Proxy.
@benchmark_test(
    exclude={
        "deepseek-ai/deepseek-r1-0528",
        "deepseek-ai/deepseek-v3.2",
        "qwen/qwen3-235b-a22b-instruct-2507",
        "qwen/qwen3-next-80b-a3b-instruct",
    }
)
@kbench.task()
def test_image_url(llm):
    """Sends an image URL directly to the model."""
    # Kaggle logo
    image_url = "https://www.kaggle.com/static/images/site-logo.png"

    image = images.from_url(image_url)

    response = llm.prompt("What does this logo say?", image=image)

    kbench.assertions.assert_contains_regex(
        r"(?i)kaggle",
        response,
        expectation="LLM should identify the Kaggle logo.",
    )


# %%
# --- Test Case: Image inputs (Base64) ---


# Excluded: text-only models that don't support image inputs via Model Proxy.
# Known failures: gpt-5.5 — model reports "no image attached" for 1x1 pixel.
@benchmark_test(
    exclude={
        "deepseek-ai/deepseek-r1-0528",
        "deepseek-ai/deepseek-v3.2",
        "qwen/qwen3-235b-a22b-instruct-2507",
        "qwen/qwen3-next-80b-a3b-instruct",
    }
)
@kbench.task()
def test_image_base64(llm):
    """Sends a base64 encoded image with explicit format specification."""
    # Example: A small red dot (PNG)
    # This is a 1x1 red pixel in PNG format
    red_dot_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

    image = images.from_base64(
        red_dot_b64, format="png", extra_api_params={"detail": "low"}
    )

    response = llm.prompt("What color is this image?", image=image)

    kbench.assertions.assert_contains_regex(
        r"(?i)red|pink|salmon|coral",
        response,
        expectation="LLM should identify the color red.",
    )


# %%
# --- Test Case: Image inputs (local file) ---


# Excluded: text-only models that don't support image inputs via Model Proxy.
@benchmark_test(
    exclude={
        "deepseek-ai/deepseek-r1-0528",
        "deepseek-ai/deepseek-v3.2",
        "qwen/qwen3-235b-a22b-instruct-2507",
        "qwen/qwen3-next-80b-a3b-instruct",
    }
)
@kbench.task()
def test_image_local_file(llm):
    # Kaggle logo
    image_url = "https://www.kaggle.com/static/images/site-logo.png"

    with download_temp_image(image_url, suffix=".png") as image_path:
        image = images.from_path(image_path)
        response = llm.prompt("What does this logo say?", image=image)

    kbench.assertions.assert_contains_regex(
        r"(?i)kaggle",
        response,
        expectation="LLM should identify the Kaggle logo.",
    )


# %%
# --- Test Case: Video inputs (URL) ---


# Only Gemini models support video input via Model Proxy.
@benchmark_test(
    include={
        "google/gemini-2.5-flash",
        "google/gemini-2.5-pro",
        "google/gemini-3-flash-preview",
    }
)
@kbench.task()
def test_video_url(llm):
    """Sends a YouTube video URL to the model."""
    # Big Buck Bunny video.
    video_url = "https://www.youtube.com/watch?v=aqz-KE-bpKQ"

    video = videos.from_url(video_url)

    response = llm.prompt("What is this video about? Describe it briefly.", video=video)

    kbench.assertions.assert_contains_regex(
        r"(?i)bunny|rabbit|animal",
        response,
        expectation="LLM should identify the Big Buck Bunny video content.",
    )


# %%
# --- Test Case: Audio inputs (base64) ---


SPEECH_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "speech.mp3")
SPEECH_TRANSCRIPTION_PATTERN = r"(?i)quick\s+brown\s+fox|lazy\s+dog"


@benchmark_test(include=AUDIO_LLM_NAMES)
@kbench.task()
def test_audio_base64(llm):
    """Sends a base64-encoded speech audio clip and asks the model to transcribe it."""
    with open(SPEECH_FIXTURE, "rb") as f:
        audio_content = audios.from_base64(
            base64.b64encode(f.read()).decode(), format="mp3"
        )

    response = llm.prompt("Transcribe this audio exactly.", audio=audio_content)

    kbench.assertions.assert_contains_regex(
        SPEECH_TRANSCRIPTION_PATTERN,
        response,
        expectation="LLM should transcribe the speech audio.",
    )


# %%
# --- Test Case: Audio inputs (local file) ---


@benchmark_test(include=AUDIO_LLM_NAMES)
@kbench.task()
def test_audio_local_file(llm):
    """Sends a speech audio file loaded from disk and asks the model to transcribe it."""
    audio_content = audios.from_path(SPEECH_FIXTURE)

    response = llm.prompt("Transcribe this audio exactly.", audio=audio_content)

    kbench.assertions.assert_contains_regex(
        SPEECH_TRANSCRIPTION_PATTERN,
        response,
        expectation="LLM should transcribe the speech audio.",
    )


# %%
# --- Test Case: Audio inputs (URL) ---


@benchmark_test(include=AUDIO_LLM_NAMES)
@kbench.task()
def test_audio_url(llm):
    """Sends speech audio loaded from a URL and asks the model to transcribe it."""
    import respx

    with open(SPEECH_FIXTURE, "rb") as f:
        audio_bytes = f.read()

    url = "https://example.com/speech.mp3"
    with respx.mock:
        respx.get(url).respond(
            200, content=audio_bytes, headers={"Content-Type": "audio/mpeg"}
        )
        audio_content = audios.from_url(url)

    response = llm.prompt("Transcribe this audio exactly.", audio=audio_content)

    kbench.assertions.assert_contains_regex(
        SPEECH_TRANSCRIPTION_PATTERN,
        response,
        expectation="LLM should transcribe the speech audio.",
    )


# %%
# --- Test Case: Reasoning parameter ---
# Verifies that `reasoning=` doesn't error across providers.
# Actual wiring (thinking_config, reasoning_effort) is covered by unit tests;
# trace capture is verified in test_reasoning_captures_traces for models that
# support it (not all models return traces).


# Known failures: gemini-2.0-flash, gemma-4-31b — do not support reasoning.
@benchmark_test()
@kbench.task()
def test_reasoning_param(llm):
    """Tests that the unified reasoning parameter works across providers."""
    response = llm.prompt(
        "What is 2 + 2? Reply with just the number.",
        reasoning="low",
    )

    kbench.assertions.assert_contains_regex(
        r"4",
        response,
        expectation="Model should answer 4.",
    )


# %%
# --- Test Case: Reasoning traces ---
# Tests that reasoning traces are automatically captured on the message
# when reasoning is enabled, accessible via message.reasoning_traces.


# Only Gemini models expose reasoning traces via the API.
# Known failures: gemini-3-flash-preview — intermittently returns empty traces
# (flaky, currently passing).
@benchmark_test(
    include={
        "google/gemini-2.5-flash",
        "google/gemini-2.5-pro",
        "google/gemini-3-flash-preview",
    },
)
@kbench.task()
def test_reasoning_captures_traces(llm):
    """Tests that reasoning captures reasoning traces on the message."""
    llm.prompt(
        "How many r's are in the word 'strawberry'? Think step by step.",
        reasoning="high",
    )

    chat = kbench.chats.get_current_chat()
    last_message = chat.messages[-1]
    assert last_message.reasoning_traces is not None, (
        "Reasoning traces should be accessible via message.reasoning_traces"
    )
    assert len(last_message.reasoning_traces) > 0, (
        "Reasoning traces should not be empty"
    )


# %%
# --- Test Case: extra_api_params forwarding ---
# Proves that extra_api_params are actually forwarded to the provider by capping
# max output tokens to a small value. If extra_api_params were silently dropped,
# the response would be hundreds of tokens long instead of truncated.

API_PARAMS_LLM_NAMES = {
    "google/gemini-3-flash-preview",
}


@kbench.task()
def _extra_api_params_max_tokens_task_genai(llm):
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


@kbench.task()
def _extra_api_params_max_tokens_task_openai(llm):
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


@pytest.mark.parametrize(
    "llm, api",
    [
        pytest.param(
            kbench.kaggle.load_model(key, api="genai"),
            "genai",
            id=f"genai-{key}",
        )
        for key in sorted(API_PARAMS_LLM_NAMES)
    ],
)
def test_extra_api_params_max_tokens_genai(llm, api):
    run = _extra_api_params_max_tokens_task_genai.run(llm)
    assert run.passed


@pytest.mark.parametrize(
    "llm, api",
    [
        pytest.param(
            kbench.kaggle.load_model(key, api="openai"),
            "openai",
            id=f"openai-{key}",
        )
        for key in sorted(API_PARAMS_LLM_NAMES)
    ],
)
def test_extra_api_params_max_tokens_openai(llm, api):
    run = _extra_api_params_max_tokens_task_openai.run(llm)
    assert run.passed


# %%
# --- Test Case: Image with detail parameter (OpenAI only) ---
# Tests that extra_api_params={"detail": "low"} on images is forwarded via OpenAI.

IMAGE_DETAIL_LLM_NAMES = {
    "google/gemini-3-flash-preview",
}


@kbench.task()
def _image_detail_task(llm):
    """Tests that detail api_param on images is forwarded via OpenAI."""
    red_dot_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

    image = images.from_base64(
        red_dot_b64,
        format="png",
        extra_api_params={"detail": "low"},
    )

    response = llm.prompt("What color is this image?", image=image)

    kbench.assertions.assert_contains_regex(
        r"(?i)red|pink|salmon|coral",
        response,
        expectation="LLM should identify the color red.",
    )


@pytest.mark.parametrize(
    "llm, api",
    [
        pytest.param(
            kbench.kaggle.load_model(key, api="openai"),
            "openai",
            id=f"openai-{key}",
        )
        for key in sorted(IMAGE_DETAIL_LLM_NAMES)
    ],
)
def test_image_with_detail(llm, api):
    run = _image_detail_task.run(llm)
    assert run.passed


# %%
# --- Test Case: Image with media_resolution parameter (GenAI only) ---
# Tests that extra_api_params={"media_resolution": ...} on images is forwarded via GenAI.

MEDIA_RESOLUTION_LLM_NAMES = {
    "google/gemini-3-flash-preview",
}


@kbench.task()
def _media_resolution_task(llm):
    """Tests that media_resolution api_param on images is forwarded via GenAI."""
    red_dot_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

    image = images.from_base64(
        red_dot_b64,
        format="png",
        extra_api_params={"media_resolution": {"level": "MEDIA_RESOLUTION_LOW"}},
    )

    response = llm.prompt("What color is this image?", image=image)

    kbench.assertions.assert_contains_regex(
        r"(?i)red|pink|salmon|coral",
        response,
        expectation="LLM should identify the color red.",
    )


@pytest.mark.parametrize(
    "llm, api",
    [
        pytest.param(
            kbench.kaggle.load_model(key, api="genai"),
            "genai",
            id=f"genai-{key}",
        )
        for key in sorted(MEDIA_RESOLUTION_LLM_NAMES)
    ],
)
def test_image_with_media_resolution(llm, api):
    run = _media_resolution_task.run(llm)
    assert run.passed


# %%
# --- Test Case: Tool Use ---


def run_simple_calculator(a: float, b: float, operator: str) -> float:
    """Supported operators are: + - * and /"""
    if operator == "+":
        return a + b
    if operator == "-":
        return a - b
    if operator == "*":
        return a * b
    if operator == "/":
        return a / b
    raise ValueError(f"Unknown operator: {operator}")


@benchmark_test(
    exclude={
        "deepseek-ai/deepseek-r1-0528",
    }
)
@kbench.task()
def test_simple_tool_use(llm):
    problem = "What is 50 plus 25?"
    expected_answer = 75.0

    final_answer = llm.prompt(problem, tools=[run_simple_calculator])
    kbench.assertions.assert_tool_was_invoked(run_simple_calculator)

    kbench.assertions.assert_true(
        str(int(expected_answer)) in final_answer,
        f"Expected '{expected_answer}' to be in the final answer, got '{final_answer}'.",
    )


# %%
def increment_counter() -> int:
    """Increments a counter and returns the value."""
    increment_counter.count += 1
    return increment_counter.count


@benchmark_test(
    exclude={
        "deepseek-ai/deepseek-r1-0528",
    }
)
@kbench.task()
def test_stateful_tool_double_execution(llm):
    increment_counter.count = 0  # Reset for each test run

    llm.prompt("Call the increment_counter tool.", tools=[increment_counter])

    kbench.assertions.assert_equal(
        1, increment_counter.count, expectation="Tool should be executed exactly once."
    )


# %%
def add_tool(a: float, b: float) -> float:
    """Adds two numbers."""
    add_tool.calls += 1
    return a + b


def multiply_tool(a: float, b: float) -> float:
    """Multiplies two numbers."""
    multiply_tool.calls += 1
    return a * b


# Gemini 2.x via OpenAI backend rejects multiple tool declarations.
# GenAI backend handles multiple tools correctly.
@benchmark_test(
    exclude={
        "deepseek-ai/deepseek-r1-0528",
    }
)
@kbench.task()
def test_multiple_tool_selection(llm):
    add_tool.calls = 0
    multiply_tool.calls = 0

    llm.prompt(
        "What is 12 multiplied by 34? Use the multiply_tool.",
        tools=[add_tool, multiply_tool],
    )

    kbench.assertions.assert_equal(
        1, multiply_tool.calls, expectation="Multiply tool should be called once."
    )
    kbench.assertions.assert_equal(
        0, add_tool.calls, expectation="Add tool should not be called."
    )


# %%
def get_user_profile(user_id: str) -> dict:
    """Returns user profile information as a dictionary."""
    if user_id == "user_123":
        return {"name": "Alice", "role": "Admin", "skills": ["Python", "SQL"]}
    return {"name": "Unknown", "role": "User", "skills": []}


# Known failures: gemini-2.0-flash returns incorrect role on both APIs.
@benchmark_test(
    exclude={
        "deepseek-ai/deepseek-r1-0528",
    }
)
@kbench.task()
def test_complex_tool_return(llm):
    response = llm.prompt(
        "Get the profile for user_123 and tell me what their role is.",
        tools=[get_user_profile],
    )

    kbench.assertions.assert_contains_regex(
        r"(?i)admin", response, expectation="Model should identify the role as Admin."
    )


# %%
def flaky_tool() -> str:
    """This tool always fails with an error."""
    raise ValueError("Tool execution failed simulated error.")


@benchmark_test(
    exclude={
        "deepseek-ai/deepseek-r1-0528",
    }
)
@kbench.task()
def test_tool_error_handling(llm):
    response = llm.prompt(
        "Call the flaky_tool and report what happens.", tools=[flaky_tool]
    )

    kbench.assertions.assert_contains_regex(
        r"(?i)error|failed|valueerror",
        response,
        expectation="Model should report the tool failure.",
    )


# %%
# --- Test Case: Multi-Step Tool Chain ---


def lookup_city_population(city: str) -> int:
    """Looks up the population of a city. Returns the population as an integer."""
    populations = {"Tokyo": 14_000_000, "Paris": 2_100_000, "London": 9_000_000}
    return populations.get(city, 0)


def format_population(population: int) -> str:
    """Formats a population number with thousands separators."""
    return f"{population:,}"


# Known failures (openai): Gemini 2.x rejects multiple tool declarations.
# Known failures (genai): deepseek-v3.2 — flaky 500 errors on multi-step chains.
@pytest.mark.slow
@benchmark_test(
    exclude={
        "deepseek-ai/deepseek-r1-0528",
    }
)
@kbench.task()
def test_multi_step_tool_chain(llm):
    """Tests that the LLM can chain tool calls: look up a value then format it."""
    response = llm.prompt(
        "What is the population of Tokyo? "
        "First look it up with lookup_city_population, "
        "then format the result with format_population.",
        tools=[lookup_city_population, format_population],
    )

    kbench.assertions.assert_tool_was_invoked(lookup_city_population)
    kbench.assertions.assert_contains_regex(
        r"14,000,000",
        response,
        expectation="Response should contain the formatted population of Tokyo.",
    )


# %%
# --- Test Case: Tools with Structured Output ---


class CityInfo(BaseModel):
    """Structured info about a city."""

    name: str = Field(description="The city name.")
    population: int = Field(description="The city's population.")


def get_city_data(city_name: str) -> dict:
    """Returns data about a city including its population."""
    data = {
        "Berlin": {"name": "Berlin", "population": 3_700_000},
        "Sydney": {"name": "Sydney", "population": 5_300_000},
    }
    return data.get(city_name, {"name": city_name, "population": 0})


# Known failures:
# - Gemini 2.0-flash / 2.5-pro (flaky): the model sometimes refuses to call
#   the tool when the user prompt references a schema type (e.g. "CityInfo")
#   that isn't explained during the tool-calling phase. The two-phase loop
#   withholds schema= from tool rounds, so the model doesn't know what
#   "CityInfo" means and asks for clarification instead of calling the tool.
#   This does not affect Gemini 3.x or other model families.
@benchmark_test(
    exclude={
        "deepseek-ai/deepseek-r1-0528",
        "deepseek-ai/deepseek-v3.2",
    }
)
@kbench.task()
def test_tool_with_schema_output(llm):
    """Tests that tools and schema= work together: tool provides data,
    LLM returns structured output."""
    result = llm.prompt(
        "Look up the city data for Berlin and return it as a CityInfo.",
        tools=[get_city_data],
        schema=CityInfo,
    )

    kbench.assertions.assert_tool_was_invoked(get_city_data)
    kbench.assertions.assert_true(
        isinstance(result, CityInfo),
        f"Expected CityInfo, got {type(result).__name__}",
    )
    kbench.assertions.assert_equal(
        "Berlin", result.name, expectation="City name should be Berlin."
    )
    kbench.assertions.assert_equal(
        3_700_000,
        result.population,
        expectation="Population should be 3,700,000.",
    )


# %%
# --- Test Case: ChatRoom — add_participant + LLM Cloning ---
# Verifies that the same LLM object can be added as multiple participants via
# add_participant(), each receiving a distinct identity (name, system_prompt)
# and producing correct responses without role collision.


CHATROOM_LLM_NAMES = {
    "google/gemini-2.5-flash",
    "google/gemini-3-flash-preview",
    "anthropic/claude-sonnet-4-6@default",
}


@benchmark_test(include=CHATROOM_LLM_NAMES)
@kbench.task()
def test_chatroom_add_participant(llm):
    """Tests that the same LLM added twice yields independent participants."""
    room = kbench.ChatRoom(
        system_prompt="A quick Q&A between two experts.",
        name="Host",
    )

    alice = room.add_participant(
        llm,
        name="Alice",
        avatar="👩",
        system_prompt="You are Alice, a Python expert. Always mention Python in your replies.",
    )
    bob = room.add_participant(
        llm,
        name="Bob",
        avatar="👨",
        system_prompt="You are Bob, a Rust expert. Always mention Rust in your replies.",
    )

    with room:
        room.post(
            "Each expert, name your favorite programming language in one sentence."
        )
        alice_reply = alice.reply()
        bob_reply = bob.reply()

    # Clones must be distinct objects
    kbench.assertions.assert_true(
        alice is not bob,
        "add_participant must return distinct objects for the same LLM.",
    )

    # Identity injection: each participant should follow their own system_prompt
    kbench.assertions.assert_contains_regex(
        r"(?i)python",
        alice_reply,
        expectation="Alice (Python expert) should mention Python.",
    )
    kbench.assertions.assert_contains_regex(
        r"(?i)rust",
        bob_reply,
        expectation="Bob (Rust expert) should mention Rust.",
    )

    # Transcript must attribute messages to the correct sender
    kbench.assertions.assert_equal(
        "Alice",
        room.messages[1].sender.name,
        expectation="Second message sender should be Alice.",
    )
    kbench.assertions.assert_equal(
        "Bob",
        room.messages[2].sender.name,
        expectation="Third message sender should be Bob.",
    )


# %%
# --- Test Case: ChatRoom — Structured Output via reply(schema=) ---
# Verifies that reply(schema=) returns structured output (dataclass) from
# within a ChatRoom context, combining multi-participant rooms with schema.


@dataclasses.dataclass(frozen=True)
class _CityFact:
    """A structured fact about a city."""

    city: str
    country: str
    population_millions: float


@benchmark_test(include=CHATROOM_LLM_NAMES)
@kbench.task()
def test_chatroom_talk_structured_output(llm):
    """Tests that reply(schema=) works inside a ChatRoom."""
    room = kbench.ChatRoom(
        system_prompt="A geography quiz game.",
        name="QuizMaster",
    )

    player = room.add_participant(
        llm,
        name="Player",
        system_prompt="You are a geography expert. Answer questions accurately.",
    )

    with room:
        room.post(
            "What is the capital of France? Provide city, country, and approximate population in millions."
        )
        fact = player.reply(schema=_CityFact)

    kbench.assertions.assert_contains_regex(
        r"(?i)paris",
        fact.city,
        expectation="City should be Paris.",
    )
    kbench.assertions.assert_contains_regex(
        r"(?i)france",
        fact.country,
        expectation="Country should be France.",
    )
    kbench.assertions.assert_true(
        0.5 < fact.population_millions < 15.0,
        f"Population should be reasonable, got {fact.population_millions}M.",
    )


# %%
# --- Test Case: ChatRoom — Multi-Turn Conversation ---
# Verifies that room.post() and reply() produce correct multi-turn histories
# and that room.messages captures the full transcript after exit.


@benchmark_test(include=CHATROOM_LLM_NAMES)
@kbench.task()
def test_chatroom_multi_turn(llm):
    """Tests multi-turn conversation: 2 rounds of moderator prompt → LLM reply."""
    room = kbench.ChatRoom(
        system_prompt="A two-round trivia game.",
        name="Trivia",
    )

    player = room.add_participant(
        llm,
        name="Player",
        system_prompt="You are a trivia contestant. Answer each question in one concise sentence.",
    )

    with room:
        # Round 1
        room.post("Round 1: What is the chemical symbol for gold?")
        r1 = player.reply()

        # Round 2
        room.post("Round 2: What is the chemical symbol for silver?")
        r2 = player.reply()

    # Transcript must contain all messages (2 posts + 2 replies = 4)
    kbench.assertions.assert_equal(
        4,
        len(room.messages),
        expectation="Room should have 4 messages (2 posts + 2 replies).",
    )

    # Content verification
    kbench.assertions.assert_contains_regex(
        r"(?i)au",
        r1,
        expectation="Answer should contain 'Au' for gold.",
    )
    kbench.assertions.assert_contains_regex(
        r"(?i)ag",
        r2,
        expectation="Answer should contain 'Ag' for silver.",
    )


# %%
# --- Test Case: ChatRoom — Private Channel Isolation ---
# Verifies that private_channel() messages are only visible to members
# and invisible to non-members. This is the core information-asymmetry
# primitive used by Werewolf and similar social deduction benchmarks.


@benchmark_test(include=CHATROOM_LLM_NAMES)
@kbench.task()
def test_chatroom_private_channel(llm):
    """Tests that private_channel messages are invisible to non-members."""
    room = kbench.ChatRoom(
        system_prompt="A team coordination exercise with a secret planning phase.",
        name="Coordinator",
    )

    alice = room.add_participant(
        llm,
        name="Alice",
        avatar="👩",
        system_prompt=(
            "You are Alice. In the secret channel, always mention the codeword 'BLUEPRINT'. "
            "In the public channel, never mention the codeword."
        ),
    )
    bob = room.add_participant(
        llm,
        name="Bob",
        avatar="👨",
        system_prompt="You are Bob. You do not know any secret codewords. Report what you know.",
    )

    with room:
        room.post("Public phase: everyone introduces themselves briefly.")
        alice.reply()
        bob.reply()

        # Private channel: only Alice is a member
        secret = room.private_channel([alice], name="Secret Planning")
        with secret:
            secret.post("Alice, share your secret plan and mention the codeword.")
            secret_reply = alice.reply()

        # Back in public: ask Bob to summarize what he knows
        room.post(
            "Bob, summarize everything you've heard so far. Mention any codewords if you heard any."
        )
        bob_summary = bob.reply()

    # Alice's secret reply should contain the codeword
    kbench.assertions.assert_contains_regex(
        r"(?i)blueprint",
        secret_reply,
        expectation="Alice's private message should contain the codeword 'BLUEPRINT'.",
    )

    # Bob's summary should NOT contain the codeword (he never saw it)
    kbench.assertions.assert_true(
        "blueprint" not in bob_summary.lower(),
        f"Bob should NOT know the codeword, but his summary was: '{bob_summary[:200]}'",
    )


# %%
# --- Test Case: ChatRoom — Scripted Messages via room.post() ---
# Verifies that room.post() messages are visible to LLM participants
# and that LLMs can respond to scripted messages correctly.


@benchmark_test(include=CHATROOM_LLM_NAMES)
@kbench.task()
def test_chatroom_room_post(llm):
    """Tests that room.post() messages are visible and LLMs respond correctly."""
    room = kbench.ChatRoom(
        system_prompt="A simple number guessing game. The host posts a number, the Player guesses.",
        name="NumberGame",
    )

    player = room.add_participant(
        llm,
        name="Player",
        system_prompt=(
            "You are a player in a number game. When told a number, "
            "respond with that number plus one. Reply with ONLY the number."
        ),
    )

    with room:
        room.post("The number is: 41")
        reply = player.reply()

    kbench.assertions.assert_contains_regex(
        r"42",
        reply,
        expectation="Player should respond with 42 (41 + 1).",
    )

    # Post message is in transcript
    kbench.assertions.assert_true(
        room.messages[0].sender.name == "NumberGame",
        "First message should be from the room narrator.",
    )


# %%
# --- Test Case: ChatRoom — remove_participant ---
# Verifies that room.remove_participant() drops a participant from the active
# roster so that surviving participants no longer see them, and that calling
# reply() on the removed participant raises RuntimeError.
# Pattern from: documentation/examples/chatroom_werewolf.py (night elimination).


@benchmark_test(include=CHATROOM_LLM_NAMES)
@kbench.task()
def test_chatroom_remove_participant(llm):
    """Tests that remove_participant removes a participant from the room."""
    room = kbench.ChatRoom(
        system_prompt="A survival game. Players are eliminated each round.",
        name="GameMaster",
    )

    alice = room.add_participant(
        llm,
        name="Alice",
        avatar="👩",
        system_prompt="You are Alice. Answer questions concisely.",
    )
    bob = room.add_participant(
        llm,
        name="Bob",
        avatar="👨",
        system_prompt="You are Bob. Answer questions concisely.",
    )
    charlie = room.add_participant(
        llm,
        name="Charlie",
        avatar="🧑",
        system_prompt="You are Charlie. Answer questions concisely.",
    )

    with room:
        # Pre-removal: everyone participates
        room.post("All players, say hello briefly.")
        alice.reply()
        bob.reply()
        charlie.reply()

        # Remove Bob (mirrors werewolf night elimination)
        room.remove_participant(bob)
        room.post("Bob has been eliminated! Only surviving players remain.")

        # Ask a survivor who is still in the game
        room.post(
            "Alice, list the names of ALL other players still in this conversation. "
            "Reply with only their names separated by commas."
        )
        alice_response = alice.reply()

    # Bob should NOT appear in the survivor's awareness
    kbench.assertions.assert_true(
        "bob" not in alice_response.lower(),
        f"Alice should not mention eliminated Bob, but said: '{alice_response[:200]}'",
    )

    # Charlie should still be mentioned
    kbench.assertions.assert_contains_regex(
        r"(?i)charlie",
        alice_response,
        expectation="Alice should mention surviving player Charlie.",
    )

    # Removed participant cannot reply — RuntimeError expected
    try:
        with room:
            bob.reply()
        raise AssertionError("bob.reply() should have raised RuntimeError")
    except RuntimeError:
        pass  # Expected

    # Historical messages are preserved in the transcript
    senders = [msg.sender.name for msg in room.messages]
    kbench.assertions.assert_in(
        "Bob",
        senders,
        expectation="Bob's pre-removal messages should remain in the transcript.",
    )
