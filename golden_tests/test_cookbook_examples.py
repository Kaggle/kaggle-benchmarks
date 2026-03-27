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
    - Run tests and update the report:
        `uv run pytest golden_tests/test_cookbook_examples.py --generate-report`
"""

# %%
import json
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
from kaggle_benchmarks import messages
from kaggle_benchmarks.content_types import images, videos

# Models to be tested as the primary subject.
TEST_LLM_NAMES = {
    "google/gemini-2.0-flash",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-3-flash-preview",
    "google/gemma-3-12b",
    "qwen/qwen3-235b-a22b-instruct-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
    "anthropic/claude-haiku-4-5@20251001",
    "anthropic/claude-opus-4-5@20251101",
    "anthropic/claude-sonnet-4-5@20250929",
    "deepseek-ai/deepseek-r1-0528",
    "deepseek-ai/deepseek-v3.2",
    "zai/glm-5",
    "google/gemini-3.1-flash-lite-preview",
}

# Models to be used as judges for evaluation.
JUDGE_LLM_NAMES = {
    "google/gemini-2.5-flash",
    "google/gemini-3-flash-preview",
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
@pytest.mark.parametrize("llm_name", ["google/gemini-3-flash-preview"])
@pytest.mark.parametrize("judge_llm_name", JUDGE_LLM_NAMES)
def test_assess_with_judge(llm_name, judge_llm_name):
    llm = kbench.llms[llm_name]
    judge_llm = kbench.llms[judge_llm_name]
    run = assess_with_judge_task.run(llm, judge_llm)
    assert run.passed


# %%
# --- Test Case: Structured Output (Integer Extraction) ---


@benchmark_test(
    exclude={
        "google/gemma-3-12b",
    }
)
@kbench.task()
def test_extract_int(llm):
    text = "The Apollo 11 mission landed on the Moon in 1969."
    year = llm.prompt(f"Extract the year from this text: '{text}'", schema=int)

    kbench.assertions.assert_equal(
        1969, year, expectation="Extracted year should be 1969."
    )


# %%
# --- Test Case: Structured Output (Bool Extraction) ---


@benchmark_test(
    exclude={
        "google/gemma-3-12b",
    }
)
@kbench.task()
def test_extract_bool(llm):
    text = "I absolutely loved this movie! It was fantastic."
    is_positive = llm.prompt(f"Is this review positive? '{text}'", schema=bool)

    kbench.assertions.assert_true(
        is_positive, expectation="Sentiment should be positive."
    )


# %%
# --- Test Case: Structured Output (Dict Extraction) ---


@benchmark_test(
    exclude={
        "google/gemma-3-12b",
    }
)
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


@benchmark_test(
    exclude={
        "google/gemma-3-12b",
    }
)
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


@benchmark_test(
    exclude={
        "google/gemma-3-12b",
    }
)
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


class Actor(BaseModel):
    actor_name: str
    role_name: str


class Casting(BaseModel):
    actors: list[Actor]


@benchmark_test(
    exclude={
        "google/gemma-3-12b",
    }
)
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


@benchmark_test(df=df, verify_fn=assert_multi_qa_result)
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
# --- Test Case: Image inputs (URL) ---


@benchmark_test(
    exclude={
        "deepseek-ai/deepseek-r1-0528",
        "deepseek-ai/deepseek-v3.2",
        "qwen/qwen3-235b-a22b-instruct-2507",
        "qwen/qwen3-next-80b-a3b-instruct",
        "zai/glm-5",
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


@benchmark_test(
    exclude={
        "deepseek-ai/deepseek-r1-0528",
        "deepseek-ai/deepseek-v3.2",
        "qwen/qwen3-235b-a22b-instruct-2507",
        "qwen/qwen3-next-80b-a3b-instruct",
        "anthropic/claude-sonnet-4-5@20250929",
        "zai/glm-5",
    }
)
@kbench.task()
def test_image_base64(llm):
    """Sends a base64 encoded image with explicit format specification."""
    # Example: A small red dot (PNG)
    # This is a 1x1 red pixel in PNG format
    red_dot_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

    image = images.from_base64(red_dot_b64, format="png")

    response = llm.prompt("What color is this image?", image=image)

    kbench.assertions.assert_contains_regex(
        r"(?i)red|pink|salmon|coral",
        response,
        expectation="LLM should identify the color red.",
    )


# %%
# --- Test Case: Image inputs (local file) ---


@benchmark_test(
    exclude={
        "deepseek-ai/deepseek-r1-0528",
        "deepseek-ai/deepseek-v3.2",
        "qwen/qwen3-235b-a22b-instruct-2507",
        "qwen/qwen3-next-80b-a3b-instruct",
        "zai/glm-5",
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
# --- Test Case: Many images (batch) ---

# 1x1 red pixel PNG.
RED_PIXEL_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


@benchmark_test(
    exclude={
        "anthropic/claude-opus-4-5@20251101",
        "anthropic/claude-sonnet-4-5@20250929",
    }
)
@kbench.task()
def test_many_images(llm):
    """Sends 100 small images in a single request to verify batch image handling."""
    num_images = 100

    with kbench.chats.new("many_images"):
        for _ in range(num_images):
            img = images.from_base64(RED_PIXEL_B64, format="png")
            kbench.user.send(img)

        response = llm.prompt(
            f"I sent you {num_images} small red images. "
            "Confirm you received them and say approximately how many you see."
        )

    kbench.assertions.assert_contains_regex(
        r"(?i)red|image|received",
        response,
        expectation="LLM should acknowledge receiving many red images.",
    )


# %%
# --- Test Case: Video inputs (URL) ---


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
# --- Test Case: Tool Use ---
# This doesn't work with "genai" API for now
# So test it with `-k "openai"` only.
# TODO: Rewrite this test after tool refactoring.


def run_simple_calculator(a: float, b: float, operator: str) -> float:
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
        "anthropic/claude-haiku-4-5@20251001",
        "anthropic/claude-opus-4-5@20251101",
        "anthropic/claude-sonnet-4-5@20250929",
        "deepseek-ai/deepseek-r1-0528",
        "deepseek-ai/deepseek-v3.2",
        "google/gemma-3-12b",
        "google/gemini-3.1-flash-lite-preview",
    }
)
@kbench.task()
def test_manual_tool_use(llm):
    problem = "What is 50 plus 25?"
    expected_answer = 75.0

    # Define the tool schema (JSON Schema format).
    calculator_tool = {
        "type": "function",
        "function": {
            "name": "simple_calculator",
            "description": "Calculates the result of an arithmetic operation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first number."},
                    "b": {"type": "number", "description": "The second number."},
                    "operator": {
                        "type": "string",
                        "description": "The operator (+, -, *, /).",
                    },
                },
                "required": ["a", "b", "operator"],
            },
        },
    }

    # 1. Get the initial response from the LLM.
    kbench.actors.user.send(problem)
    tool_call_msg = llm.respond(tools=[calculator_tool])
    tool_calls = tool_call_msg.tool_calls

    kbench.assertions.assert_true(bool(tool_calls), "LLM was expected to call a tool.")

    # 2. Parse the tool call arguments.
    tool_call = tool_calls[0]
    function_args = json.loads(tool_call["function"]["arguments"])
    # Removes 'signature' parameter in thinking mode.
    function_args.pop("signature", None)

    # 3. Execute the actual Python function.
    tool_result = run_simple_calculator(**function_args)

    # 4. Send the tool result back to the LLM.
    tool_actor = kbench.actors.Actor(name="Tool", role="tool", avatar="🛠️")
    tool_actor.send(
        messages.Message(
            sender=tool_actor,
            content=str(tool_result),
            _meta={"tool_call_id": tool_call["id"]},
        )
    )

    # 5. Get the final answer.
    final_answer_msg = llm.respond()
    final_answer = final_answer_msg.content

    kbench.assertions.assert_true(
        str(int(expected_answer)) in final_answer,
        f"Expected '{expected_answer}' to be in the final answer, got '{final_answer}'.",
    )
