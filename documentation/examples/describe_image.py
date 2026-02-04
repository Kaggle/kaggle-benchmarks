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

# %% [markdown]
# ---
# ## Vision-Capable Tasks for Gemini
#
# Here we add two tasks to test the vision capabilities of the Gemini model using the genai api.
# The first task takes a pre-converted base64 image string, and the second
# handles a direct image URL.
# %%
from kaggle_benchmarks import assertions, content_types, task
from kaggle_benchmarks.kaggle import load_model

llm = load_model(
    model_name="google/gemini-2.5-pro",
    api="genai",
)

# ---
# ### 1. Task for Image with Base64 Input
# ---


@task("Describe Image (Base64)")
def describe_image_base64(llm, image_base64: str, question: str, answer: str):
    """Sends a base64 image string and a question to a vision model."""
    image = content_types.images.from_base64(image_base64, caption=question)
    response = llm.prompt(image)
    assertions.assert_contains_regex(
        f"(?i){answer}",
        response,
        expectation="LLM should identify the object correctly.",
    )


dog_image_url = (
    "https://upload.wikimedia.org/wikipedia/commons/4/47/American_Eskimo_Dog.jpg"
)
dog_image_base64 = content_types.images.image_url_to_base64(dog_image_url)

describe_image_base64.run(
    llm=llm,
    image_base64=dog_image_base64,
    question="What is in the picture?",
    answer="dog",
)
# %%
# ---
# ### 2. Task for Image with URL Input
# ---


@task("Describe Image (URL)")
def describe_image_url(llm, image_url: str, question: str, answer: str):
    """Sends an image URL and a question to a vision model."""
    image = content_types.images.from_base64(
        content_types.images.image_url_to_base64(image_url),
        caption=question,
    )

    response = llm.prompt(image)
    assertions.assert_contains_regex(
        f"(?i){answer}",
        response,
        expectation="LLM should identify the object correctly.",
    )


describe_image_url.run(
    llm=llm,
    image_url=dog_image_url,
    question="Describe the main subject in this image.",
    answer="dog",
)

# %%
