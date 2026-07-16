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
# ## How to Send Images to LLM
#
# You can send images using a direct URL or a Base64 string. We also provide helper function to load a local image to base64.
# - When using URL, the image format is automatically guessed.
# - When using Base64, you need to specify the image format if it's different from default jpeg.
# - Use `from_path`` to load from local images.
# %%
import httpx

import kaggle_benchmarks as kbench
from kaggle_benchmarks.content_types import images

# %%

# %% [markdown]
# ---
# ### Example 1. Sending LLM image from URL
# ---


# %%
@kbench.task("Describe Image (URL)")
def describe_image_url(llm):
    """Sends an image URL directly to the model."""
    # Kaggle logo
    image_url = "https://www.kaggle.com/static/images/site-logo.png"

    # Create Image object from URL.
    # It will guess image type from URL.
    image = images.from_url(image_url, caption="a logo")

    response = llm.prompt("What does this logo say?", image=image)

    kbench.assertions.assert_contains_regex(
        r"(?i)kaggle",
        response,
        expectation="LLM should identify the Kaggle logo.",
    )


describe_image_url.run(kbench.llm)

# %% [markdown]
# ---
# ### Example 2. Sending LLM image from Base64 (specifying format parameter)
# ---


# %%
@kbench.task("Describe Image (Base64)")
def describe_image_base64(llm):
    """Sends a base64 encoded image with explicit format specification."""
    # Example: A small red dot (PNG)
    # This is a 1x1 red pixel in PNG format
    red_dot_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

    # Create Image object from Base64, specifying the format as 'png'
    # The 'format' parameter is important when the image is not a JPEG (default)
    image = images.from_base64(red_dot_b64, format="png", caption="a colorful dot")

    response = llm.prompt("What color is this image?", image=image)

    kbench.assertions.assert_contains_regex(
        r"(?i)red",
        response,
        expectation="LLM should identify the color red.",
    )


describe_image_base64.run(kbench.llm)

# %% [markdown]
# ---
# ### Example 3. Sending LLM image from local image file
# ---

# %%

# Download the file to local file first


def download_image(url, filename):
    try:
        with httpx.Client() as client:
            response = client.get(url)
            response.raise_for_status()  # Raise error for 4xx/5xx responses

            with open(filename, "wb") as file:
                file.write(response.content)

        print(f"Successfully downloaded: {filename}")

    except httpx.HTTPError as e:
        print(f"Error downloading image: {e}")


download_image("https://www.kaggle.com/static/images/site-logo.png", "kaggle_logo.png")


# Benchmark task using local file
@kbench.task("Describe Image (Local File)")
def describe_local_image(llm):
    """Sends a local image with explicit format specification."""

    # Load a local image from an attached Kaggle dataset
    image = images.from_path("kaggle_logo.png")
    prompt = "How many letters are there in the image?"

    response = llm.prompt(prompt, image=image, schema=int)

    kbench.assertions.assert_equal(
        6,
        response,
        expectation="LLM should recognize the logo.",
    )


describe_local_image.run(kbench.llm)

# %% [markdown]
# ---
# ### Example 4. Comparing two images
# ---
# This example demonstrates sending multiple images to the model for comparison.


# %%
@kbench.task("Compare Logos")
def compare_logos(llm):
    """Sends two images and asks for differences."""

    img1 = images.from_url(
        "https://www.kaggle.com/static/images/logos/kaggle-logo-transparent-300.png",
        caption="Logo 1",
    )
    img2 = images.from_url(
        "https://www.kaggle.com/static/images/logos/kaggle-logo-gray-300.png",
        caption="Logo 2",
    )

    # Use `send` to enable multi-image conversation.
    # Since `send` doesn't auto-convert URLs, we explicitly encode images to base64
    # so it will work with more models that don't directly accept an image URL.
    kbench.user.send(images.from_image_url(img1))
    kbench.user.send(images.from_image_url(img2))

    # For models that directly work with image URL
    # You can simply call
    # kbench.user.send(img1)
    # kbench.user.send(img2)

    response = llm.prompt("What are the main differences of these two images.")

    assessment = kbench.assertions.assess_response_with_judge(
        response_text=response,
        judge_llm=kbench.judge_llm,
        criteria=[
            "The answer should highlight the main difference is the background.",
            "The answer should mention the font are the same",
        ],
    )

    for result in assessment.results:
        kbench.assertions.assert_true(
            result.passed,
            expectation=f"Judge Criterion '{result.criterion}' should pass: {result.reason}",
        )


compare_logos.run(kbench.llm)

# %%
