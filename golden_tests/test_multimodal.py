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

"""Multimodal (image / audio / video) benchmark tasks, and their golden tests.

Each task is followed by its tests. Two kinds of task live here:

* **Offline-scriptable** tasks whose input is local data (an inline base64
  string or a bundled fixture) — ``image_base64``, ``audio_base64``,
  ``audio_local_file``. Each gets a scripted test that replays canned responses
  through ``fake(...)`` and runs with no API key, plus a live one parametrized
  over a model pool, which skips when no provider is configured. Tests asserting
  a *failure* are scripted only — a real model may legitimately answer
  correctly.
* **Live-only** tasks whose input needs the network (``image_url``,
  ``image_local_file``, ``video_url``, ``audio_url``). They get a live test
  only — faking them would exercise no real image/audio path — so they skip
  cleanly offline (pytest skips an empty parameter set).
"""

import base64
import contextlib
import os
import tempfile

import httpx
import pytest
from models import AUDIO_MODELS, VIDEO_MODELS, VISION_MODELS, fake

import kaggle_benchmarks as kbench
from kaggle_benchmarks.content_types import audios, images, videos

#: A short spoken-word clip, bundled next to this module in ``fixtures/``.
SPEECH_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "speech.mp3")
SPEECH_TRANSCRIPTION_PATTERN = r"(?i)quick\s+brown\s+fox|lazy\s+dog"

# Text-only models that don't accept image input via Model Proxy.
_KAGGLE_LOGO_URL = "https://www.kaggle.com/static/images/site-logo.png"


@contextlib.contextmanager
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


@kbench.task(name="image_base64")
def image_base64(llm):
    """Sends a base64 encoded image with explicit format specification."""
    # A 1x1 red pixel in PNG format.
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


def test_image_base64_scripted():
    assert image_base64.run(fake(["Red."])).passed


def test_image_base64_wrong_color_fails():
    assert not image_base64.run(fake(["Green."])).passed


@pytest.mark.parametrize("llm", VISION_MODELS)
def test_image_base64(llm):
    assert image_base64.run(llm).passed


@kbench.task(name="audio_base64")
def audio_base64(llm):
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


def test_audio_base64_scripted():
    assert audio_base64.run(
        fake(["The quick brown fox jumps over the lazy dog."])
    ).passed


@pytest.mark.parametrize("llm", AUDIO_MODELS)
def test_audio_base64(llm):
    assert audio_base64.run(llm).passed


@kbench.task(name="audio_local_file")
def audio_local_file(llm):
    """Sends a speech audio file loaded from disk and asks the model to transcribe it."""
    audio_content = audios.from_path(SPEECH_FIXTURE)

    response = llm.prompt("Transcribe this audio exactly.", audio=audio_content)

    kbench.assertions.assert_contains_regex(
        SPEECH_TRANSCRIPTION_PATTERN,
        response,
        expectation="LLM should transcribe the speech audio.",
    )


def test_audio_local_file_scripted():
    assert audio_local_file.run(
        fake(["The quick brown fox jumps over the lazy dog."])
    ).passed


@pytest.mark.parametrize("llm", AUDIO_MODELS)
def test_audio_local_file(llm):
    assert audio_local_file.run(llm).passed


@kbench.task(name="image_url")
def image_url(llm):
    """Sends an image URL directly to the model."""
    image = images.from_url(_KAGGLE_LOGO_URL)

    response = llm.prompt("What does this logo say?", image=image)

    kbench.assertions.assert_contains_regex(
        r"(?i)kaggle",
        response,
        expectation="LLM should identify the Kaggle logo.",
    )


@pytest.mark.parametrize("llm", VISION_MODELS)
def test_image_url(llm):
    assert image_url.run(llm).passed


@kbench.task(name="image_local_file")
def image_local_file(llm):
    """Downloads the logo to a temp file, then sends it from disk."""
    with download_temp_image(_KAGGLE_LOGO_URL, suffix=".png") as image_path:
        image = images.from_path(image_path)
        response = llm.prompt("What does this logo say?", image=image)

    kbench.assertions.assert_contains_regex(
        r"(?i)kaggle",
        response,
        expectation="LLM should identify the Kaggle logo.",
    )


@pytest.mark.parametrize("llm", VISION_MODELS)
def test_image_local_file(llm):
    assert image_local_file.run(llm).passed


@kbench.task(name="video_url")
def video_url(llm):
    """Sends a YouTube video URL to the model."""
    # Big Buck Bunny video.
    video = videos.from_url("https://www.youtube.com/watch?v=aqz-KE-bpKQ")

    response = llm.prompt("What is this video about? Describe it briefly.", video=video)

    kbench.assertions.assert_contains_regex(
        r"(?i)bunny|rabbit|animal",
        response,
        expectation="LLM should identify the Big Buck Bunny video content.",
    )


@pytest.mark.parametrize("llm", VIDEO_MODELS)
def test_video_url(llm):
    assert video_url.run(llm).passed


@kbench.task(name="audio_url")
def audio_url(llm):
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


@pytest.mark.parametrize("llm", AUDIO_MODELS)
def test_audio_url(llm):
    assert audio_url.run(llm).passed
