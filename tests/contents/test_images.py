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

import base64
import pathlib

import httpx
import numpy as np
import pytest
from PIL import Image

from kaggle_benchmarks.content_types import images

# A simple 1x1 red pixel base64 encoded string
B64_STRING = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def test_from_url():
    url = "https://example.com/image.jpg"

    img = images.from_url(url)
    assert isinstance(img, images.ImageContent)
    assert img.url == url
    assert img.to_mime() == {"mime_type": "image/jpeg", "location": url}
    assert not img.caption

    img_with_caption = images.from_url(url, caption="A test image.")
    assert img_with_caption.caption == "A test image."


def test_from_base64():
    img = images.from_base64(B64_STRING, format="png")
    assert isinstance(img, images.ImageBase64)
    assert img.b64_string == B64_STRING
    assert img.mime_type == "image/png"
    assert not img.caption

    img_with_caption = images.from_base64(
        B64_STRING, format="png", caption="A test image."
    )
    assert img_with_caption.caption == "A test image."


def test_from_array():
    array = np.zeros((20, 20, 3), dtype=np.uint8)
    img = images.from_array(array)
    assert isinstance(img, images.ImageBase64)
    assert img.mime_type == "image/jpeg"
    assert isinstance(base64.b64decode(img.b64_string), bytes)


def test_from_path(tmp_path: pathlib.Path):
    file_path = tmp_path / "test_image.png"
    Image.fromarray(np.ones((5, 5, 3), dtype=np.uint8) * 255).save(file_path, "PNG")

    temp_png_path = str(file_path)
    img_content = images.from_path(temp_png_path)
    assert isinstance(img_content, images.ImageBase64)
    assert img_content.mime_type == "image/png"
    assert len(img_content.b64_string) > 0


def test_image_base64_properties():
    img = images.ImageBase64(B64_STRING, mime_type="image/png")
    assert img.b64_string == B64_STRING
    assert img.mime_type == "image/png"
    assert img.url == f"data:image/png;base64,{B64_STRING}"
    assert not img.caption

    expected = {"mime_type": "image/png", "content": B64_STRING}
    assert img.to_mime() == expected

    img_with_caption = images.ImageBase64(
        B64_STRING, mime_type="image/png", caption="A test image."
    )
    assert img_with_caption.caption == "A test image."


def test_from_image_url(mocker):
    """Tests creating ImageBase64 from ImageURL."""
    url = "https://example.com/image.png"
    image_url = images.ImageURL(url)

    # Mock the internal function that does the download and encoding
    mock_image_url_to_base64 = mocker.patch(
        "kaggle_benchmarks.content_types.images.image_url_to_base64",
        return_value=B64_STRING,
    )

    img_base64 = images.from_image_url(image_url)

    mock_image_url_to_base64.assert_called_once_with(url)
    assert isinstance(img_base64, images.ImageBase64)
    assert img_base64.b64_string == B64_STRING
    assert img_base64.mime_type == "image/png"


def test_image_url_to_base64_success(mocker):
    """Tests successful fetching and base64 encoding of an image from a URL."""
    # The content of the mock response will be the raw bytes of our test image.
    image_bytes = base64.b64decode(B64_STRING)
    mock_response = mocker.Mock()
    mock_response.content = image_bytes
    mock_response.raise_for_status.return_value = None  # Simulate a successful request

    # Patch 'requests.get' to return our mock response
    mock_get = mocker.patch(
        "kaggle_benchmarks.content_types.images.httpx.Client.get",
        return_value=mock_response,
    )

    url = "https://example.com/image.png"
    result = images.image_url_to_base64(url)

    # Verify that requests.get was called correctly
    mock_get.assert_called_once_with(url, headers={"User-Agent": "test"})

    # Verify the result is the expected base64 string
    assert result == B64_STRING


def test_image_url_to_base64_http_error(mocker):
    """Tests that an HTTPError from requests is propagated."""
    # Configure the mock to raise an HTTPError when raise_for_status is called
    mock_response = mocker.Mock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error", request=mocker.Mock(), response=mock_response
    )

    # Patch 'requests.get' to return our mock response
    mock_get = mocker.patch(
        "kaggle_benchmarks.content_types.images.httpx.Client.get",
        return_value=mock_response,
    )

    url = "https://example.com/notfound.png"

    # Verify that the HTTPError is raised
    with pytest.raises(httpx.HTTPStatusError):
        images.image_url_to_base64(url)

    # Verify that requests.get was still called
    mock_get.assert_called_once_with(url, headers={"User-Agent": "test"})
