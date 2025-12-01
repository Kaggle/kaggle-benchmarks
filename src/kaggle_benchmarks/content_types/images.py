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

import abc
import base64
import io
import mimetypes

import httpx
import numpy as np
import panel as pn


class ImageContent(abc.ABC):
    @property
    @abc.abstractmethod
    def url(self) -> str: ...

    @abc.abstractmethod
    def to_mime(self) -> dict[str, str]:
        """Dumps the image as a MIME dictionary."""
        raise NotImplementedError

    def get_payload(self) -> list[dict[str, str | dict[str, str]]]:
        """Returns the image payload in a standardized dictionary format."""
        return [{"type": "image_url", "image_url": {"url": self.url}}]

    def _repr_markdown_(self) -> str:
        """Returns a Markdown representation of the image."""
        return f"![image]({self.url})"


class ImageURL(ImageContent):
    def __init__(self, url: str):
        self._url = url

    @property
    def url(self) -> str:
        return self._url

    def __panel__(self) -> pn.viewable.Viewable:
        """Renders the image using a Panel Image pane."""
        return pn.pane.image.Image(self.url)

    def to_mime(self) -> dict[str, str]:
        """Returns a MIME dictionary pointing to the image location."""
        return {
            "mime_type": mimetypes.guess_type(self.url)[0],
            "location": self.url,
        }


class ImageBase64(ImageContent):
    def __init__(self, b64_string: str, mime_type: str):
        self.b64_string = b64_string
        self.mime_type = mime_type

    @property
    def url(self) -> str:
        """Generates a data URL from the internal array on demand."""
        return f"data:{self.mime_type};base64,{self.b64_string}"

    def to_mime(self) -> dict[str, str]:
        """Returns a MIME dictionary with the inline Base64 content."""
        return {
            "mime_type": self.mime_type,
            "content": self.b64_string,
        }


def from_path(path: str) -> ImageBase64:
    """Creates ImageContent from a local image file path."""
    with open(path, "rb") as image_file:
        return ImageBase64(
            base64.b64encode(image_file.read()).decode(),
            mimetypes.guess_type(path)[0],
        )


def from_base64(base64: str | bytes, format: str = "jpeg") -> ImageBase64:
    """Creates ImageContent directly from a base64 string."""
    if isinstance(base64, bytes):
        base64 = base64.decode("utf-8")

    return ImageBase64(base64, mime_type=f"image/{format}")


def from_url(url: str) -> ImageURL:
    """Creates ImageContent from an image URL."""
    return ImageURL(url)


def from_array(array: np.ndarray) -> ImageBase64:
    """Creates ImageContent from an image array."""
    from PIL import Image

    pil_img = Image.fromarray(array)
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG")
    return ImageBase64(
        base64.b64encode(buff.getvalue()).decode(), mime_type="image/jpeg"
    )


def from_image_url(image_url: ImageURL) -> ImageBase64:
    """Creates ImageBase64 from an ImageURL, downloading and encoding it."""
    url_str = image_url.url
    full_mime_type = image_url.to_mime().get("mime_type")
    if full_mime_type:
        image_format = full_mime_type.split("/")[-1]
    else:
        image_format = "jpeg"
    b64_data = image_url_to_base64(url_str)
    return from_base64(b64_data, image_format)


def image_url_to_base64(url: str) -> str:
    """Load an image from its url to base64."""
    # Explicit User-Agent: https://meta.wikimedia.org/wiki/User-Agent_policy
    headers = {"User-Agent": "test"}
    client = httpx.Client()
    response = client.get(url, headers=headers)
    response.raise_for_status()
    return base64.b64encode(response.content).decode("utf-8")
