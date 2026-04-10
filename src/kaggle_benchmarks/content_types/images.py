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
import functools
import io
import mimetypes

import httpx
import numpy as np


class ImageContent(abc.ABC):
    def __init__(self, caption: str | None = None, **kwargs):
        self.caption = caption
        self.api_params = kwargs

    @property
    @abc.abstractmethod
    def url(self) -> str: ...

    @property
    @abc.abstractmethod
    def b64_string(self) -> str: ...

    @property
    @abc.abstractmethod
    def mime_type(self) -> str: ...

    @abc.abstractmethod
    def to_mime(self) -> dict[str, str]:
        """Dumps the image as a MIME dictionary."""
        raise NotImplementedError

    def get_payload(self) -> list[dict[str, str | dict[str, str]]]:
        """Returns the image payload in a standardized dictionary format."""
        return [{"type": "image_url", "image_url": {"url": self.url}}]

    def _repr_markdown_(self) -> str:
        """Returns a Markdown representation of the image."""
        return f"![image]({self.url})\n\n{self.caption}"


class ImageURL(ImageContent):
    def __init__(self, url: str, caption: str | None = None, **kwargs):
        super().__init__(caption=caption, **kwargs)
        self._url = url

    @property
    def url(self) -> str:
        return self._url

    def __panel__(self):
        """Renders the image using a Panel Image pane."""
        import panel as pn

        return pn.pane.image.Image(self.url)

    @property
    def mime_type(self) -> str:
        return mimetypes.guess_type(self.url)[0]

    def to_mime(self) -> dict[str, str]:
        """Returns a MIME dictionary pointing to the image location."""
        return {
            "mime_type": self.mime_type,
            "location": self.url,
        }

    @functools.cached_property
    def b64_string(self) -> str:
        return image_url_to_base64(self.url)


class ImageBase64(ImageContent):
    def __init__(self, b64_string: str, mime_type: str, caption: str | None = None, **kwargs):
        super().__init__(caption=caption, **kwargs)
        self._b64_string = b64_string
        self._mime_type = mime_type

    @property
    def b64_string(self) -> str:
        return self._b64_string

    @property
    def mime_type(self) -> str:
        return self._mime_type

    @property
    def url(self) -> str:
        return f"data:{self.mime_type};base64,{self.b64_string}"

    def to_mime(self) -> dict[str, str]:
        """Returns a MIME dictionary with the inline Base64 content."""
        return {
            "mime_type": self.mime_type,
            "content": self.b64_string,
        }


def from_path(path: str, **kwargs) -> ImageBase64:
    """Creates ImageContent from a local image file path."""
    with open(path, "rb") as image_file:
        return ImageBase64(
            base64.b64encode(image_file.read()).decode(),
            mimetypes.guess_type(path)[0],
            **kwargs,
        )


def from_url(url: str, caption: str | None = None, **kwargs) -> ImageURL:
    """Creates ImageContent from an image URL."""
    return ImageURL(url, caption=caption, **kwargs)


def from_base64(
    base64: str | bytes, format: str = "jpeg", caption: str | None = None, **kwargs
) -> ImageBase64:
    """Creates ImageContent directly from a base64 string."""
    if isinstance(base64, bytes):
        base64 = base64.decode("utf-8")

    return ImageBase64(base64, mime_type=f"image/{format}", caption=caption, **kwargs)


def from_array(array: np.ndarray, **kwargs) -> ImageBase64:
    """Creates ImageContent from an image array."""
    from PIL import Image

    pil_img = Image.fromarray(array)
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG")
    return ImageBase64(
        base64.b64encode(buff.getvalue()).decode(), mime_type="image/jpeg", **kwargs
    )


def from_image_url(image_url: ImageURL) -> ImageBase64:
    """Creates ImageBase64 from an ImageURL, downloading and encoding it."""
    return ImageBase64(
        image_url.b64_string, image_url.mime_type, caption=image_url.caption,
        **image_url.api_params,
    )


def image_url_to_base64(url: str) -> str:
    """Load an image from its url to base64."""
    # Explicit User-Agent: https://meta.wikimedia.org/wiki/User-Agent_policy
    headers = {"User-Agent": "MyImageDownloader/1.0 (myemail@example.com)"}
    client = httpx.Client()
    response = client.get(url, headers=headers)
    response.raise_for_status()
    return base64.b64encode(response.content).decode("utf-8")
