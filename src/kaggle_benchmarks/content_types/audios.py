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

import base64
import mimetypes
from typing import Any

import httpx


class AudioContent:
    def __init__(
        self,
        b64_string: str,
        mime_type: str,
        caption: str | None = None,
        extra_api_params: dict[str, Any] | None = None,
    ):
        self._b64_string = b64_string
        self._mime_type = mime_type
        self.caption = caption
        self.extra_api_params = dict(extra_api_params) if extra_api_params else {}

    @property
    def b64_string(self) -> str:
        return self._b64_string

    @property
    def mime_type(self) -> str:
        return self._mime_type

    @property
    def url(self) -> str:
        """Returns the data URL for this audio content."""
        return f"data:{self.mime_type};base64,{self.b64_string}"

    @property
    def _format(self) -> str:
        """Returns the short format name (e.g. 'mp3', 'wav') from the MIME type."""
        ext = mimetypes.guess_extension(self.mime_type)
        if ext:
            return ext.lstrip(".")
        return self.mime_type.split("/")[-1].removeprefix("x-")

    def to_mime(self) -> dict[str, str]:
        return {
            "mime_type": self.mime_type,
            "content": self.b64_string,
        }

    def _repr_markdown_(self) -> str:
        """Returns an HTML audio player for notebook rendering."""
        html = f'<audio controls src="{self.url}"></audio>'
        if self.caption:
            html += f"\n\n{self.caption}"
        return html

    def __panel__(self):
        """Renders the audio using a Panel HTML pane."""
        import panel as pn

        html = f'<audio controls src="{self.url}"></audio>'
        if self.caption:
            html += f"<p>{self.caption}</p>"
        return pn.pane.HTML(html)


def from_url(
    url: str,
    caption: str | None = None,
    timeout: float = 30.0,
    extra_api_params: dict[str, Any] | None = None,
) -> AudioContent:
    """Creates AudioContent from a URL by fetching and base64-encoding the audio."""
    try:
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise ValueError(f"Failed to fetch audio from URL: {url}") from e
    mime_type = (
        response.headers.get("content-type", "").split(";")[0]
        or mimetypes.guess_type(url)[0]
        or "audio/*"
    )
    return AudioContent(
        base64.b64encode(response.content).decode(),
        mime_type,
        caption=caption,
        extra_api_params=extra_api_params,
    )


def from_path(
    path: str,
    caption: str | None = None,
    extra_api_params: dict[str, Any] | None = None,
) -> AudioContent:
    """Creates AudioContent from a local audio file path."""
    with open(path, "rb") as audio_file:
        return AudioContent(
            base64.b64encode(audio_file.read()).decode(),
            mimetypes.guess_type(path)[0] or "audio/*",
            caption=caption,
            extra_api_params=extra_api_params,
        )


def from_base64(
    b64_string: str | bytes,
    format: str = "mp3",
    caption: str | None = None,
    extra_api_params: dict[str, Any] | None = None,
) -> AudioContent:
    """Creates AudioContent directly from a base64 string."""
    if isinstance(b64_string, bytes):
        b64_string = b64_string.decode("utf-8")
    try:
        base64.b64decode(b64_string, validate=True)
    except ValueError as e:
        raise ValueError(f"Invalid base64 audio data: {e}") from e
    return AudioContent(
        b64_string,
        mime_type=f"audio/{format}",
        caption=caption,
        extra_api_params=extra_api_params,
    )
