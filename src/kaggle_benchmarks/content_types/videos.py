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

import abc
import mimetypes
import re
from typing import Any

_YOUTUBE_URL_PATTERN = re.compile(
    r"^https?://(?:www\.)?(?:youtube\.com/(?:watch\?v=|shorts/)|youtu\.be/)[\w-]+"
)


class VideoContent(abc.ABC):
    def __init__(self, overwrite_api_params: dict[str, Any] | None = None):
        self.overwrite_api_params = overwrite_api_params or {}

    @property
    @abc.abstractmethod
    def url(self) -> str: ...

    @property
    @abc.abstractmethod
    def mime_type(self) -> str: ...

    def get_payload(self) -> list[dict[str, str | dict[str, str]]]:
        """Returns the video payload for the OpenAI Chat Completions API format.

        The OpenAI Chat Completions spec has no dedicated video content part type,
        so we use `image_url` as a generic file URL carrier.
        """
        return [{"type": "image_url", "image_url": {"url": self.url}}]


class VideoURL(VideoContent):
    def __init__(self, url: str, overwrite_api_params: dict[str, Any] | None = None):
        super().__init__(overwrite_api_params=overwrite_api_params)
        self._url = url

    @property
    def url(self) -> str:
        return self._url

    def __panel__(self):
        """Renders the video as a clickable link."""
        import panel as pn

        return pn.pane.HTML(f'<a href="{self.url}" target="_blank">{self.url}</a>')

    @property
    def mime_type(self) -> str:
        return mimetypes.guess_type(self.url)[0] or "video/*"

    def to_mime(self) -> dict[str, str]:
        return {
            "mime_type": self.mime_type,
            "location": self.url,
        }


def from_url(url: str, overwrite_api_params: dict[str, Any] | None = None) -> VideoURL:
    """Creates VideoContent from a video URL (e.g. a YouTube link).

    Currently only YouTube URLs are supported.
    """
    if not _YOUTUBE_URL_PATTERN.match(url):
        raise ValueError(
            f"Unsupported video URL: {url}\n"
            "Only YouTube URLs are currently supported "
            "(e.g. https://www.youtube.com/watch?v=aqz-KE-bpKQ)."
        )
    return VideoURL(url, overwrite_api_params=overwrite_api_params)
