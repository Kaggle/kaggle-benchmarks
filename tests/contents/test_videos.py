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

import pytest

from kaggle_benchmarks.content_types import videos


def test_from_url():
    url = "https://www.youtube.com/watch?v=abc123"

    video = videos.from_url(url)
    assert isinstance(video, videos.VideoContent)
    assert isinstance(video, videos.VideoURL)
    assert video.url == url
    assert video.mime_type == "video/*"


def test_from_url_rejects_non_youtube():
    with pytest.raises(ValueError, match="Only YouTube URLs are currently supported"):
        videos.from_url("https://example.com/video.mp4")


def test_video_url_to_mime():
    url = "https://www.youtube.com/watch?v=abc123"
    video = videos.VideoURL(url)
    assert video.to_mime() == {"mime_type": "video/*", "location": url}


def test_video_url_get_payload():
    url = "https://www.youtube.com/watch?v=abc123"
    video = videos.VideoURL(url)
    payload = video.get_payload()
    assert payload == [{"type": "image_url", "image_url": {"url": url}}]


def test_extras_stored():
    url = "https://www.youtube.com/watch?v=abc123"
    video = videos.from_url(url, video_metadata={"fps": 1.0})
    assert video.api_params == {"video_metadata": {"fps": 1.0}}


def test_extras_default_empty():
    url = "https://www.youtube.com/watch?v=abc123"
    video = videos.from_url(url)
    assert video.api_params == {}
