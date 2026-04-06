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
import pathlib
import wave

import numpy as np

from kaggle_benchmarks.content_types import audio

# A trivial base64 string for testing
B64_STRING = base64.b64encode(b"\x00\x01\x02\x03").decode()


def test_from_url(mocker):
    audio_bytes = b"\xff\xfb\x90\x00"
    mock_response = mocker.Mock()
    mock_response.content = audio_bytes
    mock_response.headers = {"content-type": "audio/mpeg"}
    mock_response.raise_for_status.return_value = None

    mocker.patch(
        "kaggle_benchmarks.content_types.audio.httpx.get",
        return_value=mock_response,
    )

    a = audio.from_url("https://example.com/speech.mp3")
    assert isinstance(a, audio.AudioContent)
    assert a.mime_type == "audio/mpeg"
    assert base64.b64decode(a.b64_string) == audio_bytes


def test_from_url_with_caption(mocker):
    mock_response = mocker.Mock()
    mock_response.content = b"\x00"
    mock_response.headers = {"content-type": "audio/mpeg"}
    mock_response.raise_for_status.return_value = None
    mocker.patch(
        "kaggle_benchmarks.content_types.audio.httpx.get",
        return_value=mock_response,
    )

    a = audio.from_url("https://example.com/speech.mp3", caption="A speech clip.")
    assert a.caption == "A speech clip."


def test_from_base64():
    a = audio.from_base64(B64_STRING, format="mp3")
    assert isinstance(a, audio.AudioContent)
    assert a.b64_string == B64_STRING
    assert a.mime_type == "audio/mp3"
    assert a.caption is None


def test_from_base64_bytes():
    a = audio.from_base64(B64_STRING.encode(), format="wav")
    assert isinstance(a, audio.AudioContent)
    assert a.b64_string == B64_STRING
    assert a.mime_type == "audio/wav"


def test_url():
    a = audio.AudioContent(B64_STRING, mime_type="audio/mp3")
    assert a.url == f"data:audio/mp3;base64,{B64_STRING}"


def test_get_payload():
    a = audio.AudioContent(B64_STRING, mime_type="audio/mp3")
    payload = a.get_payload()
    assert payload == [
        {"type": "input_audio", "input_audio": {"data": B64_STRING, "format": "mp3"}}
    ]


def test_from_path(tmp_path: pathlib.Path):
    file_path = tmp_path / "test_audio.mp3"
    file_path.write_bytes(b"\xff\xfb\x90\x00")

    a = audio.from_path(str(file_path))
    assert isinstance(a, audio.AudioContent)
    assert a.mime_type == "audio/mpeg"
    assert base64.b64decode(a.b64_string) == b"\xff\xfb\x90\x00"


def test_repr_markdown():
    a = audio.AudioContent(B64_STRING, mime_type="audio/mp3")
    md = a._repr_markdown_()
    assert "<audio controls" in md
    assert a.url in md


def test_repr_markdown_with_caption():
    a = audio.AudioContent(B64_STRING, mime_type="audio/mp3", caption="My audio")
    md = a._repr_markdown_()
    assert "My audio" in md
    assert "<audio controls" in md
