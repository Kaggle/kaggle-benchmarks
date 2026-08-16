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

import contextlib
import tempfile

import pytest

from kaggle_benchmarks import tools


@contextlib.contextmanager
def _skip_if_browser_unavailable():
    """Skips (instead of failing) when the Playwright browser isn't installed.

    The ``playwright`` package is a dependency, but its Chromium browser is a
    separate download (``playwright install chromium``). Without it, launching
    raises; re-raise anything else so real failures still surface.
    """
    try:
        yield
    except Exception as e:  # noqa: BLE001 - narrowed by the message check below
        message = str(e).lower()
        if "executable doesn't exist" in message or "playwright install" in message:
            pytest.skip("Playwright browser is not installed")
        raise


@pytest.mark.asyncio
async def test_snapshot():
    html_string = "<html><head><title>Test</title></head><body><h1>Hello</h1><script>console.log('hello')</script></body></html>"
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(f"{tmp_dir}/index.html", "w") as f:
            f.write(html_string)

        with _skip_if_browser_unavailable():
            snapshot = await tools.web.async_take_snapshot(
                f"file://{tmp_dir}/index.html"
            )
        assert html_string == snapshot.html
        assert ["hello"] == snapshot.logs


def test_browser_screenshot():
    with _skip_if_browser_unavailable(), tools.web.Browser() as browser:
        response = browser.take_screenshot(
            "<html><head><title>Test</title></head><body><h1>Hello</h1></body></html>"
        )
        assert isinstance(response, tools.web.Image)


def test_browser_snapshot():
    with _skip_if_browser_unavailable(), tools.web.Browser() as browser:
        response = browser.take_snapshot(
            "<html><head><title>Test</title></head><body><h1>Hello</h1></body></html>"
        )
        assert isinstance(response, tools.web.Snapshot)
