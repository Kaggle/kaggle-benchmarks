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

import tempfile

import pytest

from kaggle_benchmarks import tools


@pytest.mark.asyncio
async def test_snapshot():
    html_string = "<html><head><title>Test</title></head><body><h1>Hello</h1><script>console.log('hello')</script></body></html>"
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(f"{tmp_dir}/index.html", "w") as f:
            f.write(html_string)

        snapshot = await tools.web.async_take_snapshot(f"file://{tmp_dir}/index.html")
        assert html_string == snapshot.html
        assert ["hello"] == snapshot.logs


def test_browser_screenshot():
    with tools.web.Browser() as browser:
        response = browser.take_screenshot(
            "<html><head><title>Test</title></head><body><h1>Hello</h1></body></html>"
        )
        assert isinstance(response, tools.web.Image)


def test_browser_snapshot():
    with tools.web.Browser() as browser:
        response = browser.take_snapshot(
            "<html><head><title>Test</title></head><body><h1>Hello</h1></body></html>"
        )
        assert isinstance(response, tools.web.Snapshot)
