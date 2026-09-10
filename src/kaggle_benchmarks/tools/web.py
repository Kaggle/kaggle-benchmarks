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

import asyncio
import base64
import dataclasses
import re

from kaggle_benchmarks import actors, chats
from kaggle_benchmarks.envs import InternalUnsafeLocalEnvironment

_MISSING_DEPS_MESSAGE = (
    "The web browser tool requires additional dependencies. Install them with:\n"
    "    uv pip install kaggle-benchmarks[web-tools]"
)


def _require_playwright():
    try:
        import nest_asyncio
        from playwright.async_api import async_playwright
    except ImportError as e:
        raise ImportError(_MISSING_DEPS_MESSAGE) from e

    # required as jupyter runs its own asyncio loop
    nest_asyncio.apply()
    return async_playwright


def extract_html(html_string: str) -> str:
    if not html_string:
        return ""

    match = re.search(r"<html.*?>([\s\S]*)</html>", html_string, re.IGNORECASE)

    if match is None:
        return ""

    return match.group(0)


@dataclasses.dataclass
class Image:
    extension: str
    b64: str
    text: str

    def _repr_markdown_(self):
        return f'![{self.text}](data:image/{self.extension};base64,{self.b64} "{self.text}")'


@dataclasses.dataclass
class Snapshot:
    screenshot: Image
    html: str
    logs: list[str] = dataclasses.field(default_factory=list)

    def _repr_markdown_(self):
        logs = "\n\n".join(self.logs)
        if logs.strip():
            logs = f"Console:\n```\n{logs}\n```"

        return f"{self.screenshot._repr_markdown_()}\n\n{logs}"


async def async_take_snapshot(
    url: str,
    full_page: bool = False,
    wait_after_load_ms: int = 0,
    width: int = 400,
    height: int = 600,
) -> Snapshot:
    async_playwright = _require_playwright()
    logs = []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_viewport_size({"width": width, "height": height})
        page.on("console", lambda msg: logs.append(msg.text))
        await page.goto(url)

        if wait_after_load_ms > 0:
            await page.wait_for_timeout(wait_after_load_ms)

        screenshot_bytes = await page.screenshot(full_page=full_page)
        screenshot = Image(
            extension="png",
            text="screenshot",
            b64=base64.b64encode(screenshot_bytes).decode(),
        )
        html = await page.content()
        await browser.close()

    return Snapshot(screenshot=screenshot, html=html, logs=logs)


class Browser(actors.Actor):
    def __init__(self):
        _require_playwright()
        super().__init__(role="tool", avatar="🌐", name="Browser")
        self.env = InternalUnsafeLocalEnvironment(_internal=True)

    def __enter__(self):
        return self

    def _take_snapshot(self, page: str, wait_before: int = 0, **kwargs) -> Snapshot:
        if not page.startswith("http"):
            self.env["page.html"] = page
            page = f"file://{self.env.directory / 'page.html'}"

        return asyncio.run(
            async_take_snapshot(
                url=page,
                wait_after_load_ms=wait_before,
                **kwargs,
            )
        )

    @chats.emits_message
    def take_screenshot(self, page: str, wait_before: int = 0, **kwargs) -> Image:
        return self._take_snapshot(page, wait_before, **kwargs).screenshot

    @chats.emits_message
    def take_snapshot(self, page: str, wait_before: int = 0, **kwargs) -> Snapshot:
        return self._take_snapshot(page, wait_before, **kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        pass
