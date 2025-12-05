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

import difflib
import enum
import json
import os
import re
from pathlib import Path

import hishel
import hishel.httpx
import httpx
from openai import OpenAI, OpenAIError


class Status(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class JSONEncoder(json.encoder.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "name"):
            return f"{type(obj).__name__}: {obj.name}"
        if hasattr(obj, "__name__"):
            return f"{type(obj).__name__}: {obj.__name__}"
        return repr(obj)


class UnconditionalCacheTransport(httpx.HTTPTransport):
    def __init__(self, cache_timeout_seconds: int = 7 * 24 * 60 * 60) -> None:
        super().__init__()
        self.cache_timeout_seconds = cache_timeout_seconds

    def handle_request(self, request):
        response = super().handle_request(request)

        # Force the header to be cacheable for 1 week
        # This overrides whatever the server said (e.g. no-store)
        response.headers["Cache-Control"] = (
            f"public, max-age={self.cache_timeout_seconds}"
        )

        # Remove 'Expires' or 'Pragma' if they exist to avoid conflicts
        response.headers.pop("Pragma", None)
        response.headers.pop("Expires", None)

        return response


def build_httpx_client(filename: str = "cache") -> httpx.Client:
    from kaggle_benchmarks._config import config

    if config.disable_caching:
        return httpx.Client()

    return hishel.httpx.SyncCacheClient(
        transport=hishel.httpx.SyncCacheTransport(
            next_transport=UnconditionalCacheTransport(
                cache_timeout_seconds=config.cache_timeout_seconds
            ),
        ),
        storage=hishel.SyncSqliteStorage(
            database_path=config.cache_directory / f"{filename}.sqlite",
        ),
    )


def extract_code_block(text: str, name: str | None = None, greedy: bool = True) -> str:
    """Extracts a code block (if exist) from a markdown string."""
    name = name or r"\w*"
    matches = re.findall(
        f"```{name}(.+{'' if greedy else '?'})```", text, flags=re.DOTALL
    )
    if matches:
        return matches[0]
    return text


def string_diff_as_markdown(str1: str, str2: str, sep: str = "\n") -> str:
    differ = difflib.Differ()
    diff = sep.join(
        differ.compare(str1.splitlines(keepends=True), str2.splitlines(keepends=True))
    )

    return f"""```diff
{diff}
```"""


def highlight_matched_spans(
    text: str, pattern: str, highlight_color: str = "yellow"
) -> str:
    def replacer(match):
        matched_text = match.group(0)
        return (
            f'<span style="background-color: {highlight_color};">{matched_text}</span>'
        )

    return re.sub(pattern, replacer, text)


def normalize_name(name: str) -> str:
    """Replaces special characters such as spaces and slashes in a name."""
    name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return re.sub(r"[^a-zA-Z0-9_.-]", "", name)


def _get_model_proxy_env_vars() -> tuple[str, str]:
    api_key = os.environ.get("MODEL_PROXY_API_KEY")
    base_url = os.environ.get("MODEL_PROXY_URL")
    if not api_key or not base_url:
        raise ValueError(
            "MODEL_PROXY_API_KEY and MODEL_PROXY_URL must be set in environment variables."
        )
    return api_key, base_url


def get_openai_client() -> OpenAI:
    api_key, base_url = _get_model_proxy_env_vars()
    return OpenAI(api_key=api_key, base_url=base_url)


def prompt_llm_with_openai_api(prompt: str, model: str, client: OpenAI) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return response.choices[0].message.content or ""
    except OpenAIError as e:
        print(f"LLM call for task_autopilot failed: {e}")
        raise e


def task_autopilot(
    task_description: str,
    assertion_description: str,
    model: str = "google/gemini-2.5-pro",
) -> str:
    """
    Generate source code for a Kaggle benchmark task based on user inputs.
    """
    try:
        base_dir = Path(__file__).parent.parent.parent
        quick_start_prompt = (base_dir / "quick_start.md").read_text()
        user_guide_prompt = (base_dir / "user_guide.md").read_text()
    except FileNotFoundError:
        print(
            "Could not find quick_start.md for task_autopilot. "
            "The generated code may be suboptimal."
        )
        quick_start_prompt = (
            "<!-- The quick start guide was not found, but you should still "
            "generate code based on the user's request and your knowledge of the SDK. -->"
        )
        user_guide_prompt = (
            "<!-- The user guide was not found, but you should still "
            "generate code based on the user's request and your knowledge of the SDK. -->"
        )

    prompt = f"""
You are an expert Python developer specializing in the Kaggle Benchmarks SDK.
Your task is to write a complete, self-contained Python script based on a user's request.

Follow these rules:
1.  Define a task using the `@kbench.task()` decorator. The task name should be a short, descriptive string in quotes, derived from the user's request (e.g., "count 'r's in strawberry").
2.  The function signature for the task should be simple, usually just `def your_task_name(llm):`. If it returns value, specify the return type.
3.  Use `kbench.assertions` to check for correctness based on the user's assertion criteria.
4.  The script must be complete and runnable. End the script with a call to run the task, like `your_task_name.run(kbench.llm)`.
5.  Do not include any placeholders for the user to fill in; the script should be final.
6.  Only output the raw Python code, without any surrounding text, explanations, or markdown fences like ````python.
7.  Add a comment at the beginning of the task definition indicating this auto-generated code. User should use wth cautions.

========== Here is a quick start guide of the library: ==========
{quick_start_prompt}
==================================================================

========== Here is a user guide of the library: ==========
{user_guide_prompt}
==================================================================

Here are the user inputs

**Task Description:**
{task_description}

**Assertion Criteria:**
{assertion_description}

---
### GENERATED PYTHON SCRIPT:
    """

    client = get_openai_client()
    generated_code = prompt_llm_with_openai_api(prompt, model=model, client=client)
    return extract_code_block(generated_code, name="python", greedy=False)
