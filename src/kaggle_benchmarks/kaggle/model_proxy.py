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

import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import openai
from google import genai
from google.genai import types

from kaggle_benchmarks import utils
from kaggle_benchmarks.actors.llms import GoogleGenAI, LLMChat, OpenAI

_BASE_DIR = Path(__file__).parent.parent.parent.parent
_KAGGLE_INSTALL_COMMAND = "pip install kaggle"
_KAGGLE_AUTH_COMMAND = f"cd {_BASE_DIR} && kaggle benchmarks auth"


def _validate_proxy_config(
    url: str | None = None,
    api_key: str | None = None,
) -> None:
    """Raise if required auth env vars are missing."""
    missing = []
    if not url:
        missing.append("MODEL_PROXY_URL")
    if not api_key:
        missing.append("MODEL_PROXY_API_KEY")
    if not missing:
        return

    missing_list = "\n".join(f"  - {v}" for v in missing)
    separator = "-" * len(_KAGGLE_AUTH_COMMAND)
    raise ValueError(
        f"\n\nMissing environment variables for Kaggle authentication:\n\n{missing_list}\n\n"
        f"Authenticate by running:\n{separator}\n{_KAGGLE_INSTALL_COMMAND}\n{_KAGGLE_AUTH_COMMAND}\n{separator}\n"
    )


def _validate_proxy_token_expiry(
    expiry_time: str | None = None,
) -> None:
    """Warn if the auth token has expired."""
    # Kaggle notebook-based flows don't set MODEL_PROXY_EXPIRY_TIME, so skip the check.
    if not expiry_time:
        return

    try:
        expiry = datetime.fromisoformat(expiry_time)
    except (ValueError, TypeError):
        return

    if expiry <= datetime.now(timezone.utc):
        separator = "-" * len(_KAGGLE_AUTH_COMMAND)
        warnings.warn(
            "\n\nKaggle authentication has expired. Re-authenticate by running:\n"
            f"{separator}\n{_KAGGLE_INSTALL_COMMAND}\n{_KAGGLE_AUTH_COMMAND}\n{separator}\n",
            stacklevel=2,
        )


class ModelProxy:
    def __new__(
        cls,
        model: str,
        api: str = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
        expiry_time: str | None = None,
        **kwargs,
    ) -> LLMChat:
        resolved_api_key = api_key or os.getenv("MODEL_PROXY_API_KEY")
        resolved_base_url = base_url or os.getenv("MODEL_PROXY_URL")
        resolved_expiry_time = expiry_time or os.getenv("MODEL_PROXY_EXPIRY_TIME")

        _validate_proxy_config(url=resolved_base_url, api_key=resolved_api_key)
        _validate_proxy_token_expiry(expiry_time=resolved_expiry_time)

        llm_instance = None
        # Qwen and DeepSeek models support response_format, but the schema must be under 64 characters.
        kwargs.setdefault(
            "support_structured_outputs",
            "meta" not in model and "qwen" not in model and "deepseek" not in model,
        )

        if api == "genai":
            resolved_base_url = re.sub(r"/openapi", "/genai", resolved_base_url)
            client = genai.Client(
                api_key=resolved_api_key,
                http_options=types.HttpOptions(
                    api_version="v1",
                    base_url=resolved_base_url,
                ),
            )
            llm_instance = GoogleGenAI(client, model, **kwargs)

        elif api == "openai":
            client = openai.OpenAI(
                api_key=resolved_api_key,
                base_url=resolved_base_url,
                http_client=utils.build_httpx_client("model_proxy"),
            )

            # TODO (b/439876083): Disable temperature parameter till this is resolved.
            kwargs.setdefault("support_temperature", False)
            llm_instance = OpenAI(client, model, **kwargs)

        else:
            raise ValueError(f"Unsupported API: '{api}'. Must be 'openai' or 'genai'.")

        if llm_instance:
            llm_instance.stream_responses = False
        return llm_instance
