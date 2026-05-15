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
import warnings

import openai
from google import genai
from google.genai import types

from kaggle_benchmarks import utils
from kaggle_benchmarks.actors.llms import GoogleGenAI, LLMChat, OpenAI


def validate_model_proxy_config(
    url: str | None = None,
    api_key: str | None = None,
    raise_on_error: bool = False,
) -> None:
    """Warn (or raise) if required auth env vars are missing."""
    missing = []
    if not url:
        missing.append("MODEL_PROXY_URL")
    if not api_key:
        missing.append("MODEL_PROXY_API_KEY")
    if not missing:
        return

    install_command = "pip install kaggle"
    auth_command = "kaggle benchmarks auth"
    missing_list = "\n".join(f"  - {v}" for v in missing)
    separator = "-" * len(auth_command)
    msg = (
        f"\n\nMissing environment variables for Kaggle authentication:\n\n{missing_list}\n\n"
        f"Authenticate by running:\n{separator}\n{install_command}\n{auth_command}\n{separator}\n"
    )

    if raise_on_error:
        raise ValueError(msg)
    warnings.warn(msg, stacklevel=2)


class ModelProxy:
    def __new__(
        cls,
        model: str,
        api: str = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ) -> LLMChat:
        resolved_api_key = api_key or os.getenv("MODEL_PROXY_API_KEY")
        resolved_base_url = base_url or os.getenv("MODEL_PROXY_URL")

        validate_model_proxy_config(
            url=resolved_base_url, api_key=resolved_api_key, raise_on_error=True
        )

        # Normalize base URL
        for suffix in ("/openapi", "/genai"):
            if resolved_base_url.endswith(suffix):
                resolved_base_url = resolved_base_url[: -len(suffix)]
                break

        llm_instance = None
        # Qwen and DeepSeek models support response_format, but the schema must be under 64 characters.
        kwargs.setdefault(
            "support_structured_outputs",
            "meta" not in model
            and "qwen" not in model
            and "deepseek" not in model
            and "gemma" not in model,
        )

        if api == "genai":
            resolved_base_url = resolved_base_url + "/genai"
            client = genai.Client(
                api_key=resolved_api_key,
                http_options=types.HttpOptions(
                    api_version="v1",
                    base_url=resolved_base_url,
                ),
            )
            llm_instance = GoogleGenAI(client, model, **kwargs)

        elif api == "openai":
            resolved_base_url = resolved_base_url + "/openapi"
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
