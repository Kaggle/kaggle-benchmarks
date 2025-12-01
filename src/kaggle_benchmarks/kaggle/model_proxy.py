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

import os
import re

import openai
from google import genai
from google.genai import types

from kaggle_benchmarks import utils
from kaggle_benchmarks.actors.llms import GoogleGenAI, LLMChat, OpenAI


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
        llm_instance = None

        if api == "genai":
            if not resolved_base_url:
                raise ValueError(
                    "MODEL_PROXY_URL must be set via parameter or environment variable."
                )
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

            kwargs.setdefault("support_structured_outputs", "meta" not in model)
            # TODO (b/439876083): Disable temperature parameter till this is resolved.
            kwargs.setdefault("support_temperature", False)
            llm_instance = OpenAI(client, model, **kwargs)

        else:
            raise ValueError(f"Unsupported API: '{api}'. Must be 'openai' or 'genai'.")

        if llm_instance:
            llm_instance.stream_responses = False
        return llm_instance
