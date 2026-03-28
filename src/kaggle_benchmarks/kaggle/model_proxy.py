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


### CHANGE 1: Added a human-readable error message constant.
### When Kaggle's backend rejects a request due to geo-restriction or account
### eligibility, the raw openai.PermissionDeniedError gives users no useful
### context. This string replaces that with actionable steps and links to the
### known GitHub issues (#85, #96) so users know it's a backend problem,
### not something wrong with their code.
_LOCATION_ERROR_MSG = (
    "Kaggle's model proxy rejected this request because your account or region "
    "is not currently supported for this model/API.\n\n"
    "This is a Kaggle backend restriction — it is not caused by your code.\n\n"
    "Steps to try:\n"
    "  1. Check https://github.com/Kaggle/kaggle-benchmarks/issues for known outages.\n"
    "  2. Verify your Kaggle account is eligible for model access at kaggle.com/settings.\n"
    "  3. If the issue is widespread, follow issue #85 or #96 for updates.\n\n"
    "Original error: {original}"
)


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

        # Qwen and DeepSeek models support response_format, but the schema must be under 64 characters.
        kwargs.setdefault(
            "support_structured_outputs",
            "meta" not in model and "qwen" not in model and "deepseek" not in model,
        )

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

            # TODO (b/439876083): Disable temperature parameter till this is resolved.
            kwargs.setdefault("support_temperature", False)

            ### CHANGE 2: Swapped OpenAI(...) for _LocationAwareOpenAI(...).
            ### This is a one-word change. Instead of instantiating the base
            ### OpenAI actor directly, we now use our thin subclass below that
            ### intercepts PermissionDeniedError and re-raises it with the
            ### helpful message from _LOCATION_ERROR_MSG.
            llm_instance = _LocationAwareOpenAI(client, model, **kwargs)

        else:
            raise ValueError(f"Unsupported API: '{api}'. Must be 'openai' or 'genai'.")

        if llm_instance:
            llm_instance.stream_responses = False
        return llm_instance


### CHANGE 3: Added _LocationAwareOpenAI subclass.
### Inherits everything from the base OpenAI actor unchanged. The only
### override is _request(), which wraps the parent call in a try/except.
### If a PermissionDeniedError fires and its message contains "location",
### we re-raise it with _LOCATION_ERROR_MSG instead of the raw OpenAI error.
### If it's a different PermissionDeniedError (wrong key, etc.), we let it
### propagate normally so unrelated errors aren't swallowed.
class _LocationAwareOpenAI(OpenAI):
    """Wraps OpenAI to surface a helpful error when Kaggle's proxy rejects
    the request due to an unsupported user location or account restriction."""

    def _request(self, *args, **kwargs):
        try:
            return super()._request(*args, **kwargs)
        except openai.PermissionDeniedError as e:
            if "location" in str(e).lower():
                raise openai.PermissionDeniedError(
                    _LOCATION_ERROR_MSG.format(original=e),
                    response=e.response,
                    body=e.body,
                ) from e
            raise
