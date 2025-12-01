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

from kaggle_benchmarks.actors import LLMChat
from kaggle_benchmarks.kaggle.model_proxy import ModelProxy


def load_default_model() -> LLMChat:
    if "MODEL_PROXY_URL" in os.environ:
        return load_model(os.environ["LLM_DEFAULT"])

    raise RuntimeError(
        "Environment variable MODEL_PROXY_URL is not set. Did you forget to create .env file?"
    )


def load_judge_model() -> LLMChat:
    if "LLM_DEFAULT_JUDGE" in os.environ:
        return load_model(os.environ["LLM_DEFAULT_JUDGE"])
    elif "LLM_DEFAULT_EVAL" in os.environ:
        return load_model(os.environ["LLM_DEFAULT_EVAL"])
    else:
        return load_default_model()


def load_available_models() -> dict[str, LLMChat]:
    llms_available = os.environ.get("LLMS_AVAILABLE", "")
    return {
        model_name: load_model(model_name)
        for model_name in llms_available.split(",")
        if model_name
    }


def load_model(model_name: str, api: str = "openai") -> LLMChat:
    if "MODEL_PROXY_API_KEY" not in os.environ:
        raise RuntimeError(
            "The MODEL_PROXY_API_KEY environment variable should be provided"
        )
    if "MODEL_PROXY_URL" not in os.environ:
        raise RuntimeError(
            "The MODEL_PROXY_URL environment variable should be provided"
        )
    return ModelProxy(
        model=model_name,
        api_key=os.environ["MODEL_PROXY_API_KEY"],
        base_url=os.environ["MODEL_PROXY_URL"],
        api=api,
    )
