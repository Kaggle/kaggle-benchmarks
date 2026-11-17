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

import pytest

from kaggle_benchmarks.kaggle import model_proxy


@pytest.mark.parametrize(
    "model, api, expected",
    [
        ("google/gemini-3.8-flash", "openai", True),
        ("google/gemini-3.8-flash", "genai", True),
        ("qwen/qwen3.8-max", "openai", True),
        ("qwen/qwen3.8-max", "genai", False),
        ("qwen/qwen3-next-80b-a3b-instruct", "openai", False),
        ("deepseek-ai/deepseek-v4.1-flash", "openai", False),
    ],
)
def test_support_structured_outputs(model, api, expected):
    llm = model_proxy.ModelProxy(
        model=model, api=api, api_key="key", base_url="https://example.test/models"
    )

    assert llm.support_structured_outputs is expected
