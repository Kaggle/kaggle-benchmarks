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

# %% [markdown]
# ---
# title: Reasoning with Thoughts
# ---
# This example demonstrates how to use the `reasoning_level` and
# `include_thoughts` parameters when loading a model via Model Proxy.
#
# - `reasoning_level` maps to the OpenAI `reasoning_effort` parameter
#   and controls how much reasoning the model performs. Options: "low",
#   "medium", "high".
# - `include_thoughts` sends the `google.thinking_config` extra body
#   so the proxy returns inline thinking traces wrapped in `<think>` tags.
# %%
from kaggle_benchmarks import assertions, task
from kaggle_benchmarks.kaggle import load_model

question = "How many r's are in 'strawberry'?"


@task("Reasoning with thoughts")
def reasoning_with_thoughts(llm, question: str):
    """Evaluates reasoning with thinking traces enabled."""
    answer = llm.prompt(question)
    assertions.assert_not_empty(answer, "LLM should return a non-empty answer.")


# %% [markdown]
# ### 1. reasoning_level only
# %%
llm_reasoning = load_model("google/gemini-2.5-pro", reasoning_level="high")
reasoning_with_thoughts.run(llm_reasoning, question)

# %% [markdown]
# ### 2. reasoning_level + include_thoughts
# %%
llm_both = load_model(
    "google/gemini-2.5-pro", reasoning_level="high", include_thoughts=True
)
reasoning_with_thoughts.run(llm_both, question)

# %% [markdown]
# ### 3. include_thoughts only
# %%
llm_thoughts = load_model("google/gemini-2.5-pro", include_thoughts=True)
reasoning_with_thoughts.run(llm_thoughts, question)

# %%
