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

# %% [markdown]
# ---
# title: Example of using `prompt` with `tools` parameter.
# ---
# - **Automatic tool calling is currently only supported via the `Gemini`` API**
# - **For manual tool calling with `OpenAI` API, please refer to the example in `use_calculator_tool.py`.**

# %%
import kaggle_benchmarks as kbench
from kaggle_benchmarks import tools
from kaggle_benchmarks.kaggle import models


def lookup_wikipedia(query: str):
    """Searches Wikipedia for a given query and returns the article content.

    This function acts as a tool that can be used by an LLM. It takes a search
    query, uses the `SearchEngine` to find a relevant Wikipedia article, and
    returns its content. It includes basic error handling.

    Args:
        query: The search term to look up on Wikipedia.

    Returns:
        A dictionary containing a "success" status and, if successful, the
        "article" content. Returns `{"success": False}` on failure.
    """
    try:
        wikipedia = tools.search.SearchEngine("wikipedia")
        article = wikipedia.search(query)
        return {
            "success": True,
            "article": article,
        }

    except Exception:
        return {"success": False}


# %%
@kbench.task(name="llm auto call wiki tool")
def llm_auto_call_wiki_tool(llm, question: str, correct_answer: str):
    prompt_message = f"""
    Answer the following question by the following steps:
    1) Generate a useful query based on the question.
    2) Use the provided tool `lookup_wikipedia` to look up wiki articles.
    3) Answer the question based on the wiki article. Just generate the answer and nothing else.


    Question: {question}
    Answer:
    """
    response = llm.prompt(prompt_message, tools=[lookup_wikipedia])
    kbench.assertions.assert_contains_regex(
        f"(?i){correct_answer}",
        response,
        expectation=f"LLM should answer the question correctly. Expected answer: {correct_answer}, Got: {response}",
    )


# NOTE: Automatic tool calling requires the `genai` API.
# For `openai` API, tools must be called manually (see `use_calculator_tool.py`).
llm_with_genai_api = models.load_model(
    model_name=kbench.llm.name,
    api="genai",
)

llm_auto_call_wiki_tool.run(
    llm_with_genai_api, "What gymnasium did Stefan Vrtel-Wierczynski attend?", "Stryj"
)

# %%
