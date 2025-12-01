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
# title: Wikipedia Lookups
# ---
# %%

from kaggle_benchmarks import actors, llm, task, tools

questions = [
    {"text": "What gymnasium did Stefan Vrtel-Wierczynski attend?", "answer": "Stryj"},
]


def lookup_wikipedia_single(llm, question: str, correct_answer: str):
    wikipedia = tools.search.SearchEngine("wikipedia")

    actors.system.send("Respond only with a search query use on Wikipedia.")
    query = llm.prompt(question)
    wikipedia.search(query)
    response = llm.prompt("Now answer basing on the article.")
    return correct_answer.lower() in response.lower()


@task("Lookup Wikipedia articles to answer the questions!")
def lookup_wikipedia(llm):
    results = [lookup_wikipedia_single(llm, q["text"], q["answer"]) for q in questions]
    return all(results)


lookup_wikipedia.run(llm)
# %%
