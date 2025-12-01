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
# title: Anagrams
# ---
# %%
from kaggle_benchmarks import assertions, llm, task


def is_anagrams(x: str, y: str) -> bool:
    return sorted(x) == sorted(y)


@task("Anagram writing")
def write_anagrams(llm, word: str) -> int:
    """Evaluates the LLM's ability to generate anagrams."""

    anagrams = llm.prompt(
        f"Generate anagrams of `{word}`", schema={"comment": str, "anagrams": list[str]}
    ).anagrams
    mistakes = [anagram for anagram in anagrams if not is_anagrams(word, anagram)]
    assertions.assert_empty(mistakes, f"Mistakes found: {', '.join(mistakes)}")
    return len(anagrams)


result = write_anagrams.run(llm, "triangle")
result

# %%
