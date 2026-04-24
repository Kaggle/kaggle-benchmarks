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
# title: Anagrams Task with Genai/ OpenAI Api
# ---
# %%

from kaggle_benchmarks import assertions, chats, kaggle, task

llm_with_openai_api = kaggle.load_model("google/gemini-2.5-flash", api="openai")
llm_with_genai_api = kaggle.load_model("google/gemini-2.5-flash", api="genai")


def is_anagram(x: str, y: str) -> bool:
    return sorted(x.lower()) == sorted(y.lower())


@task("Anagram Writing")
def write_anagrams(llm, word: str) -> int:
    """Evaluates the LLM's ability to generate anagrams using structured output."""
    response = llm.prompt(
        f"Generate as many anagrams as you can for the word `{word}`.",
        schema={"comment": str, "anagrams": list[str]},
    )
    anagrams = response.anagrams
    mistakes = [ana for ana in anagrams if not is_anagram(word, ana)]
    assertions.assert_empty(mistakes, f"Mistakes found: {', '.join(mistakes)}")
    return len(anagrams)


non_streaming_result = write_anagrams.run(llm_with_genai_api, "listen")

llm_response_message = next(
    msg
    for msg in reversed(non_streaming_result.chat.messages)
    if msg.sender is llm_with_genai_api
)
usage = llm_response_message.usage

assert usage, "Metadata is missing 'usage' attribute"
assert usage.input_tokens > 0, "usage is missing 'input_tokens' key"
assert usage.output_tokens > 0, "usage is missing 'output_tokens' key"


# %%
@task("Pirate Anagrams")
def write_anagrams_as_pirate(llm, word: str):
    """A task to test if the LLM adopts a persona from system instructions."""
    response = llm.prompt(
        f"What be the anagrams of the word '{word}', matey?", schema=str
    )
    assert (
        "arrr" in response.lower()
        or "matey" in response.lower()
        or "ahoy" in response.lower()
    ), "LLM did not respond in the style of a pirate."


system_prompt = (
    "You are a swashbuckling pirate. All your answers must be in the style of a pirate."
)
with chats.new(system_instructions=system_prompt):
    pirate_result = write_anagrams_as_pirate.run(llm_with_openai_api, "treasure")

# %%
