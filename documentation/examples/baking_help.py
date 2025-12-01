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
# title: Baking help
# ---
# Adaptataion of
# https://github.com/carlini/yet-another-applied-llm-benchmark/blob/apache2/tests/baking_help.py#L3
# %%
from kaggle_benchmarks import assertions, chats, llm, task
from kaggle_benchmarks.kaggle import models


@task(name="Missing step in recipe.")
def baking_help(llm, eval_llm) -> bool:
    """Test if the model can identify a missing incredient in a recipe. Identifying incorrect steps is much harder than missing steps."""
    response = llm.prompt("""
I'm making pancakes for breakfast. I added a cup of flour, a teaspoon of salt, and a few tablespoons of sugar to a bowl. I stirred it together, then added a cup of milk, a beaten egg, and a few tablespoons of oil, and stirred until just mixed. Then I put 1/4 a cup on a hot frying pan, and flipped it when brown. But they're terrible! Why? List the main reason.
""")

    with chats.new("Evaluation"):
        eval_response = eval_llm.prompt(f"""
Below is a student's answer to a question on what is missing from some pancake recipe: {response}

Does the student mention that baking powder or baking soda is missing? Think out loud, and then end your answer with either the string "The student passes" or "The student fails".
""")
        assertions.assert_in("student passes", eval_response.lower())

    return "student passes" in eval_response.lower()


baking_help.run(llm, llm)

# %%
# Or use Google genai as backend api
genai_llm = models.load_model(llm.name, api="genai")
baking_help.run(genai_llm, genai_llm)
# %%
