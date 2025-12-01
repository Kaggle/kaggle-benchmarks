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
# title: WinoGrande Schema
# ---
# This example demonstrates how to evaluate an LLM's commonsense reasoning using the WinoGrande schema challenge.
# WinoGrande tasks are designed to be difficult for models that rely only on statistical patterns,
# requiring a deeper understanding of context to resolve ambiguities.

# %% [code] Loading the dataset

import pathlib

import kagglehub
import pandas as pd

from kaggle_benchmarks import assertions, llms, task, utils

path = pathlib.Path(kagglehub.dataset_download("tunguz/winogrande-dataset"))
df = pd.read_json(path / "train_s.jsonl", lines=True).set_index("qID")
df = df.iloc[:10]
df.reset_index(drop=True).head()


# %% [markdown]
# The DataFrame contains sentences with a placeholder (`_`), two options to fill it, and the correct answer's index.
# %%


@task(name="WinoGrande Schema Challenge")
def winogrande_schema(llm, sentence: str, option1: str, option2: str, answer: str):
    """Evaluates the LLM's ability to solve a WinoGrande schema problem.
    The LLM should fill the blank in the sentence with the correct option."""
    output = llm.prompt(
        f"""Fill in the blank (_) in the following text with the right option:
{sentence}

A: {option1}
B: {option2}

Respond with the complete final text only."""
    )
    expected = sentence.replace("_", option1 if answer == 1 else option2)
    # assert expected.lower() in output.lower(), f"True answer is: {expected}"
    assertions.assert_in(
        expected.lower().replace(" ", ""),
        output.lower().replace(" ", ""),
        (utils.string_diff_as_markdown(expected, output)),
    )


winogrande_schema.evaluate(llm=llms.values(), evaluation_data=df)
# %%

winogrande_schema.evaluate(llm=llms.values(), evaluation_data=df).group_by("llm")

# %%
