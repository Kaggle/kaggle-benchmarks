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
# title: Pokedex
# ---
# %%

import io

import pandas as pd

from kaggle_benchmarks import LLMChat, llm, prompting, task, utils


@prompting.handler(types=pd.DataFrame)
def pandas_dataframe(_):
    content = yield "Write output as a csv file. Make sure to include a header and use quotation marks if needed."
    content = utils.extract_code_block(content)
    return pd.read_csv(io.StringIO(content))


@task(name="Pokédex")
def pokedex(llm: LLMChat) -> bool:
    """Checks whether llm can name 20 pokemons."""

    df = llm.prompt("Can you list first 20 pokemons?", schema=pd.DataFrame)
    return len(df) == 20


pokedex.run(llm)

# %%
