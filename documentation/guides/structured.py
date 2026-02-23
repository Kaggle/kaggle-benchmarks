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
# title: Working with Structured Output
# ---
# When interacting with Large Language Models (LLMs), you often need to extract specific
# information in a structured format rather than just free-form text.
# This guide demonstrates how to use the `schema` parameter in `llm.prompt()` to directly
# obtain parsed Python objects, simplifying data extraction and further processing.

# %%
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from kaggle_benchmarks import chats, config, llm

# So we can see what exactly was sent to the API
config.show_message_details = True

# %% [markdown]

# You can request simple Python types like `bool`, `int`, or `str`.
# The LLM will be asked to provide a response that can be directly parsed into this type.

# This avoids manual parsing of textual "yes"/"no" or "true"/"false" responses, making your code cleaner and less error-prone.

# In detail section you can see that extra message was automatically added to prompt LLM to produce response in a particular format and the response was automatically parsed into boolean value.
# %%

with chats.new("Prime number check") as t:
    response = llm.prompt(
        "Is 19999999 a prime number",
        schema=bool,
    )

t


# %% [markdown]
# ## Using Dataclasses
#
# For more complex structures, you can use Python `dataclasses`.
# Define your desired structure, and the LLM's output will be parsed into an instance of your dataclass.
# %%


@dataclass
class Bio:
    date: str
    place: str


with chats.new("albert") as t:
    response = llm.prompt(
        "When and where Albert Enshtein was born?",
        schema=Bio,
    )

t


# %% [markdown]
# ## Using Pydantic Models
#
# For advanced scenarios requiring validation, type hints with constraints (like `Literal`),
# and descriptions that can guide the LLM, Pydantic `BaseModel`s are an excellent choice.
#
# The `Field` descriptions in Pydantic models can be particularly helpful, as they can be passed to the LLM as part of the schema definition to better guide its output format.

# %%

# If model doesn't support structured output it will be asked to write a response using auto-generated json-schema
llm.support_structured_outputs = False


class NamedEntity(BaseModel):
    type: Literal["PERSON", "LOCATION", "ORGANIZATION"] = Field(
        description="The type of the named entity (e.g., PERSON, LOCATION, ORGANIZATION)."
    )
    text: str = Field(description="The text of the named entity.")


class Entities(BaseModel):
    entities: list[NamedEntity] = Field(description="List of etities.")


system = """You are an expert Named Entity Recognition system.
Your task is to extract the named entities from the text and return them as json."""

text = """Albert Einstein was born in Ulm, Germany, on March 14, 1879. He developed the theory of relativity.
He later moved to Princeton, New Jersey. He worked at the Institute for Advanced Study. He died on April 18, 1955."""

with chats.new("NER", system_instructions=system) as t:
    response = llm.prompt(text, schema=Entities)
    assert len(response.entities) > 3

    assert ("PERSON", "Albert Einstein") in [
        (e.type, e.text) for e in response.entities
    ], response.entities

t

# %%
