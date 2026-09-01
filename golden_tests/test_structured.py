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

"""Structured-output & simple-prompt benchmark tasks, and their golden tests.

Each task is followed by its tests: a scripted one that replays canned responses
through ``fake(...)`` and runs with no API key, and a live one parametrized over
a model pool, which skips when no provider is configured. Tests asserting a
*failure* are scripted only — a real model may legitimately answer correctly.

Known live failures here: genai/gpt-5.5 (Model Proxy sends an empty
json_schema.name) and openai/deepseek-r1 (nondeterministic on int extraction).
"""

from dataclasses import dataclass

import pytest
from models import ALL_MODELS, fake
from pydantic import BaseModel, Field

import kaggle_benchmarks as kbench


@dataclass
class RPGCharacter:
    name: str
    class_type: str
    level: int
    inventory: str


class Planet(BaseModel):
    name: str
    mass_earth_masses: float = Field(description="Mass relative to Earth")
    has_life: bool = Field(description="Whether the planet is known to have life")
    moons: list[str] = Field(default_factory=list, description="List of major moons")


class FriendsActor(BaseModel):
    actor_name: str
    role_name: str


class Casting(BaseModel):
    actors: list[FriendsActor]


@kbench.task(name="capital_of_france")
def capital_of_france(llm):
    """A minimal single-prompt task: the answer must mention Paris."""
    response = llm.prompt("What is the capital of France?")
    kbench.assertions.assert_in(
        "paris", response.lower(), expectation="Answer should mention Paris."
    )


def test_capital_of_france_scripted():
    assert capital_of_france.run(fake(["Paris."])).passed


def test_capital_of_france_wrong_city_fails():
    assert not capital_of_france.run(fake(["London."])).passed


@kbench.task(name="remembers_name")
def remembers_name(llm):
    """A multi-turn task: the model must recall a name across two prompts."""
    llm.prompt("My name is Alice. Please remember it.")
    response = llm.prompt("What is my name?")
    kbench.assertions.assert_in(
        "alice", response.lower(), expectation="Model should remember the name Alice."
    )


def test_remembers_name_scripted():
    assert remembers_name.run(fake(["Hi Alice.", "Your name is Alice."])).passed


def test_remembers_name_forgotten_fails():
    assert not remembers_name.run(fake(["Hi.", "I don't know."])).passed


@kbench.task(name="extract_int")
def extract_int(llm):
    text = "The Apollo 11 mission landed on the Moon in 1969."
    year = llm.prompt(f"Extract the year from this text: '{text}'", schema=int)
    kbench.assertions.assert_equal(
        1969, year, expectation="Extracted year should be 1969."
    )


def test_extract_int_scripted():
    assert extract_int.run(fake([{"value": 1969}])).passed


def test_extract_int_wrong_year_fails():
    assert not extract_int.run(fake([{"value": 1900}])).passed


@pytest.mark.parametrize("llm", ALL_MODELS)
def test_extract_int(llm):
    assert extract_int.run(llm).passed


@kbench.task(name="extract_bool")
def extract_bool(llm):
    text = "I absolutely loved this movie! It was fantastic."
    is_positive = llm.prompt(f"Is this review positive? '{text}'", schema=bool)
    kbench.assertions.assert_true(
        is_positive, expectation="Sentiment should be positive."
    )


def test_extract_bool_scripted():
    assert extract_bool.run(fake([{"value": True}])).passed


def test_extract_bool_negative_fails():
    assert not extract_bool.run(fake([{"value": False}])).passed


@pytest.mark.parametrize("llm", ALL_MODELS)
def test_extract_bool(llm):
    assert extract_bool.run(llm).passed


@kbench.task(name="extract_dict")
def extract_dict(llm):
    text = "Contact info: John Doe, age 42, works as a Software Engineer."
    person_schema = {"name": str, "age": int, "occupation": str}
    person = llm.prompt(f"Extract person details from: '{text}'", schema=person_schema)
    kbench.assertions.assert_equal(
        "John Doe", person.name, expectation="Name should be John Doe."
    )
    kbench.assertions.assert_equal(42, person.age, expectation="Age should be 42.")
    kbench.assertions.assert_contains_regex(
        r"(?i)software engineer",
        person.occupation,
        expectation="Occupation should match.",
    )


def test_extract_dict_scripted():
    response = {"name": "John Doe", "age": 42, "occupation": "Software Engineer"}
    assert extract_dict.run(fake([response])).passed


@pytest.mark.parametrize("llm", ALL_MODELS)
def test_extract_dict(llm):
    assert extract_dict.run(llm).passed


@kbench.task(name="extract_dataclass")
def extract_dataclass(llm):
    character = llm.prompt(
        "Generate a level 5 wizard character for a fantasy game.", schema=RPGCharacter
    )
    kbench.assertions.assert_true(
        len(character.name) > 0, expectation="Character should have a name."
    )
    kbench.assertions.assert_contains_regex(
        r"(?i)wizard", character.class_type, expectation="Class should be Wizard."
    )
    kbench.assertions.assert_equal(5, character.level, expectation="Level should be 5.")
    kbench.assertions.assert_true(
        len(character.inventory) > 0, expectation="Inventory should not be empty."
    )


def test_extract_dataclass_scripted():
    response = {
        "name": "Gandalf",
        "class_type": "Wizard",
        "level": 5,
        "inventory": "a staff",
    }
    assert extract_dataclass.run(fake([response])).passed


@pytest.mark.parametrize("llm", ALL_MODELS)
def test_extract_dataclass(llm):
    assert extract_dataclass.run(llm).passed


@kbench.task(name="extract_pydantic")
def extract_pydantic(llm):
    planet = llm.prompt("Provide information about the planet Jupiter.", schema=Planet)
    kbench.assertions.assert_contains_regex(
        r"(?i)jupiter", planet.name, expectation="Planet name should be Jupiter."
    )
    kbench.assertions.assert_true(
        planet.mass_earth_masses > 300,
        expectation="Jupiter mass should be > 300 Earths.",
    )
    kbench.assertions.assert_true(
        len(planet.moons) > 0, expectation="Jupiter should have moons."
    )


def test_extract_pydantic_scripted():
    response = {
        "name": "Jupiter",
        "mass_earth_masses": 317.8,
        "has_life": False,
        "moons": ["Io", "Europa"],
    }
    assert extract_pydantic.run(fake([response])).passed


@pytest.mark.parametrize("llm", ALL_MODELS)
def test_extract_pydantic(llm):
    assert extract_pydantic.run(llm).passed


@kbench.task(name="extract_composite_pydantic")
def extract_composite_pydantic(llm):
    casting = llm.prompt("List the 6 main characters of Friends.", schema=Casting)
    kbench.assertions.assert_equal(len(casting.actors), 6)
    names = ", ".join([actor.actor_name for actor in casting.actors])
    role_names = ", ".join([actor.role_name for actor in casting.actors])
    kbench.assertions.assert_in("Jennifer", names)
    kbench.assertions.assert_in("Ross", role_names)


def test_extract_composite_pydantic_scripted():
    response = {
        "actors": [
            {"actor_name": "Jennifer Aniston", "role_name": "Rachel Green"},
            {"actor_name": "Courteney Cox", "role_name": "Monica Geller"},
            {"actor_name": "Lisa Kudrow", "role_name": "Phoebe Buffay"},
            {"actor_name": "Matt LeBlanc", "role_name": "Joey Tribbiani"},
            {"actor_name": "Matthew Perry", "role_name": "Chandler Bing"},
            {"actor_name": "David Schwimmer", "role_name": "Ross Geller"},
        ]
    }
    assert extract_composite_pydantic.run(fake([response])).passed


@pytest.mark.parametrize("llm", ALL_MODELS)
def test_extract_composite_pydantic(llm):
    assert extract_composite_pydantic.run(llm).passed
