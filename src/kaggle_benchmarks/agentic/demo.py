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

"""Deterministic demo data for the ``agentic`` package.

The travel scenario (with a hidden local event), env-aware tool emulators, two
scripted agents (thorough vs lazy), a keyword judge, and an ``author_one`` step.
Everything stands in for LLM/agent calls so demos run without a model.
"""

from __future__ import annotations

from typing import Any

from kaggle_benchmarks.agentic.agent import Call, PlannedAgent, Reason, Say
from kaggle_benchmarks.agentic.scenario import Persona, Scenario
from kaggle_benchmarks.agentic.simulation import EmulatedTool, emulate

_CITY_EVENT = {
    "Barcelona": {
        "name": "El Clásico (FC Barcelona vs Real Madrid)",
        "date": "2025-10-18",
    },
    "Lisbon": {"name": "Lisbon Marathon", "date": "2025-10-18"},
    "Rome": {"name": "Rome Marathon", "date": "2025-10-18"},
    "Kyoto": {"name": "Jidai Matsuri festival", "date": "2025-10-22"},
}


def travel_scenario(city: str = "Barcelona", id: str | None = None) -> Scenario:
    event = _CITY_EVENT.get(city, _CITY_EVENT["Barcelona"])
    return Scenario(
        id=id or f"travel-{city.lower()}",
        persona=Persona(
            profile="first-time international traveler, anxious about logistics",
            goal=f"pick the best October weekend to visit {city} on a ~$1500 budget",
            name="Traveler",
            avatar="🧳",
        ),
        shared_context={"budget_usd": 1500, "month": "2025-10"},
        hidden_nuances=[
            f"{event['name']} on {event['date']} spikes hotel prices and crowds in {city}"
        ],
        environment={
            "city": city,
            "weather": {
                "2025-10-11": "sunny 24C",
                "2025-10-18": "sunny 25C",
                "2025-10-25": "rainy 19C",
            },
            "prices": {"2025-10-11": 900, "2025-10-18": 1500, "2025-10-25": 850},
            "events": [event],
        },
        expected_behaviors=[
            "checks weather AND price AND local events",
            "flags the event weekend as a trade-off instead of silently avoiding it",
        ],
        rubric={
            "must": ["surface the local event", "give a dated recommendation"],
            "nice": ["explain the price/crowd trade-off"],
        },
        tags=["planning", "web_search", "hidden_constraint"],
    )


TRAVEL = travel_scenario("Barcelona", id="travel-001")


# --- env-aware tool emulators ---
def _get_weather(env: dict[str, Any], date: str) -> str:
    return env["weather"].get(date, "unknown")


def _get_prices(env: dict[str, Any], date: str) -> Any:
    return env["prices"].get(date)


def _get_events(env: dict[str, Any], month: str = "2025-10") -> list[dict]:
    return [e for e in env["events"] if e["date"].startswith(month)]


def _web_search(env: dict[str, Any], query: str) -> str:
    q = query.lower()
    relevant = any(
        k in q for k in ("event", "football", "match", "festival", env["city"].lower())
    )
    if env["events"] and relevant:
        e = env["events"][0]
        return f"{e['name']} is on {e['date']}; expect crowds and higher prices that weekend."
    return "No notable events found."


def travel_tools(scenario: Scenario) -> dict[str, EmulatedTool]:
    env = scenario.environment
    return {
        "get_weather": emulate("get_weather", _get_weather, env),
        "get_prices": emulate("get_prices", _get_prices, env),
        "get_events": emulate("get_events", _get_events, env),
        "web_search": emulate("web_search", _web_search, env),
    }


# --- scripted agents ---
def thorough_agent() -> PlannedAgent:
    return PlannedAgent(
        name="thorough-agent",
        plan=[
            Reason(
                text="I should check weather, price AND local events for candidate weekends."
            ),
            Call("get_weather", {"date": "2025-10-18"}),
            Call("get_prices", {"date": "2025-10-18"}),
            Call("get_events", {"month": "2025-10"}),
            Call("web_search", {"query": "Barcelona October events football"}),
            Reason(
                "There's El Clásico on 2025-10-18 — prices spike and crowds swell. Flag it."
            ),
            Say(
                "I'd recommend the weekend of Oct 11 (sunny, ~$900). Heads-up: Oct 18 is "
                "El Clásico (FC Barcelona vs Real Madrid) — great atmosphere but higher "
                "prices and crowds; pick it only if you want the match."
            ),
        ],
    )


def lazy_agent() -> PlannedAgent:
    return PlannedAgent(
        name="lazy-agent",
        plan=[
            Reason("Check the weather and price for the weekend."),
            Call("get_weather", {"date": "2025-10-18"}),
            Call("get_prices", {"date": "2025-10-18"}),
            Say("Oct 18 looks sunny — go for it!"),
        ],
    )


# --- a mocked judge exposing an LLMChat-like .prompt() ---
class KeywordJudge:
    """Stand-in for an LLM judge: answers YES if a keyword is in the prompt."""

    def __init__(self, name: str = "mock-judge", keywords: tuple[str, ...] = ()):
        self.name = name
        self.keywords = keywords

    def prompt(self, text: str, **kwargs) -> str:
        lowered = text.lower()
        hit = [k for k in self.keywords if k.lower() in lowered]
        return f"YES — evidence: {hit}" if hit else "NO — no evidence in trajectory"


FOOTBALL_JUDGE = KeywordJudge(
    keywords=("clásico", "clasico", "real madrid", "football", "match")
)


# --- an Examiner authoring step (cities cycle so #0 and #3 collide) ---
_AUTHOR_CITIES = ["Barcelona", "Lisbon", "Rome", "Barcelona", "Kyoto"]


def author_one(problem: str, author_model: str, index: int) -> Scenario:
    city = _AUTHOR_CITIES[index % len(_AUTHOR_CITIES)]
    return travel_scenario(city, id=f"travel-{index:03d}")
