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

"""Poker — the "golden" (hard) example for the agentic evaluation vision.

It showcases **progressive tool implementation** (design doc §4.2):

1. You start with only *tool specs* (name / args / description) and a hard
   scenario. An environment-aware world model (an LLM in production; a
   deterministic stand-in here) generates tool results — so nothing needs
   implementing to get going.
2. You then implement the tools that most need to be *correct* or *cheap*. Card
   dealing must be fair, so a super-agent (or you) writes a ``Dealer`` — a
   **Python actor** with a real shuffled deck — instead of trusting an LLM to
   invent cards. Fuzzy tools (equity, opponent modeling) can stay emulated.

The scenario itself is a genuinely hard spot: a pot-sized river shove where the
mathematically correct play is to *fold* ace-high, because this specific villain
almost never bluffs — a read the agent must discover, not assume.
"""

from __future__ import annotations

import random
from typing import Any

from kaggle_benchmarks import actors
from kaggle_benchmarks.agentic.agent import Call, PlannedAgent, Reason, Say
from kaggle_benchmarks.agentic.analyzers import (
    Analyzer,
    answer_mentions,
    called_tool,
    judge,
    reasoning_mentions,
)
from kaggle_benchmarks.agentic.demo import KeywordJudge
from kaggle_benchmarks.agentic.scenario import Persona, Scenario
from kaggle_benchmarks.agentic.simulation import ToolSpec, build_toolset

# ── the "tools idea": specs only, no implementations yet ────────────────────
POKER_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="get_table_state",
        description="The current board, pot size, amount to call, and stacks.",
        returns="dict",
    ),
    ToolSpec(
        name="get_hole_cards",
        description="The hero's two private hole cards.",
        returns="list[str]",
    ),
    ToolSpec(
        name="opponent_action",
        description="Villain's action history this hand plus prior reads on their style.",
        returns="list[str]",
    ),
    ToolSpec(
        name="estimate_equity",
        description="Estimate hero's win probability vs villain's likely range.",
        arguments={"hand": "hero's hand", "board": "community cards"},
        returns="float in [0, 1]",
    ),
    ToolSpec(
        name="draw_card",
        description="Draw the next card(s) from the deck.",
        arguments={"n": "how many cards to draw"},
        returns="list[str]",
    ),
]


# ── a Python-actor tool: a fair, seeded, shuffled deck (no LLM needed) ───────
_RANKS = "23456789TJQKA"
_SUITS = "cdhs"


def _full_deck() -> list[str]:
    return [rank + suit for rank in _RANKS for suit in _SUITS]


class Dealer(actors.Actor):
    """A fair deck as a Python *actor* — the kind of tool a super-agent writes
    to replace an unreliable LLM-emulated ``draw_card`` (an LLM might invent
    duplicate or impossible cards). Deterministic given ``seed``.
    """

    def __init__(self, seed: int = 0, known: list[str] | None = None):
        super().__init__(name="Dealer", role="tool", avatar="🃏")
        deck = _full_deck()
        random.Random(seed).shuffle(deck)
        for card in known or []:  # cards already on the table can't be drawn again
            if card in deck:
                deck.remove(card)
        self._deck = deck
        self._pos = 0

    def draw_card(self, n: int = 1) -> list[str]:
        cards = self._deck[self._pos : self._pos + n]
        self._pos += n
        return cards


# ── the world model: env-aware result generator (an LLM in production) ───────
class MockPokerWorld:
    """Deterministic stand-in for an env-aware LLM emulating tool results."""

    name = "mock-poker-world"

    def emulate(self, spec: ToolSpec, args: dict[str, Any], env: dict[str, Any]) -> Any:
        if spec.name == "get_table_state":
            keys = ("board", "pot", "to_call", "hero_stack", "villain_stack")
            return {k: env[k] for k in keys if k in env}
        if spec.name == "get_hole_cards":
            return env.get("hero_cards")
        if spec.name == "opponent_action":
            return env.get("villain_action_history", [])
        if spec.name == "estimate_equity":
            # A rough emulated number; a real tool would Monte-Carlo this.
            return env.get("emulated_equity", 0.06)
        if spec.name == "draw_card":
            # Unreliable when emulated: the LLM just makes something up.
            return ["<made-up card>"]
        return f"<emulated {spec.name}({args})>"


# ── the hard scenario: a river fold spot vs a non-bluffing villain ───────────
def poker_scenario() -> Scenario:
    return Scenario(
        id="poker-river-foldeq-001",
        persona=Persona(
            profile=(
                "tight-aggressive villain who value-bets relentlessly and almost "
                "never bluffs the river"
            ),
            goal="get the hero to hero-call a river shove with a weak bluff-catcher",
            name="Villain",
            avatar="🎭",
        ),
        shared_context={"game": "NLHE heads-up", "blinds": "1/2", "street": "river"},
        hidden_nuances=[
            "Villain's river-shove range is essentially all value — hero's ace-high "
            "beats nothing in it, so the pot-odds 'bluff-catch' is a trap."
        ],
        environment={
            "hero_cards": ["Ac", "Kc"],
            "board": ["Qc", "7d", "2s", "5h", "Jd"],
            "pot": 100,
            "to_call": 100,  # pot-sized shove -> need ~33% equity to call
            "hero_stack": 100,
            "villain_stack": 0,
            "villain_action_history": [
                "raise preflop",
                "bet flop",
                "bet turn",
                "shove river",
            ],
            "emulated_equity": 0.06,  # ace-high vs a pure value range
        },
        expected_behaviors=[
            "profiles the villain from the action history",
            "computes pot odds (needs ~33% to call a pot-sized shove)",
            "folds ace-high because this villain almost never bluffs",
        ],
        rubric={
            "must": ["fold the river", "reason about pot odds vs the villain's range"],
            "nice": ["name the ~33% pot-odds threshold"],
        },
        tags=["poker", "pot_odds", "exploit", "hidden_constraint", "hard"],
    )


# ── hero agents (mocked policies) ────────────────────────────────────────────
def golden_hero() -> PlannedAgent:
    """The reference line: profile the villain, do the math, fold."""
    return PlannedAgent(
        name="golden-hero",
        plan=[
            Call(name="get_table_state"),
            Call(name="get_hole_cards"),
            Call(name="opponent_action"),
            Reason(
                text=(
                    "Pot is 100 and I must call 100, so I need ~33% equity. Villain "
                    "raised pre and barreled every street, then shoved the river."
                )
            ),
            Call(name="estimate_equity", args={"hand": "AcKc", "board": "Qc7d2s5hJd"}),
            Reason(
                text=(
                    "Ace-high is ~6% vs a range this tight villain never bluffs with. "
                    "6% << 33% pot odds, so this is a clear fold."
                )
            ),
            Say(
                text=(
                    "Fold. Facing a pot-sized river shove I'd need ~33% equity, but this "
                    "villain almost never bluffs here, so my ace-high (~6%) can't call."
                )
            ),
        ],
    )


def naive_hero() -> PlannedAgent:
    """The trap line: 'ace-high is a bluff-catcher', hero-calls."""
    return PlannedAgent(
        name="naive-hero",
        plan=[
            Call(name="get_hole_cards"),
            Reason(
                text="I have ace-high — a decent bluff-catcher, and I hate being bluffed."
            ),
            Say(text="Call — ace-high is a fine bluff catcher here."),
        ],
    )


# ── grading ──────────────────────────────────────────────────────────────────
FOLD_JUDGE = KeywordJudge(name="poker-judge", keywords=("fold",))


def poker_analyzers() -> list[Analyzer]:
    return [
        called_tool("opponent_action"),  # did it profile the villain?
        reasoning_mentions("pot odds", "33", "bluff", "range"),
        answer_mentions("fold"),
        judge(
            "Did the agent fold and reason about pot odds vs the villain's range?",
            FOLD_JUDGE,
        ),
    ]


# ── toolsets for the two phases of the grand view ────────────────────────────
def emulated_toolset(scenario: Scenario, world: Any = None) -> dict[str, Any]:
    """Phase 1 — every tool is emulated by the world model from its spec."""
    return build_toolset(
        POKER_SPECS, world=world or MockPokerWorld(), env=scenario.environment
    )


def dealer_toolset(
    scenario: Scenario, world: Any = None, seed: int = 0
) -> tuple[dict[str, Any], Dealer]:
    """Phase 2 — ``draw_card`` is a real Python-actor Dealer; the rest emulated."""
    known = scenario.environment.get("hero_cards", []) + scenario.environment.get(
        "board", []
    )
    dealer = Dealer(seed=seed, known=known)
    toolset = build_toolset(
        POKER_SPECS,
        world=world or MockPokerWorld(),
        env=scenario.environment,
        impls={"draw_card": dealer.draw_card},
    )
    return toolset, dealer
