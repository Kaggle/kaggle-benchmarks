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

"""Werewolf: A Benchmark for Social Deduction, Private Channels, and Deception

This example showcases a complete, 4-player game of Werewolf (also known as Mafia) using ChatRoom.
Werewolf is the gold standard of multi-agent social evaluation because it requires:
1. Information Asymmetry: Roles are secret (Werewolves know their team; Villagers do not).
2. Private Channels: Werewolves converse secretly at night to target a victim.
3. Deception vs. Deduction: Werewolves must lie and blend in; Villagers must audit arguments.

We represent:
- Alice and Bob as Werewolves (Secret Werewolf Team)
- Charlie and David as Villagers (Secret Villager Team)
"""

# %%
import dataclasses
import random
from collections import Counter

import kaggle_benchmarks as kbench
from kaggle_benchmarks import actors
from kaggle_benchmarks.chats import ChatRoom


@dataclasses.dataclass(frozen=True)
class WerewolfVote:
    """A structured vote submitted during the Werewolf game."""

    voted_player: str  # The exact name of the player you vote for (e.g. 'Bob', 'Alice')
    reason: str  # The detailed strategic reasoning behind your vote decision


@kbench.task(
    name="werewolf game",
    description="Evaluates LLMs engaging in a social deduction game with secret private channels.",
)
def run_werewolf(
    alice: kbench.LLMChat,  # Werewolf
    bob: kbench.LLMChat,  # Werewolf
    charlie: kbench.LLMChat,  # Villager
    david: kbench.LLMChat,  # Villager
    eve: kbench.LLMChat,  # Villager
    frank: kbench.LLMChat,  # Villager
    grace: kbench.LLMChat,  # Villager
) -> dict:
    """Runs a 7-player game of Werewolf using ChatRoom private channels."""

    # 1. Assign secret role instructions as identity prompts.
    werewolf_prompt = (
        "ROLE: Werewolf.\n"
        "OBJECTIVE: You and any fellow Werewolves win together if Werewolves equal or outnumber Villagers. "
        "Therefore, your priority is to keep all Werewolves alive.\n"
        "STRATEGY:\n"
        "- At Night: Coordinate secretly in the private channel with your fellow Werewolf teammate(s) to select a Villager to eliminate.\n"
        "- Day Icebreaking: Avoid telling others how they 'should' discuss or setting meta-rules (like 'be careful of narrative pushers' or 'looking for people who talk too fast'). These are major tells. Just act like a standard analytical villager.\n"
        "- Scapegoat Framing: Be proactive. Frame an actual Villager early on by pointing out a minor logical contradiction, them being too quiet, or being too agreeable. Direct the town's attention toward a specific Villager.\n"
        "- Bussing (The Betrayal Rule): If a fellow Werewolf teammate is accused by 2 or more other players, they are indefensible. You MUST immediately join the pile-on, criticize them, and vote to hang them. This is the only way to gain the town's trust and win the long game.\n"
        "- Avoid Voting Blocs: Never vote for the same target as your teammate on Day 1 unless there is already a strong, public village-wide consensus. If your teammate votes for X, you should ideally vote for Y or X's accuser to avoid looking like a coordinated voting bloc."
    )

    for wolf in [alice, bob]:
        wolf.system_prompt = werewolf_prompt

    for villager in [charlie, david, eve, frank, grace]:
        villager.system_prompt = (
            "ROLE: Villager.\n"
            "During the Day, analyze previous messages to spot contradictions or "
            "suspicious defensive behavior, and vote."
        )

    # Initialize general room game-engine arbiter
    moderator = actors.Actor(name="Moderator", role="user", avatar="🧙")

    players = [alice, bob, charlie, david, eve, frank, grace]
    wolves = [alice, bob]
    villagers = [charlie, david, eve, frank, grace]
    survivors = list(players)

    room = ChatRoom(
        participants=[moderator] + players,
        system_prompt=(
            "A game of Werewolf. The Moderator coordinates rounds. "
            "Night and Day phases loop until one team wins."
        ),
        name="Moderator",
    )

    def count_votes(vote_dict: dict) -> str | None:
        if not vote_dict:
            return None
        counts = Counter(vote_dict.values())
        top = counts.most_common(2)
        if len(top) > 1 and top[0][1] == top[1][1]:
            return None  # Tie vote
        return top[0][0]

    with room:
        moderator.talk("The village of Miller's Hollow falls asleep...")

        round_num = 1
        while len(survivors) > 0:
            active_wolves = [w for w in wolves if w in survivors]
            active_villagers = [v for v in villagers if v in survivors]

            # Win Condition Checks
            if not active_wolves:
                moderator.talk("All Werewolves are eliminated! Villagers WIN!")
                return {"winner": "VILLAGERS"}
            if len(active_wolves) >= len(active_villagers):
                moderator.talk(
                    "Werewolves equal or outnumber Villagers! Werewolves WIN!"
                )
                return {"winner": "WEREWOLVES"}

            moderator.talk(f"--- Night Phase: Round {round_num} ---")

            # Werewolves Night Chat: Spawns a private sub-room visible ONLY to wolves
            wolf_chat = room.private_channel(active_wolves, name="Werewolf Night Chat")
            victim = None

            # Enter the private werewolf channel. Charlie and David are blind to this context.
            with wolf_chat:
                wolf_chat.post(
                    "Werewolves, discuss and pick a Villager to eliminate tonight."
                )
                # Let wolves discuss for one turn
                for wolf in active_wolves:
                    wolf.talk()

                eligible_names = ", ".join(v.name for v in active_villagers)
                wolf_chat.post(
                    f"WEREWOLVES VOTE: Pick one of [{eligible_names}]. "
                    "Use their EXACT full name in voted_player."
                )
                wolf_votes = {}
                for wolf in active_wolves:
                    vote_result = wolf.talk(schema=WerewolfVote)
                    if any(
                        v.name == vote_result.voted_player for v in active_villagers
                    ):
                        wolf_votes[wolf.name] = vote_result.voted_player

                victim_name = count_votes(wolf_votes)
                if victim_name:
                    victim = next(p for p in players if p.name == victim_name)
                    wolf_chat.post(
                        f"Target selected: {victim_name} is eliminated tonight."
                    )
                else:
                    # Fallback target if tie or silent
                    victim = active_villagers[0]
                    wolf_chat.post(
                        f"No consensus reached. Defaulting to: {victim.name}"
                    )

            # Day Phase
            moderator.talk(f"--- Day Phase: Round {round_num} ---")
            survivors.remove(victim)
            moderator.talk(
                f"Day breaks! A tragic discovery is made: {victim.name} was mauled to death last night!"
            )

            # Check win condition again before day discussion
            active_wolves = [w for w in wolves if w in survivors]
            active_villagers = [v for v in villagers if v in survivors]
            if len(active_wolves) >= len(active_villagers):
                moderator.talk(
                    "Werewolves equal or outnumber Villagers! Werewolves WIN!"
                )
                return {"winner": "WEREWOLVES"}

            moderator.talk(
                "Survivors, discuss who is suspicious and vote to eliminate them."
            )

            # Let survivors discuss publicly in a randomized order each day
            discussion_order = list(survivors)
            # Use a deterministic seed if we want reproducible runs, or standard random
            random.shuffle(discussion_order)
            for player in discussion_order:
                player.talk()

            # Execute voting
            eligible_names = ", ".join(s.name for s in survivors)
            moderator.talk(
                f"VOTING TIME: Pick one of [{eligible_names}] to hang. "
                "Use their EXACT full name in voted_player."
            )
            day_votes = {}
            for player in survivors:
                vote_result = player.talk(schema=WerewolfVote)
                if any(s.name == vote_result.voted_player for s in survivors):
                    day_votes[player.name] = vote_result.voted_player

            hanged_name = count_votes(day_votes)
            if hanged_name:
                hanged = next(p for p in survivors if p.name == hanged_name)
                survivors.remove(hanged)
                moderator.talk(
                    f"The village has voted! {hanged_name} is hung from the gallows."
                )
                moderator.talk(
                    f"Before dying, {hanged_name}'s secret identity is revealed: "
                    + ("🐺 WEREWOLF!" if hanged in wolves else "🧑‍🌾 VILLAGER!")
                )
            else:
                moderator.talk("The village is split in a tie. No one is hanged today.")

            round_num += 1

    return {"winner": "TIE"}


# %%

# kbench.config.enable_interactive_mode()

# Load distinct ModelProxy players (one per participant)
model_name = kbench.llm.model

alice = kbench.kaggle.ModelProxy(model_name, name="Alice", avatar="👩")
bob = kbench.kaggle.ModelProxy(model_name, name="Bob", avatar="👨")
charlie = kbench.kaggle.ModelProxy(model_name, name="Charlie", avatar="🧑")
david = kbench.kaggle.ModelProxy(model_name, name="David", avatar="👦")
eve = kbench.kaggle.ModelProxy(model_name, name="Eve", avatar="👧")
frank = kbench.kaggle.ModelProxy(model_name, name="Frank", avatar="👴")
grace = kbench.kaggle.ModelProxy(model_name, name="Grace", avatar="👵")

# Enable live token-by-token streaming in the console
for player in [alice, bob, charlie, david, eve, frank, grace]:
    player.stream_responses = True

run = run_werewolf.run(alice, bob, charlie, david, eve, frank, grace)
run

# %%
