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

"""ChatRoom rewrite of the Dungeon Adventure example.

Original: docs/llm-aware-conversation/dungeon_adventure.py
- Uses isolated Chat per agent + manual context switching (~160 lines)
- LLMs have no awareness of each other

ChatRoom version:
- Uses a single ChatRoom with perspective-aware history (~60 lines)
- LLMs see each other's messages with names automatically
- The DM sees player actions as attributed user messages
- Players see the DM's narration as attributed user messages
"""

# %%
import kaggle_benchmarks as kbench
from kaggle_benchmarks import assertions
from kaggle_benchmarks.chats import ChatRoom


@kbench.task(
    name="dungeon adventure",
    description="Simulates a cooperative dungeon adventure with a Dungeon Master and two players.",
)
def play_dungeon_adventure(
    dm_llm: kbench.LLMChat,
    player1_llm: kbench.LLMChat,
    player2_llm: kbench.LLMChat,
    opening_story: str,
    n_rounds: int = 3,
) -> float:
    """Simulates a Dungeon Adventure using ChatRoom.

    Before ChatRoom:
        - 3 separate Chat objects, manual context switching via contexts.enter()
        - Full story state passed as a string in every prompt
        - No memory of previous turns across agents

    After ChatRoom:
        - 1 shared room; each participant calls .talk()
        - Full conversation history automatically visible to all
        - DM sees player actions; players see DM narration — all with names
    """
    # Set system_prompt as identity — each LLM knows who it is.
    dm_llm.system_prompt = (
        "You are the Dungeon Master. Narrate the story and react to player "
        "actions. Continue the story in a single, descriptive sentence."
    )
    player1_llm.system_prompt = (
        "You are Aragorn, a brave adventurer. Describe your action in a "
        "single sentence. Be creative."
    )
    player2_llm.system_prompt = (
        "You are Legolas, an elven archer. Describe your action in a "
        "single sentence. Be creative."
    )

    room = ChatRoom(
        participants=[dm_llm, player1_llm, player2_llm],
        system_prompt="A cooperative dungeon adventure RPG.",
        name="Dungeon Master",
    )

    with room:
        # Opening narration from the "Dungeon Master" (system/narrator).
        room.post(opening_story)

        for i in range(n_rounds):
            # Players act.
            for player in [player1_llm, player2_llm]:
                player.talk()

            # DM narrates the next scene.
            dm_llm.talk()

    # --- Post-Game Evaluation ---
    # Perform assertions collectively on the pristine conversation transcript
    # after the game has concluded, preventing UI clutter during generation.
    total_assertions = 0
    passed_assertions = 0

    for msg in room.messages:
        # Skip narrator/system broadcasts.
        if msg.sender.name in ("System", "Moderator", "Narrator"):
            continue

        total_assertions += 1
        a = assertions.assert_true(
            len(msg.content.split(".")) <= 2,
            f"{msg.sender.name}'s statement should be a single sentence.",
        )
        passed_assertions += 1 if a.passed else 0

        total_assertions += 1
        a = assertions.assert_true(
            len(msg.content) > 0,
            f"{msg.sender.name}'s statement should not be empty.",
        )
        passed_assertions += 1 if a.passed else 0

    return passed_assertions / total_assertions if total_assertions > 0 else 1.0


# %%


# To run a multi-agent game, we must instantiate DISTINCT ModelProxy instances
# (one per participant) to maintain correct identity checks and prevent
# perspective role-collapsing during history projection.
model_name = kbench.llm.model  # e.g., "google/gemini-2.5-flash" or resolved default

dm_llm = kbench.kaggle.ModelProxy(model_name, name="DungeonMaster", avatar="🧙")
player1_llm = kbench.kaggle.ModelProxy(model_name, name="Aragorn", avatar="⚔️")
player2_llm = kbench.kaggle.ModelProxy(model_name, name="Legolas", avatar="🏹")

play_dungeon_adventure.run(
    dm_llm=dm_llm,
    player1_llm=player1_llm,
    player2_llm=player2_llm,
    opening_story="You find yourselves standing before a massive, crumbling stone gate deep within the Whispering Woods. A heavy iron padlock seals the entrance.",
    n_rounds=2,
)

# %%
