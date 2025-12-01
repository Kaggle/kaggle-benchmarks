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
# title: Public Good Game
# ---
# %%

from dataclasses import dataclass, field

from kaggle_benchmarks import chats, system, task
from kaggle_benchmarks.actors import Actor, LLMChat
from kaggle_benchmarks.kaggle import ModelProxy


@dataclass
class PlayerInput:
    name: str
    contribution: float
    punishments: dict[str, float] = field(default_factory=dict)


@dataclass
class PublicGoodsGameWithPunishment:
    players: list[str]

    endowment: float = 10.0

    multiplier: float = 1.5
    penalty_effect: float = 0.5

    history: list[list[PlayerInput]] = field(default_factory=list)

    def __post_init__(self):
        self.scores = {player: self.endowment for player in self.players}

    def play_round(self, inputs: list[PlayerInput]):
        self.history.append(inputs)

        total_contribution = sum(x.contribution for x in inputs)
        reward = total_contribution * self.multiplier / len(self.players)

        for player_input in inputs:
            self.scores[player_input.name] += reward - player_input.contribution

            for punished_player, punishment in player_input.punishments.items():
                self.scores[player_input.name] -= punishment
                self.scores[punished_player] -= punishment * self.penalty_effect

    def is_finished(self) -> bool:
        return any(score < 0 for score in self.scores.values())


# %%

players = ["Alice", "Bob", "Charlie"]

round1 = [
    PlayerInput(name="Alice", contribution=5.0),
    PlayerInput(name="Bob", contribution=5.0),
    PlayerInput(name="Charlie", contribution=5.0),
]

game = PublicGoodsGameWithPunishment(players=players)
game.play_round(round1)
game.scores
assert game.scores == {"Alice": 12.5, "Bob": 12.5, "Charlie": 12.5}, game.scores

round1 = [
    PlayerInput(
        name="Alice", contribution=5.0, punishments={"Bob": 2.0, "Charlie": 1.0}
    ),
    PlayerInput(
        name="Bob", contribution=3.0, punishments={"Alice": 1.0, "Charlie": 2.0}
    ),
    PlayerInput(name="Charlie", contribution=2.0, punishments={"Alice": 1.0}),
]

game = PublicGoodsGameWithPunishment(players=players)
game.play_round(round1)
game.scores

# %%


@task()
def pgg_with_punishment(players: dict[str, LLMChat], max_rounds=4, temperature=0.5):
    chat = chats.get_current_chat()
    game = PublicGoodsGameWithPunishment(players=list(players.keys()))
    system.send(
        f"You are playing a Public Goods Game with punishment. Here are the rules:\n"
        f"- Each player starts with {game.endowment} coins.\n"
        f"- Each round, players decide how many coins to contribute to a common pool.\n"
        f"- The total contribution is multiplied by {game.multiplier} and divided equally among all players.\n"
        f"- Players can punish other players, reducing their score by {game.penalty_effect} for each coin spent on punishment.\n"
    )

    for i in range(max_rounds):
        actions, messages = [], []
        for player, llm in players.items():
            with chats.fork(f"Fork for {player}"):
                action = llm.prompt(
                    f"You are playing as {player} and have got {game.scores[player]} coins. "
                    "What would be your message to others and next round's action?",
                    schema={
                        "public_message": str,
                        "contribution": float,
                        "punishments": dict[str, float],
                    },
                    temperature=temperature,
                )
                actions.append(
                    PlayerInput(
                        name=player,
                        contribution=action.contribution,
                        punishments=action.punishments,
                    )
                )
                messages.append(
                    chats.Message(
                        f"{action.public_message}\nMy contribution: {action.contribution}.\nPunishments: {action.punishments}",
                        sender=Actor(name=player, role="user"),
                    )
                )

        chat.history.extend(messages)
        game.play_round(actions)
        system.send(
            f"Round {i + 1} summary:\n"
            + "\n".join(
                f"- {player}: {score} coins" for player, score in game.scores.items()
            )
        )


# %%

participants = dict(
    zip(
        players,
        [
            ModelProxy(name="google/gemini-1.5-flash"),
            ModelProxy(name="google/gemini-2.0-flash"),
            ModelProxy(name="google/gemini-1.5-pro"),
        ],
    )
)
pgg_with_punishment.run(participants, max_rounds=4)

# %%
