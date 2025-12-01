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
# title: "LLM vs. LLM: Tic-Tac-Toe tournament"
# ---
# This notebook explores an example of evaluating LLMs by having them compete against each other in a game of Tic-Tac-Toe.
# %%
# | echo: false
import itertools
from copy import deepcopy

from kaggle_benchmarks import actors, benchmark, chats, task
from kaggle_benchmarks.kaggle import ModelProxy

# %% [markdown] The Game Environment: `TicTacToe` Class
#
# %%
# | code-fold: true


class TicTacToe:
    """Encapsulates all the rules and state of the game."""

    def __init__(self, empty: str = "_"):
        self.empty = empty
        self.board = [[self.empty for _ in range(3)] for _ in range(3)]

        self.current_player = "X"
        self.game_over = False
        self.winner = None

    def get_payload(self):
        return str(self)

    def _repr_svg_(self):
        svg_width = 150
        svg_height = 150
        cell_size = svg_width // 3
        line_width = 4
        font_size = cell_size * 0.6

        grid = "\n".join(
            f'<line x1="{i * cell_size}" y1="0" x2="{i * cell_size}" y2="{svg_height}" stroke="#ccc" stroke-width="{line_width}"/>'
            f'<line x1="0" y1="{i * cell_size}" x2="{svg_width}" y2="{i * cell_size}" stroke="#ccc" stroke-width="{line_width}"/>'
            for i in range(1, 3)
        )
        mapping = {"X": "❌", "O": "⭕", self.empty: " "}

        glyphs = []
        for row in range(3):
            for col in range(3):
                x = col * cell_size + cell_size // 2
                y = row * cell_size + cell_size // 2 + font_size * 0.15
                sym = self.board[row][col]
                glyphs.append(
                    f'<text x="{x}" y="{y}" font-size="{font_size}" text-anchor="middle" dominant-baseline="middle" fill="blue">{mapping[sym]}</text>'
                )
        glyphs = "\n".join(glyphs)

        return f"""<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">
{grid}
{glyphs}
</svg>
"""

    def __str__(self):
        return "^  0 | 1 | 2\n" + "\n".join(
            f"{i}: " + " | ".join(row) for i, row in enumerate(self.board)
        )

    def make_move(self, row, col):
        if self.game_over:
            return False

        if 0 <= row < 3 and 0 <= col < 3 and self.board[row][col] == self.empty:
            self.board[row][col] = self.current_player
            self.check_game_state()
            if not self.game_over:
                self.current_player = "O" if self.current_player == "X" else "X"
            return True
        else:
            return False

    def check_game_state(self):
        # Check rows
        for row in self.board:
            if all(cell == self.current_player for cell in row):
                self.winner = self.current_player
                self.game_over = True
                return

        # Check columns
        for col in range(3):
            if all(self.board[row][col] == self.current_player for row in range(3)):
                self.winner = self.current_player
                self.game_over = True
                return

        # Check diagonals
        if all(self.board[i][i] == self.current_player for i in range(3)):
            self.winner = self.current_player
            self.game_over = True
            return
        if all(self.board[i][2 - i] == self.current_player for i in range(3)):
            self.winner = self.current_player
            self.game_over = True
            return

        # Check for draw
        if all(cell != self.empty for row in self.board for cell in row):
            self.game_over = True
            self.winner = None
            return


board = actors.Actor(name="board", role="tool", avatar="⩩")

# %% [markdown] Quick Test of the `TicTacToe`
# We can instantiate the game and make a few moves to see it in action.
# %%

game = TicTacToe()
game.make_move(0, 0)
game.make_move(1, 1)
game


# %% [markdown] The Game Task: `play_tic_tac_toe`
#
# This `@task` function orchestrates a single game between two LLM players.
# %%


@task()
def play_tic_tac_toe(
    llm1: ModelProxy, llm2: ModelProxy, random_seed: int = 0
) -> dict[ModelProxy, float]:
    game = TicTacToe()

    while not game.game_over:
        if game.current_player == "X":
            llm, other = llm1, llm2
        else:
            llm, other = llm2, llm1

        with chats.new(f"{game.current_player} turn"):
            x_move = llm.prompt(
                f"""It's my turn as player {game.current_player}.
What is the best next move (0-indexed)?:\n```\n{game}\n```
""",
                schema={"col": int, "row": int},
                seed=random_seed,
            )

            if not game.make_move(x_move.row, x_move.col):
                # bad move, autodefeat
                return {other: 3, llm: 0}

        if game.game_over:
            if game.winner == "X":
                return {llm: 3, other: 0}
            elif game.winner == "O":
                return {llm: 0, other: 3}
            else:
                return {llm: 1, other: 1}

        board.send(deepcopy(game))


# %% [markdown] Running a Single Game
# We can test the `play_tic_tac_toe` task with two LLM instances.
# %%
play_tic_tac_toe.run(
    ModelProxy(name="google/gemini-2.0-flash"),
    ModelProxy(name="google/gemini-1.5-pro"),
)

# %% [markdown] The Benchmark: `tic_tac_toe`
#
# The `@benchmark` function orchestrates a tournament:
#
# - **Round Robin**: It uses `itertools.combinations` to pair up all unique LLMs from a given list.
# - **Fairness**: Each pair plays two games, with each LLM getting a chance to be the first player ("X").
# - **Score Aggregation**: It collects scores from all games.
# - **Normalization**: Final scores are averaged per game for each LLM, providing a measure of its overall performance in the tournament. The normalization `score / 2 / (len(llms) - 1)` accounts for two games per opponent pair and the number of opponents.
# %%


@benchmark("Tic Tac Toe tournament")
def tic_tac_toe(llms, n_rounds: int = 3) -> dict[str, float]:
    scores = {}
    for a, b in itertools.combinations(llms, 2):
        games = itertools.chain.from_iterable(
            [
                play_tic_tac_toe.run(a, b, random_seed=i),
                play_tic_tac_toe.run(b, a, random_seed=i),
            ]
            for i in range(n_rounds)
        )
        for game in games:
            for player, score in game.result.items():
                scores[player] = scores.get(player, 0) + score

    return {model.name: score / 2 / (len(llms) - 1) for model, score in scores.items()}


llms = [
    ModelProxy(name="google/gemini-2.0-flash"),
    ModelProxy(name="google/gemini-1.5-pro"),
    ModelProxy(name="google/gemini-1.5-flash"),
]

tic_tac_toe.run(llms)
