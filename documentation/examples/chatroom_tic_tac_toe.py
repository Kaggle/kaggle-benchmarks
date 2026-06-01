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

"""ChatRoom rewrite of the Tic-Tac-Toe example.

Original: docs/llm-aware-conversation/game_tic_tac_toe.py
- Creates a brand new Chat per turn with no memory of previous moves
- LLMs don't see each other's moves
- Game state manually serialized into each prompt

ChatRoom version:
- Game engine is an Actor that posts board state
- LLMs see the full move history with perspective awareness
- talk(schema=TicTacToeMove) returns structured output directly
"""

# %%
import dataclasses

import kaggle_benchmarks as kbench
from kaggle_benchmarks import rooms

# --- Game Logic (unchanged from original) ---


@dataclasses.dataclass(frozen=True)
class TicTacToeMove:
    """Represents a single move in Tic-Tac-Toe."""

    col: int
    row: int


class TicTacToe:
    """Implements the logic for a game of Tic-Tac-Toe."""

    def __init__(self, empty: str = "_"):
        self.empty = empty
        self.board = [[self.empty for _ in range(3)] for _ in range(3)]
        self.current_player = "X"
        self.game_over = False
        self.winner = None

    def make_move(self, move: TicTacToeMove) -> bool:
        if self.game_over:
            return False
        row, col = move.row, move.col
        if 0 <= row < 3 and 0 <= col < 3 and self.board[row][col] == self.empty:
            self.board[row][col] = self.current_player
            self._check_game_state()
            if not self.game_over:
                self.current_player = "O" if self.current_player == "X" else "X"
            return True
        return False

    def is_game_over(self) -> bool:
        return self.game_over

    def get_scores(self) -> dict[str, float]:
        if self.winner == "X":
            return {"X": 1.0, "O": 0.0}
        elif self.winner == "O":
            return {"X": 0.0, "O": 1.0}
        else:
            return {"X": 0.5, "O": 0.5}

    def __str__(self):
        return "^  0 | 1 | 2\n" + "\n".join(
            f"{i}: " + " | ".join(row) for i, row in enumerate(self.board)
        )

    def _check_game_state(self):
        for row in self.board:
            if all(cell == self.current_player for cell in row):
                self.winner = self.current_player
                self.game_over = True
                return
        for col in range(3):
            if all(self.board[row][col] == self.current_player for row in range(3)):
                self.winner = self.current_player
                self.game_over = True
                return
        if all(self.board[i][i] == self.current_player for i in range(3)):
            self.winner = self.current_player
            self.game_over = True
            return
        if all(self.board[i][2 - i] == self.current_player for i in range(3)):
            self.winner = self.current_player
            self.game_over = True
            return
        if all(cell != self.empty for row in self.board for cell in row):
            self.game_over = True
            self.winner = None


# --- ChatRoom-based Game Runner ---


@kbench.task(
    name="tic-tac-toe chatroom",
    description="Runs a game of Tic-Tac-Toe inside a ChatRoom and returns the scores.",
)
def run_tic_tac_toe(
    llm: kbench.LLMChat,
    judge_llm: kbench.LLMChat,
) -> dict:
    """Runs Tic-Tac-Toe using ChatRoom.

    Before ChatRoom:
        - Brand new Chat per turn — zero memory of previous moves
        - Full game state manually serialized into each prompt
        - LLMs unaware they are playing against each other

    After ChatRoom:
        - Game engine is an Actor that posts board state
        - Players see the full history (own moves as "assistant", peer as "user")
        - reply(schema=TicTacToeMove) returns structured output directly
    """
    room = rooms.ChatRoom(
        system_prompt=(
            "A game of Tic-Tac-Toe. The Game participant posts the current "
            "board state. Players take turns making moves."
        ),
        name="Game",
    )

    player_x = room.add_participant(
        llm,
        name="PlayerX",
        avatar="❌",
        system_prompt="You are player X in Tic-Tac-Toe. When it's your turn, respond with your move (row and col, 0-indexed).",
    )
    player_o = room.add_participant(
        judge_llm,
        name="PlayerO",
        avatar="⭕",
        system_prompt="You are player O in Tic-Tac-Toe. When it's your turn, respond with your move (row and col, 0-indexed).",
    )

    players = {"X": player_x, "O": player_o}

    game = TicTacToe()

    with room:
        room.post(f"Game starts! Initial board:\n{game}")

        while not game.is_game_over():
            current = players[game.current_player]
            move = current.reply(schema=TicTacToeMove)

            if not game.make_move(move):
                # Invalid move — opponent wins by forfeit.
                opponent_id = "O" if game.current_player == "X" else "X"
                return {opponent_id: 1.0, game.current_player: 0.0}

            room.post(f"Board after move:\n{game}")

    return game.get_scores()


# %%

run_tic_tac_toe.run(llm=kbench.llm, judge_llm=kbench.judge_llm)

# %%
