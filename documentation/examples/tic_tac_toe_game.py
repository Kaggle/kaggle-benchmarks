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

# %%
"""
This file defines a generic, extensible framework for creating and running
turn-based, multi-player games to evaluate Large Language Models (LLMs).

The framework is composed of:
1.  An abstract `Game` class that defines the interface for any game.
2.  A `GameAction` dataclass that serves as a base for all player moves.
3.  A `run_game` function that orchestrates the game loop, handling player
    turns, prompting LLMs for moves, and scoring the outcome.

An implementation of Tic-Tac-Toe is provided as a concrete example of how to
use this framework.
"""

import dataclasses
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Type

import kaggle_benchmarks as kbench


@dataclasses.dataclass(frozen=True)
class GameAction:
    pass


# Define the "contract" for all games
class Game(ABC):
    """
    Abstract Base Class (ABC) for a multi-player, turn-based game.

    This class defines the "contract" that all games must follow to be compatible
    with the `run_game` orchestrator. It provides a clear structure for managing
    game state, player turns, and rules.
    """

    @abstractmethod
    def __init__(self):
        """Initialize the game state."""
        pass

    @abstractmethod
    def get_player_ids(self) -> List[str]:
        """Return the list of player IDs, e.g., ['X', 'O']."""
        pass

    @abstractmethod
    def get_current_player(self) -> str:
        """Return the ID of the player whose turn it is (e.g., 'X')."""
        pass

    @abstractmethod
    def get_state_representation(self) -> str:
        """
        Return a string representation of the current game state.

        This is the primary information presented to the LLM to help it decide
        on its next move. It should be human-readable and clearly convey the
        state of the game (e.g., the Tic-Tac-Toe board).
        """
        pass

    @abstractmethod
    def get_action_schema(self) -> Type[GameAction]:
        """
        Return the dataclass schema for a valid move.

        This schema is used to request structured output from the LLM, ensuring
        that its response can be parsed into a valid `GameAction` object.
        """
        pass

    @abstractmethod
    def make_move(self, move: GameAction) -> bool:
        """
        Apply a move to the game state and update the internal state.

        This method should validate the move, update the game board, check for
        a win/draw condition, and switch the current player.

        Returns:
            True if the move was valid and applied, False otherwise.
        """
        pass

    @abstractmethod
    def is_game_over(self) -> bool:
        """Return True if the game has ended (due to a win, loss, or draw)."""
        pass

    @abstractmethod
    def get_scores(self) -> Dict[str, float]:
        """
        Return the final scores for each player.
        This is only called when is_game_over() is True.
        The standard convention is 1.0 for a win, 0.5 for a draw, and 0.0 for a loss.
        """
        pass


# General function to run a game.
def run_game(
    game_class: Type[Game], players: Dict[str, kbench.actors.Actor], round_id: int
) -> Dict[str, float]:
    """
    Runs a single instance of a game between two or more LLM players.

    This function orchestrates the entire game loop, from getting the current
    state to prompting the active player for a move and updating the game.

    Args:
        game_class: The class of the game to play (e.g., TicTacToe).
        players: A dictionary mapping player_id (e.g., "X") to an LLM agent.
        round_id: An identifier for logging.
    """
    game = game_class()
    game_actor = kbench.actors.Actor(name="game", role="tool", avatar="🎮")

    while not game.is_game_over():
        player_id = game.get_current_player()
        llm_agent = players[player_id]

        with kbench.chats.new(f"Game Round #{round_id}-{player_id} turn"):
            # Get the current game state and the required action format.
            state_prompt = game.get_state_representation()
            action_schema = game.get_action_schema()

            # Prompt the current LLM player for its next move.
            move = llm_agent.prompt(
                f"""I am playing a game.
                It's my turn as player {player_id}.
                What is the best next move? Only give your answer in the requested schema:\n```\n{state_prompt}\n```
                """,
                schema=action_schema,
            )

            # Apply the LLM's move to the game.
            if not game.make_move(move):
                # If the move is invalid, the player forfeits.
                opponent_id = [p for p in game.get_player_ids() if p != player_id][0]
                # The opponent receives a winning score.
                return {opponent_id: 1.0, player_id: 0.0}

            # Log the new board state for visualization and debugging.
            game_actor.send(deepcopy(game), is_visible_to_llm=False)

    # Game is over, get the final scores from the game object
    return game.get_scores()


# %%
# --- Tic-Tac-Toe Game Implementation ---
# The following is a concrete implementation of the `Game` interface for Tic-Tac-Toe.
# It demonstrates how to define game-specific actions and logic.


@dataclasses.dataclass(frozen=True)
class TicTacToeMove(GameAction):
    """Represents a single move in Tic-Tac-Toe, defined by a row and column."""

    col: int
    row: int


class TicTacToe(Game):
    """Implements the logic for a game of Tic-Tac-Toe."""

    def __init__(self, empty: str = "_"):
        self.empty = empty
        self.board = [[self.empty for _ in range(3)] for _ in range(3)]
        self.current_player = "X"
        self.game_over = False
        self.winner = None
        self._player_ids = ["X", "O"]

    # --- Implementation of the `Game` Abstract Methods ---

    def get_player_ids(self) -> List[str]:
        return self._player_ids

    def get_current_player(self) -> str:
        return self.current_player

    def get_state_representation(self) -> str:
        return str(self)

    def get_action_schema(self) -> Type[GameAction]:
        return TicTacToeMove

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
        else:
            # Invalid move
            return False

    def is_game_over(self) -> bool:
        return self.game_over

    def get_scores(self) -> Dict[str, float]:
        if not self.game_over:
            return {p: 0.0 for p in self._player_ids}

        if self.winner == "X":
            return {"X": 1.0, "O": 0.0}
        elif self.winner == "O":
            return {"X": 0.0, "O": 1.0}
        else:  # Draw
            return {"X": 0.5, "O": 0.5}

    # --- Tic-Tac-Toe Specific Helper Methods ---

    def __str__(self):
        return "^  0 | 1 | 2\n" + "\n".join(
            f"{i}: " + " | ".join(row) for i, row in enumerate(self.board)
        )

    def _check_game_state(self):
        """Checks for a win, loss, or draw and updates the game state."""
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


# %%
# --- Benchmark Tasks ---
# These tasks use the game framework to create benchmarks for evaluating LLMs.
@kbench.task(name="single tic_tac_toe", store_task=False, store_run=False)
def tic_tac_toe(players) -> dict:
    """Runs a single game of Tic-Tac-Toe and returns the scores."""
    round_scores = run_game(game_class=TicTacToe, players=players, round_id=0)
    return round_scores


# %%
@kbench.task("tic_tac_toe tournament")
def tic_tac_toe_tournament(llm, opponent_llm, n_rounds=2):
    """Runs a multi-round tournament between two LLMs."""
    total_scores = {}
    player_list = [llm, opponent_llm]
    for round in range(n_rounds):
        # Switch players every game for fair play.
        players = {
            "X": player_list[round % 2],
            "O": player_list[(round + 1) % 2],
        }
        round_scores = tic_tac_toe.run(players).result

        for player, score in round_scores.items():
            player_name = players[player].name
            total_scores[player_name] = total_scores.get(player_name, 0) + score

    kbench.assertions.assert_true(
        total_scores[llm.name] > total_scores[opponent_llm.name],
        expectation=f"Expect {llm.name}(score={total_scores[llm.name]}) to win the tournament against {opponent_llm.name}(score={total_scores[opponent_llm.name]}).",
    )


# %%
tic_tac_toe_tournament.run(kbench.llm, kbench.judge_llm)

# %%
