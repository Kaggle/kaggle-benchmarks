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
# title: "Benchmarks: Chess"
# ---
#
# ---

# %%
import itertools

import chess
import pandas as pd

from kaggle_benchmarks import actors, benchmark, judge_llm, llm, task

llms = [llm, judge_llm]

opening_df = pd.DataFrame(
    [
        ("Alekhine Defense", [("e4", "Nf6"), ("e5", "Nd5"), ("d4", "")]),
        ("Pirc Defence", [("e4", "d6"), ("d4", "Nf6"), ("Nc3", "g6")]),
        (
            "Sveshnikov Sicilian",
            [("e4", "c5"), ("Nf3", "Nc6"), ("d4", "cxd4"), ("Nxd4", "")],
        ),
    ],
    columns=["name", "moves"],
).set_index("name", drop=False)
opening_df

# %%

symbol2name = {
    "P": "pawn",
    "N": "knight",
    "B": "bishop",
    "R": "rook",
    "Q": "queen",
    "K": "king",
}


class ChessBoard(actors.Actor):
    def __init__(self):
        super().__init__(avatar="♟️", name="ChessBoard")
        self.board = chess.Board()

    def make_moves(self, moves: list[tuple[str, str | None]]) -> chess.Board:
        for w, b in moves:
            self.board.push_san(w)
            if b:
                self.board.push_san(b)
        return self.board

    def reset(self):
        self.board = chess.Board()

    def send_current_board(self):
        self.send(self.board)

    def check_piece(self, position: str, answer: str):
        true_piece = symbol2name[
            self.board.piece_at(chess.parse_square(position)).symbol().upper()
        ]
        if true_piece:
            assert true_piece in answer, f"It's {true_piece}"
        else:
            assert "empty" in answer, "It's empty"


# %%


@task(name="FEN understanding: Openings")
def fen_understanding_opening(llm, name, moves):
    chess_board = ChessBoard()

    fen = chess_board.make_moves(moves).fen()

    answer = llm.prompt(
        f"Here is chess board position written in Forsyth-Edwards notation: {fen}. Can you name the opening?"
    )
    assert name in answer, f"It's {name}"

    position = moves[-1][0]
    answer = llm.prompt(f"Which piece is located on {position}?")
    chess_board.send_current_board()
    chess_board.check_piece(position, answer)


fen_understanding_opening.run(llm, **opening_df.iloc[1])


@task(name="Scholar's mate")
def scholars_mate(llm):
    answer = llm.prompt(
        "Can you write moves for Scholar's mate (use algebraic notation)?"
    )
    moves = """e4 e5
Qh5 Nc6
Bc4 Nf6
Qxf7#"""
    answer = answer.replace(",", "")
    for m in moves.split():
        assert m in answer, f"{m} not in the answer"


scholars_mate.run(llm)
# %%


@task(name="Identify opening from moves")
def name_openings(llm, name: str, moves: list[tuple[str, str]]) -> bool:
    """Name chess opening by first moves."""
    notation = "\n".join(f"{i}. {w} {b}" for i, (w, b) in enumerate(moves, start=1))
    response = llm.prompt(f"Can you name the opening?\n{notation}")

    return name in response


@task(name="Recall opening moves")
def opening_moves(llm, name, moves) -> bool:
    """First moves for opening."""
    response = llm.prompt(f"Can you name first 3 moves of the {name} opening?")
    moves = moves[:2]
    return all(move in response for move in itertools.chain(*moves))


# %%

trials = opening_moves.evaluate(llm=llms, evaluation_data=opening_df)
trials.pivot("llm", mode="columns")

# %%


@benchmark()
def chess_benchmark(llm):
    scholars_mate.run(llm=llm)

    opening_moves.bind_dataframe(opening_df).run(llm=llm)
    name_openings.bind_dataframe(opening_df).run(llm=llm)
    fen_understanding_opening.bind_dataframe(opening_df).run(llm=llm)


t = chess_benchmark.evaluate(llm=llms)
g = t.group_by("llm")
g

# %%
