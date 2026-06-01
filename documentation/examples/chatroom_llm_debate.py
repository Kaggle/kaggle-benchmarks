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

"""LLM Debate: A Structured Multi-Turn Debate Benchmark

This notebook demonstrates how to build a formal, multi-turn debate benchmark using ChatRoom.
Two LLM players (Pro and Con) engage in a structured debate on a controversial topic:
1. Opening Statements (Pro -> Con)
2. Rebuttals (Pro -> Con)
3. Closing Arguments (Pro -> Con)

A third independent LLM (Judge) evaluates the entire debate transcript to declare a winner.
"""

# %%
import kaggle_benchmarks as kbench
from kaggle_benchmarks import assertions, rooms


@kbench.task(
    name="structured debate",
    description="Evaluates two LLMs engaging in a structured multi-turn debate on a given topic.",
)
def run_debate(
    llm: kbench.LLMChat,
    judge_llm: kbench.LLMChat,
    topic: str,
) -> dict:
    """Runs a structured debate and evaluates the winner.

    Using ChatRoom, we establish:
    - Dedicated system prompts defining each participant's stance.
    - Automatic perspective-aware history (Pro sees Con's arguments as user inputs, etc.).
    - A shared ground-truth transcript that is fed directly to the Judge.
    """
    pro_prompt = (
        f"You are the Pro debater. Your goal is to argue IN FAVOR of the topic: '{topic}'.\n"
        "Keep your responses concise, focused, and persuasive. "
        "Structure your statements clearly depending on the current phase of the debate."
    )
    con_prompt = (
        f"You are the Con debater. Your goal is to argue AGAINST the topic: '{topic}'.\n"
        "Keep your responses concise, focused, and persuasive. "
        "Directly address and rebut the points raised by the Pro debater."
    )

    room = rooms.ChatRoom(
        system_prompt=(
            f"A structured formal debate on the topic: '{topic}'.\n"
            "The debate consists of three structured phases:\n"
            "1. Opening Statements: Present core arguments.\n"
            "2. Rebuttals: Directly counter your opponent's arguments.\n"
            "3. Closing Arguments: Summarize your case and make your final pitch."
        ),
        name="Moderator",
    )

    pro_llm = room.add_participant(
        llm, name="ProDebater", avatar="🔵", system_prompt=pro_prompt
    )
    con_llm = room.add_participant(
        llm, name="ConDebater", avatar="🔴", system_prompt=con_prompt
    )
    judge_llm = room.add_participant(judge_llm, name="Judge", avatar="⚖️")

    with room:
        # Phase 1: Opening Statements
        room.post("--- Phase 1: Opening Statements ---")
        room.post(
            f"Pro debater, present your opening statement in favor of: '{topic}'."
        )
        pro_opening = pro_llm.reply()

        room.post("Con debater, present your opening statement against.")
        con_opening = con_llm.reply()

        # Phase 2: Rebuttals
        room.post("--- Phase 2: Rebuttals ---")
        room.post("Pro debater, present your rebuttal to Con's opening statement.")
        pro_rebuttal = pro_llm.reply()

        room.post(
            "Con debater, present your rebuttal to Pro's rebuttal and opening statement."
        )
        con_rebuttal = con_llm.reply()

        # Phase 3: Closing Arguments
        room.post("--- Phase 3: Closing Arguments ---")
        room.post("Pro debater, present your closing argument.")
        pro_closing = pro_llm.reply()

        room.post("Con debater, present your closing argument.")
        con_closing = con_llm.reply()

        room.post(
            "The debate has concluded. The judge will now evaluate the transcript."
        )

    # Verification: Ensure participants spoke and didn't post empty strings
    for statement in [
        pro_opening,
        con_opening,
        pro_rebuttal,
        con_rebuttal,
        pro_closing,
        con_closing,
    ]:
        assertions.assert_true(
            len(statement) > 0, "Debate statement must not be empty."
        )

    # --- Judge Evaluation ---
    # The Judge reads the raw room messages (ground-truth transcript)
    transcript = "\n".join(str(m) for m in room.messages)

    judge_prompt = (
        f"You are the independent Debate Judge. Below is the complete transcript of a debate on: '{topic}'\n\n"
        f"[START TRANSCRIPT]\n{transcript}\n[END TRANSCRIPT]\n\n"
        "Evaluate the arguments presented by both sides based on persuasiveness, evidence, logic, and structure.\n"
        "Who won this debate, Pro or Con? Provide your decision and detailed reasoning.\n"
        "Your output must end with 'WINNER: PRO' or 'WINNER: CON'."
    )

    decision = judge_llm.prompt(
        judge_prompt,
        temperature=0.0,
    )

    winner = (
        "PRO"
        if "WINNER: PRO" in decision
        else "CON"
        if "WINNER: CON" in decision
        else "UNDECIDED"
    )

    return {
        "winner": winner,
        "reasoning": decision,
    }


# %%

# Run debate reusing default and judge models
run_debate.run(
    llm=kbench.llm,
    judge_llm=kbench.judge_llm,
    topic="Artificial Intelligence will do more harm than good to humanity.",
)

# %%
