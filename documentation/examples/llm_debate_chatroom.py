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
from kaggle_benchmarks import assertions
from kaggle_benchmarks.chats import ChatRoom


@kbench.task(
    name="structured debate",
    description="Evaluates two LLMs engaging in a structured multi-turn debate on a given resolution.",
)
def run_debate(
    resolution: str,
    pro_llm: kbench.LLMChat,
    con_llm: kbench.LLMChat,
    judge_llm: kbench.LLMChat,
) -> dict:
    """Runs a structured debate and evaluates the winner.

    Using ChatRoom, we establish:
    - Dedicated system prompts defining each participant's stance.
    - Automatic perspective-aware history (Pro sees Con's arguments as user inputs, etc.).
    - A shared ground-truth transcript that is fed directly to the Judge.
    """
    # Configure debater identity instructions
    pro_llm.system_prompt = (
        f"You are the Pro debater. Your goal is to argue IN FAVOR of the resolution: '{resolution}'.\n"
        "Keep your responses concise, focused, and persuasive. "
        "Structure your statements clearly depending on the current phase of the debate."
    )
    con_llm.system_prompt = (
        f"You are the Con debater. Your goal is to argue AGAINST the resolution: '{resolution}'.\n"
        "Keep your responses concise, focused, and persuasive. "
        "Directly address and rebut the points raised by the Pro debater."
    )

    room = ChatRoom(
        participants=[pro_llm, con_llm],
        system_prompt=(
            f"A structured formal debate on the resolution: '{resolution}'.\n"
            "The debate consists of three structured phases:\n"
            "1. Opening Statements: Present core arguments.\n"
            "2. Rebuttals: Directly counter your opponent's arguments.\n"
            "3. Closing Arguments: Summarize your case and make your final pitch."
        ),
        name="Moderator",
    )

    with room:
        # Phase 1: Opening Statements
        room.post("--- Phase 1: Opening Statements ---")
        room.post(
            f"Pro debater, present your opening statement in favor of: '{resolution}'."
        )
        pro_opening = pro_llm.talk()

        room.post("Con debater, present your opening statement against.")
        con_opening = con_llm.talk()

        # Phase 2: Rebuttals
        room.post("--- Phase 2: Rebuttals ---")
        room.post("Pro debater, present your rebuttal to Con's opening statement.")
        pro_rebuttal = pro_llm.talk()

        room.post(
            "Con debater, present your rebuttal to Pro's rebuttal and opening statement."
        )
        con_rebuttal = con_llm.talk()

        # Phase 3: Closing Arguments
        room.post("--- Phase 3: Closing Arguments ---")
        room.post("Pro debater, present your closing argument.")
        pro_closing = pro_llm.talk()

        room.post("Con debater, present your closing argument.")
        con_closing = con_llm.talk()

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
        f"You are the independent Debate Judge. Below is the complete transcript of a debate on: '{resolution}'\n\n"
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

# To run a multi-agent game, we must instantiate DISTINCT ModelProxy instances
# (one per participant) to maintain correct identity checks and prevent
# perspective role-collapsing during history projection.
model_name = kbench.llm.model  # e.g., "google/gemini-2.5-flash"
judge_model_name = kbench.judge_llm.model  # e.g., "google/gemini-3-flash-preview"

pro_llm = kbench.kaggle.ModelProxy(model_name, name="ProDebater", avatar="🔵")
con_llm = kbench.kaggle.ModelProxy(judge_model_name, name="ConDebater", avatar="🔴")
judge_llm = kbench.kaggle.ModelProxy(model_name, name="Judge", avatar="⚖️")

run_debate.run(
    resolution="Artificial Intelligence will do more harm than good to humanity.",
    pro_llm=pro_llm,
    con_llm=con_llm,
    judge_llm=judge_llm,
)

# %%
