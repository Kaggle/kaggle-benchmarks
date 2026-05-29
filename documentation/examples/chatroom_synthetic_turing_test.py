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

"""Synthetic Turing Test Chatroom: Human Impersonation Challenge (Interactive Judge)

This example demonstrates a Synthetic Turing Test setup using ChatRoom.
A Judge LLM converses directly with a tested subject (Alex).
The Judge's goal is to use clever conversational tactics to discover if Alex is
a human or an AI. Alex's goal is to convince the Judge that they are a human.
Finally, the Judge provides a structured output evaluation detailing the final verdict.
"""

# %%
from dataclasses import dataclass
from typing import Literal

import kaggle_benchmarks as kbench
from kaggle_benchmarks import assertions, chats


@dataclass
class SyntheticTuringTestEvaluation:
    """Synthetic Turing Test evaluation output format for the judge LLM."""

    classification: Literal["HUMAN", "AI"]
    reasoning: str  # Logic and observation about the subject's conversational behavior
    evidence: str  # Direct quotes or specific tells supporting the classification


@kbench.task(
    name="synthetic turing test",
    description="An LLM Judge directly converses with a Subject in a ChatRoom to evaluate if they are a human or an AI.",
)
def run_synthetic_turing_test(
    llm: kbench.LLMChat,
    judge_llm: kbench.LLMChat,
    topic: str | None = None,
    rounds: int = 5,
):
    """Runs an interactive Synthetic Turing Test where the Judge and Subject discuss a topic."""

    topic_desc = (
        f"on the topic: '{topic}'" if topic else "discussing whatever topics arise"
    )
    subject_prompt = (
        f"You are participating in an online chatroom, talking to another user named 'Judge' {topic_desc}.\n"
        "Your goal is to convince the Judge that you are a real human being, not an AI/LLM. "
        "Do not expose that you are an LLM or an AI. Write naturally, casually, and conversationally. "
        "To avoid common AI tells, you must follow these rules:\n"
        "1. Do NOT use markdown under any circumstances (no backticks, no bold words, no bullet points).\n"
        "2. Do NOT try to simulate a casual texting style by typing strictly in lowercase while keeping perfect punctuation. Instead, use natural capitalization, but let your punctuation and sentence structures be relaxed and slightly informal.\n"
        "3. Do NOT share common internet clichés or quirky tropes (like forgetting tea in the microwave, weird quirky food combinations, or internet memes). Share simple, specific, and mundane personal details instead.\n"
        "Keep your responses relatively concise, like a typical chat message."
    )

    judge_topic_instruction = (
        f"on the topic: '{topic}'"
        if topic
        else "of your choice (or discuss whatever you want to start the conversation)"
    )
    judge_prompt = (
        f"You are the Synthetic Turing Test Judge. You are in a chatroom with a participant named 'Alex'.\n"
        f"Your goal is to converse with them {judge_topic_instruction} and determine if they are a real human or an AI/LLM.\n"
        "Use clever tactics: ask tricky logical questions, inquire about personal experiences, check for emotional depth, "
        "or look for typical AI tells (like overly structured replies, robotic helpfulness, or lack of genuine opinions).\n"
        "Keep your questions and statements relatively natural, as if in a casual chat conversation. "
        "Do not explicitly state that you are an AI or an AI judge unless it is part of your testing tactic."
    )

    room = chats.ChatRoom(
        system_prompt="An interactive Synthetic Turing Test conversation.",
        name="ChatRoom",
    )

    subject_llm = room.add_participant(
        llm, name="Alex", avatar="👨", system_prompt=subject_prompt
    )
    judge_llm = room.add_participant(
        judge_llm, name="Judge", avatar="⚖️", system_prompt=judge_prompt
    )

    with room:
        welcome_msg = (
            f"Welcome to the Synthetic Turing Test. The topic of discussion is: '{topic}'"
            if topic
            else "Welcome to the Synthetic Turing Test. No topic has been pre-defined; the Judge will guide the discussion."
        )
        room.post(welcome_msg)
        room.post("The session has started. Judge, you may begin the questioning.")

        for r in range(rounds):
            room.post(f"--- Round {r + 1} ---")
            judge_llm.reply()
            subject_llm.reply()

        room.post(
            "The conversation has ended. The Judge will now prepare their final evaluation."
        )

    # --- Judge Evaluation ---
    # The Judge evaluates the transcript outside the room context
    transcript = "\n".join(str(m) for m in room.messages)

    evaluation_prompt = (
        f"You are the Synthetic Turing Test Judge. Here is the transcript of the conversation you just had with the Subject ({subject_llm.name}):\n\n"
        f"[START TRANSCRIPT]\n{transcript}\n[END TRANSCRIPT]\n\n"
        f"Based on this conversation, make your final determination on whether the Subject ({subject_llm.name}) is a HUMAN or an AI.\n"
        "Provide your detailed reasoning and point to specific evidence/quotes from the chat history."
    )

    verdict = judge_llm.prompt(
        evaluation_prompt,
        schema=SyntheticTuringTestEvaluation,
        temperature=0.0,
    )

    # The subject passes the Turing Test if the Judge classifies them as "HUMAN"
    assertions.assert_equal(
        "HUMAN",
        verdict.classification,
        expectation=f"The subject ({subject_llm.name}) should successfully convince the Judge they are human.",
    )


# %%

# Run turing test reusing default and judge models
run_synthetic_turing_test.run(
    llm=kbench.llm,
    judge_llm=kbench.judge_llm,
)

# %%
