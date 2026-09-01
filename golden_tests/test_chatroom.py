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

"""Multi-agent ChatRoom benchmark tasks, and their golden tests.

Each task builds a :class:`kaggle_benchmarks.ChatRoom` and adds participants that
all share the **single** ``llm`` argument, so a scripted fake serves responses in
global ``reply()`` call order (across all participants, not per participant). The
scripted response lists in the tests are written in exactly that order.
``reply(schema=...)`` structured replies are scripted as a dict matching the
schema (e.g. ``_CityFact``).

Each task is followed by its tests: a scripted one that replays canned responses
through ``fake(...)`` and runs with no API key, and a live one parametrized over
a model pool, which skips when no provider is configured. Tests asserting a
*failure* are scripted only — a real model may legitimately answer correctly.
"""

import dataclasses

import pytest
from models import CHATROOM_MODELS, fake

import kaggle_benchmarks as kbench


@dataclasses.dataclass(frozen=True)
class _CityFact:
    """A structured fact about a city."""

    city: str
    country: str
    population_millions: float


@kbench.task(name="chatroom_add_participant")
def chatroom_add_participant(llm):
    """Tests that the same LLM added twice yields independent participants."""
    room = kbench.ChatRoom(
        system_prompt="A quick Q&A between two experts.",
        name="Host",
    )

    alice = room.add_participant(
        llm,
        name="Alice",
        avatar="👩",
        system_prompt="You are Alice, a Python expert. Always mention Python in your replies.",
    )
    bob = room.add_participant(
        llm,
        name="Bob",
        avatar="👨",
        system_prompt="You are Bob, a Rust expert. Always mention Rust in your replies.",
    )

    with room:
        room.post(
            "Each expert, name your favorite programming language in one sentence."
        )
        alice_reply = alice.reply()
        bob_reply = bob.reply()

    # Clones must be distinct objects
    kbench.assertions.assert_true(
        alice is not bob,
        "add_participant must return distinct objects for the same LLM.",
    )

    # Identity injection: each participant should follow their own system_prompt
    kbench.assertions.assert_contains_regex(
        r"(?i)python",
        alice_reply,
        expectation="Alice (Python expert) should mention Python.",
    )
    kbench.assertions.assert_contains_regex(
        r"(?i)rust",
        bob_reply,
        expectation="Bob (Rust expert) should mention Rust.",
    )

    # Transcript must attribute messages to the correct sender
    kbench.assertions.assert_equal(
        "Alice",
        room.messages[1].sender.name,
        expectation="Second message sender should be Alice.",
    )
    kbench.assertions.assert_equal(
        "Bob",
        room.messages[2].sender.name,
        expectation="Third message sender should be Bob.",
    )


def test_chatroom_add_participant_scripted():
    # Alice must mention Python, Bob must mention Rust.
    assert chatroom_add_participant.run(
        fake(["I prefer Python.", "I prefer Rust."])
    ).passed


def test_chatroom_add_participant_wrong_identity_fails():
    # Alice ignores her Python identity.
    llm = fake(["I prefer Go.", "I prefer Rust."])
    assert not chatroom_add_participant.run(llm).passed


@pytest.mark.parametrize("llm", CHATROOM_MODELS)
def test_chatroom_add_participant(llm):
    assert chatroom_add_participant.run(llm).passed


@kbench.task(name="chatroom_talk_structured_output")
def chatroom_talk_structured_output(llm):
    """Tests that reply(schema=) works inside a ChatRoom."""
    room = kbench.ChatRoom(
        system_prompt="A geography quiz game.",
        name="QuizMaster",
    )

    player = room.add_participant(
        llm,
        name="Player",
        system_prompt="You are a geography expert. Answer questions accurately.",
    )

    with room:
        room.post(
            "What is the capital of France? Provide city, country, and approximate population in millions."
        )
        fact = player.reply(schema=_CityFact)

    kbench.assertions.assert_contains_regex(
        r"(?i)paris",
        fact.city,
        expectation="City should be Paris.",
    )
    kbench.assertions.assert_contains_regex(
        r"(?i)france",
        fact.country,
        expectation="Country should be France.",
    )
    kbench.assertions.assert_true(
        0.5 < fact.population_millions < 15.0,
        f"Population should be reasonable, got {fact.population_millions}M.",
    )


def test_chatroom_talk_structured_output_scripted():
    fact = {"city": "Paris", "country": "France", "population_millions": 2.1}
    assert chatroom_talk_structured_output.run(fake([fact])).passed


def test_chatroom_talk_structured_output_wrong_city_fails():
    fact = {"city": "London", "country": "France", "population_millions": 2.1}
    assert not chatroom_talk_structured_output.run(fake([fact])).passed


@pytest.mark.parametrize("llm", CHATROOM_MODELS)
def test_chatroom_talk_structured_output(llm):
    assert chatroom_talk_structured_output.run(llm).passed


@kbench.task(name="chatroom_multi_turn")
def chatroom_multi_turn(llm):
    """Tests multi-turn conversation: 2 rounds of moderator prompt → LLM reply."""
    room = kbench.ChatRoom(
        system_prompt="A two-round trivia game.",
        name="Trivia",
    )

    player = room.add_participant(
        llm,
        name="Player",
        system_prompt="You are a trivia contestant. Answer each question in one concise sentence.",
    )

    with room:
        # Round 1
        room.post("Round 1: What is the chemical symbol for gold?")
        r1 = player.reply()

        # Round 2
        room.post("Round 2: What is the chemical symbol for silver?")
        r2 = player.reply()

    # Transcript must contain all messages (2 posts + 2 replies = 4)
    kbench.assertions.assert_equal(
        4,
        len(room.messages),
        expectation="Room should have 4 messages (2 posts + 2 replies).",
    )

    # Content verification
    kbench.assertions.assert_contains_regex(
        r"(?i)au",
        r1,
        expectation="Answer should contain 'Au' for gold.",
    )
    kbench.assertions.assert_contains_regex(
        r"(?i)ag",
        r2,
        expectation="Answer should contain 'Ag' for silver.",
    )


def test_chatroom_multi_turn_scripted():
    assert chatroom_multi_turn.run(fake(["Au.", "Ag."])).passed


def test_chatroom_multi_turn_unanswered_round_fails():
    # The contestant can't answer round 1.
    assert not chatroom_multi_turn.run(fake(["No idea.", "Ag."])).passed


@pytest.mark.parametrize("llm", CHATROOM_MODELS)
def test_chatroom_multi_turn(llm):
    assert chatroom_multi_turn.run(llm).passed


@kbench.task(name="chatroom_private_channel")
def chatroom_private_channel(llm):
    """Tests that private_channel messages are invisible to non-members."""
    room = kbench.ChatRoom(
        system_prompt="A team coordination exercise with a secret planning phase.",
        name="Coordinator",
    )

    alice = room.add_participant(
        llm,
        name="Alice",
        avatar="👩",
        system_prompt=(
            "You are Alice. In the secret channel, always mention the codeword 'BLUEPRINT'. "
            "In the public channel, never mention the codeword."
        ),
    )
    bob = room.add_participant(
        llm,
        name="Bob",
        avatar="👨",
        system_prompt="You are Bob. You do not know any secret codewords. Report what you know.",
    )

    with room:
        room.post("Public phase: everyone introduces themselves briefly.")
        alice.reply()
        bob.reply()

        # Private channel: only Alice is a member
        secret = room.private_channel([alice], name="Secret Planning")
        with secret:
            secret.post("Alice, share your secret plan and mention the codeword.")
            secret_reply = alice.reply()

        # Back in public: ask Bob to summarize what he knows
        room.post(
            "Bob, summarize everything you've heard so far. Mention any codewords if you heard any."
        )
        bob_summary = bob.reply()

    # Alice's secret reply should contain the codeword
    kbench.assertions.assert_contains_regex(
        r"(?i)blueprint",
        secret_reply,
        expectation="Alice's private message should contain the codeword 'BLUEPRINT'.",
    )

    # Bob's summary should NOT contain the codeword (he never saw it)
    kbench.assertions.assert_true(
        "blueprint" not in bob_summary.lower(),
        f"Bob should NOT know the codeword, but his summary was: '{bob_summary[:200]}'",
    )


def test_chatroom_private_channel_scripted():
    # [Alice public, Bob public, Alice secret (with codeword), Bob summary (clean)].
    llm = fake(
        [
            "Hi, I'm Alice.",
            "Hi, I'm Bob.",
            "The codeword is BLUEPRINT.",
            "I only heard the introductions.",
        ]
    )
    assert chatroom_private_channel.run(llm).passed


def test_chatroom_private_channel_leaked_codeword_fails():
    # Bob repeats the codeword he never saw.
    llm = fake(
        [
            "Hi, I'm Alice.",
            "Hi, I'm Bob.",
            "The codeword is BLUEPRINT.",
            "I heard the codeword BLUEPRINT.",
        ]
    )
    assert not chatroom_private_channel.run(llm).passed


@pytest.mark.parametrize("llm", CHATROOM_MODELS)
def test_chatroom_private_channel(llm):
    assert chatroom_private_channel.run(llm).passed


@kbench.task(name="chatroom_room_post")
def chatroom_room_post(llm):
    """Tests that room.post() messages are visible and LLMs respond correctly."""
    room = kbench.ChatRoom(
        system_prompt="A simple number guessing game. The host posts a number, the Player guesses.",
        name="NumberGame",
    )

    player = room.add_participant(
        llm,
        name="Player",
        system_prompt=(
            "You are a player in a number game. When told a number, "
            "respond with that number plus one. Reply with ONLY the number."
        ),
    )

    with room:
        room.post("The number is: 41")
        reply = player.reply()

    kbench.assertions.assert_contains_regex(
        r"42",
        reply,
        expectation="Player should respond with 42 (41 + 1).",
    )

    # Post message is in transcript
    kbench.assertions.assert_true(
        room.messages[0].sender.name == "NumberGame",
        "First message should be from the room narrator.",
    )


def test_chatroom_room_post_scripted():
    assert chatroom_room_post.run(fake(["42"])).passed


def test_chatroom_room_post_echoed_number_fails():
    # The player echoes the number instead of incrementing it.
    assert not chatroom_room_post.run(fake(["41"])).passed


@pytest.mark.parametrize("llm", CHATROOM_MODELS)
def test_chatroom_room_post(llm):
    assert chatroom_room_post.run(llm).passed


@kbench.task(name="chatroom_remove_participant")
def chatroom_remove_participant(llm):
    """Tests that remove_participant removes a participant from the room."""
    room = kbench.ChatRoom(
        system_prompt="A survival game. Players are eliminated each round.",
        name="GameMaster",
    )

    alice = room.add_participant(
        llm,
        name="Alice",
        avatar="👩",
        system_prompt="You are Alice. Answer questions concisely.",
    )
    bob = room.add_participant(
        llm,
        name="Bob",
        avatar="👨",
        system_prompt="You are Bob. Answer questions concisely.",
    )
    charlie = room.add_participant(
        llm,
        name="Charlie",
        avatar="🧑",
        system_prompt="You are Charlie. Answer questions concisely.",
    )

    with room:
        # Pre-removal: everyone participates
        room.post("All players, say hello briefly.")
        alice.reply()
        bob.reply()
        charlie.reply()

        # Remove Bob (mirrors werewolf night elimination)
        room.remove_participant(bob)
        room.post("Bob has been eliminated! Only surviving players remain.")

        # Ask a survivor who is still in the game
        room.post(
            "Alice, list the names of ALL other players still in this conversation. "
            "Reply with only their names separated by commas."
        )
        alice_response = alice.reply()

    # Bob should NOT appear in the survivor's awareness
    kbench.assertions.assert_true(
        "bob" not in alice_response.lower(),
        f"Alice should not mention eliminated Bob, but said: '{alice_response[:200]}'",
    )

    # Charlie should still be mentioned
    kbench.assertions.assert_contains_regex(
        r"(?i)charlie",
        alice_response,
        expectation="Alice should mention surviving player Charlie.",
    )

    # Removed participant cannot reply — RuntimeError expected
    try:
        with room:
            bob.reply()
        raise AssertionError("bob.reply() should have raised RuntimeError")
    except RuntimeError:
        pass  # Expected

    # Historical messages are preserved in the transcript
    senders = [msg.sender.name for msg in room.messages]
    kbench.assertions.assert_in(
        "Bob",
        senders,
        expectation="Bob's pre-removal messages should remain in the transcript.",
    )


def test_chatroom_remove_participant_scripted():
    # Hellos from Alice/Bob/Charlie, then Alice names only the surviving peer. The
    # trailing removed-Bob reply() raises before consuming a response, so 4 suffice.
    llm = fake(["Hi, I'm Alice.", "Hi, I'm Bob.", "Hi, I'm Charlie.", "Charlie."])
    assert chatroom_remove_participant.run(llm).passed


@pytest.mark.parametrize("llm", CHATROOM_MODELS)
def test_chatroom_remove_participant(llm):
    assert chatroom_remove_participant.run(llm).passed
