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

import json

import pytest

from kaggle_benchmarks import actors, chats
from kaggle_benchmarks.chats import ChatRoom
from kaggle_benchmarks.messages import Message
from tests.mocks import MockedChat

# --- Context Manager ---


def test_chatroom_sets_active_context():
    """with room: must make the ChatRoom the active chat."""
    alice = MockedChat.from_contents(["hi"], name="Alice", cycle=True)
    room = ChatRoom(participants=[alice])
    with room:
        assert chats.get_current_chat() is room
    # After exit, room is no longer current.
    assert chats.get_current_chat() is not room


# --- System Prompt Enrichment ---


def test_system_prompt_includes_roster():
    alice = MockedChat.from_contents(["x"], name="Alice")
    alice.system_prompt = "argues FOR"
    bob = MockedChat.from_contents(["x"], name="Bob")
    bob.system_prompt = "argues AGAINST"
    room = ChatRoom(participants=[alice, bob], system_prompt="A debate.")

    prompt = room._build_system_prompt(alice)
    assert "You are Alice" in prompt
    assert "Bob" in prompt
    assert "argues AGAINST" in prompt
    assert "A debate." in prompt
    assert "argues FOR" in prompt  # alice's personal prompt


def test_system_prompt_room_identity():
    alice = MockedChat.from_contents(["x"], name="Alice")
    room = ChatRoom(participants=[alice], name="Narrator")

    prompt = room._build_system_prompt(alice)
    assert 'Note on "Narrator"' in prompt


# --- Perspective Projection ---


def test_perspective_self_is_assistant():
    alice = MockedChat.from_contents(["x"], name="Alice")
    bob = MockedChat.from_contents(["x"], name="Bob")
    room = ChatRoom(participants=[alice, bob])

    room.history = [
        Message(sender=alice, content="hello from alice"),
        Message(sender=bob, content="hello from bob"),
    ]

    perspective = room._build_perspective(alice)
    assert perspective[0].sender.role == "assistant"
    assert perspective[0].content == "hello from alice"
    assert perspective[1].sender.role == "user"
    assert "[Bob]:" in perspective[1].content


def test_perspective_does_not_mutate_originals():
    alice = MockedChat.from_contents(["x"], name="Alice")
    room = ChatRoom(participants=[alice])
    original = Message(sender=alice, content="original")
    room.history = [original]

    perspective = room._build_perspective(alice)
    assert perspective[0] is not original  # new object
    assert original.sender is alice  # original unchanged


# --- talk() Primitives ---


def test_llmchat_talk_outside_room_raises():
    alice = MockedChat.from_contents(["x"], name="Alice")
    with pytest.raises(RuntimeError, match="ChatRoom context"):
        alice.talk()


def test_actor_talk_outside_room_raises():
    game = actors.Actor(name="Game")
    with pytest.raises(RuntimeError, match="ChatRoom context"):
        game.talk("hello")


def test_llmchat_talk_appends_to_room():
    alice = MockedChat.from_contents(["I agree!"], name="Alice", cycle=True)
    bob = MockedChat.from_contents(["x"], name="Bob")
    room = ChatRoom(participants=[alice, bob])

    with room:
        room.post("Topic: AI safety")
        result = alice.talk()

    assert result == "I agree!"
    # Ground truth should have: system post + alice's response.
    assert len(room.messages) == 2
    assert room.messages[1].sender.name == "Alice"
    assert room.messages[1].content == "I agree!"


def test_actor_talk_appends_to_room():
    game = actors.Actor(name="Game")
    alice = MockedChat.from_contents(["x"], name="Alice")
    room = ChatRoom(participants=[game, alice])

    with room:
        game.talk("Board: X|O|_")

    assert len(room.messages) == 1
    assert room.messages[0].sender is game
    assert room.messages[0].content == "Board: X|O|_"


# --- End-to-End: Mini Debate ---


def test_mini_debate_two_rounds():
    """Two LLMs debate for 2 rounds with full perspective-aware history."""
    pro = MockedChat.from_contents(["AI is great!", "AI saves lives!"], name="Pro")
    con = MockedChat.from_contents(["AI is risky!", "AI needs regulation!"], name="Con")
    room = ChatRoom(
        participants=[pro, con],
        system_prompt="A debate on AI.",
    )

    with room:
        room.post("Topic: AI")
        pro.talk()
        con.talk()
        pro.talk()
        con.talk()

    # Ground truth: 1 system post + 4 talk messages.
    assert len(room.messages) == 5

    # Verify Pro's second invocation received the full perspective history.
    # invocations[1] is the second call to pro.invoke()
    messages_sent, kwargs = pro.invocations[1]
    # Pro should see: system post (user), Pro round 1 (assistant), Con round 1 (user)
    assert len(messages_sent) == 3
    assert messages_sent[0].sender.role == "user"  # system post
    assert messages_sent[1].sender.role == "assistant"  # Pro's own message
    assert messages_sent[2].sender.role == "user"  # Con's message
    assert "[Con]:" in messages_sent[2].content


# --- Visibility Filtering (Phase 2 MVP) ---


def test_visible_to_filters_messages():
    alice = MockedChat.from_contents(["x"], name="Alice")
    bob = MockedChat.from_contents(["x"], name="Bob")
    room = ChatRoom(participants=[alice, bob])

    with room:
        room.post("Public message")
        room.post("Secret for Alice", visible_to=[alice])

    perspective_alice = room._build_perspective(alice)
    perspective_bob = room._build_perspective(bob)

    assert len(perspective_alice) == 2  # sees both
    assert len(perspective_bob) == 1  # sees only public


# --- Issue 1 Fix: _meta["chat"] points to room, not temp ---


def test_meta_chat_points_to_room():
    alice = MockedChat.from_contents(["response"], name="Alice", cycle=True)
    room = ChatRoom(participants=[alice])

    with room:
        alice.talk()

    room_msg = room.messages[0]
    assert room_msg._meta.get("chat") is room


# --- system_prompt via from_contents ---


def test_system_prompt_via_from_contents():
    alice = MockedChat.from_contents(["x"], name="Alice", system_prompt="argues FOR")
    assert alice.system_prompt == "argues FOR"


# --- Example-Level: Mini Dungeon Adventure ---


def test_dungeon_adventure_chatroom():
    """End-to-end dungeon adventure with DM + 2 players."""
    dm = MockedChat.from_contents(
        ["The dragon roars!", "The treasure is found!"],
        name="DungeonMaster",
        system_prompt="You are the Dungeon Master.",
    )
    aragorn = MockedChat.from_contents(
        ["I draw my sword!", "I open the chest!"],
        name="Aragorn",
        system_prompt="You are Aragorn, a brave adventurer.",
    )
    legolas = MockedChat.from_contents(
        ["I fire an arrow!", "I check for traps!"],
        name="Legolas",
        system_prompt="You are Legolas, an elven archer.",
    )

    room = ChatRoom(
        participants=[dm, aragorn, legolas],
        system_prompt="A cooperative dungeon adventure RPG.",
        name="Dungeon Master",
    )

    with room:
        room.post("You find yourselves in a dimly lit tavern.")

        # Round 1: players act, then DM narrates
        aragorn.talk()
        legolas.talk()
        dm.talk()

        # Round 2
        aragorn.talk()
        legolas.talk()
        dm.talk()

    # Ground truth: 1 post + 6 talks = 7 messages.
    assert len(room.messages) == 7

    # Verify DM saw both players' actions in round 1 perspective.
    dm_msgs_r1, dm_kwargs_r1 = dm.invocations[0]
    # DM should see: post(user) + aragorn(user) + legolas(user) = 3 messages
    assert len(dm_msgs_r1) == 3
    assert "[Aragorn]:" in dm_msgs_r1[1].content
    assert "[Legolas]:" in dm_msgs_r1[2].content

    # Verify Aragorn's round 2 saw correct perspective.
    aragorn_msgs_r2, _ = aragorn.invocations[1]
    # Aragorn round 2: post + aragorn r1(assistant) + legolas r1(user)
    #   + DM r1(user) + aragorn should NOT see self as user
    assert any(m.sender.role == "assistant" for m in aragorn_msgs_r2)


# --- Example-Level: Mini Tic-Tac-Toe ---


def test_tic_tac_toe_chatroom():
    """Game engine Actor + 2 LLM players in a ChatRoom."""

    # Simulate a quick game: X wins with top row.
    player_x = MockedChat.from_contents(
        [
            json.dumps({"row": 0, "col": 0}),
            json.dumps({"row": 0, "col": 1}),
            json.dumps({"row": 0, "col": 2}),
        ],
        name="PlayerX",
    )
    player_o = MockedChat.from_contents(
        [
            json.dumps({"row": 1, "col": 0}),
            json.dumps({"row": 1, "col": 1}),
        ],
        name="PlayerO",
    )

    game_engine = actors.Actor(name="Game", role="user", avatar="🎮")
    room = ChatRoom(
        participants=[game_engine, player_x, player_o],
        system_prompt="A game of Tic-Tac-Toe.",
        name="Game",
    )

    with room:
        game_engine.talk("Game starts!")

        # Simulate a few turns manually.
        player_x.talk()
        game_engine.talk("X played (0,0)")
        player_o.talk()
        game_engine.talk("O played (1,0)")
        player_x.talk()
        game_engine.talk("X played (0,1)")
        player_o.talk()
        game_engine.talk("O played (1,1)")
        player_x.talk()
        game_engine.talk("X wins!")

    # 1 start + 3 X talks + 2 O talks + 5 engine talks = 11
    assert len(room.messages) == 11

    # Verify PlayerX's second move saw the history including O's first move.
    px_msgs_r2, _ = player_x.invocations[1]
    # Should see: start, X move 1 (assistant), X played (user), O move (user), O played (user)
    assert any("[PlayerO]:" in m.content for m in px_msgs_r2 if m.sender.role == "user")


# --- Example-Level: Formal Structured Debate ---


def run_debate_helper(
    resolution: str,
    pro_llm: actors.LLMChat,
    con_llm: actors.LLMChat,
    judge_llm: actors.LLMChat,
) -> dict[str, str]:
    """Runs a structured debate and evaluates the winner."""
    pro_llm.system_prompt = f"You are Pro: '{resolution}'"
    con_llm.system_prompt = f"You are Con: '{resolution}'"

    room = ChatRoom(
        participants=[pro_llm, con_llm],
        system_prompt=f"Debate: '{resolution}'",
        name="Moderator",
    )

    with room:
        room.post("Opening")
        pro_llm.talk()
        con_llm.talk()

        room.post("Rebuttal")
        pro_llm.talk()
        con_llm.talk()

    transcript = "\n".join(str(m) for m in room.messages)
    decision = judge_llm.prompt(f"Who won?\n{transcript}")
    winner = "PRO" if "WINNER: PRO" in decision else "CON"

    return {
        "winner": winner,
        "reasoning": decision,
    }


def test_structured_debate_chatroom():
    """Structured formal debate with Pro, Con, and Judge."""
    pro = MockedChat.from_contents(
        [
            "Opening Statement: We must act.",
            "Rebuttal: Opponent lacks vision.",
        ],
        name="ProDebater",
    )
    con = MockedChat.from_contents(
        [
            "Opening Statement: Risks are high.",
            "Rebuttal: Pro has no plan.",
        ],
        name="ConDebater",
    )
    # The Judge will read the transcript and decide who won
    judge = MockedChat.from_contents(
        ["Analyzing transcript... WINNER: PRO"],
        name="Judge",
    )

    result = run_debate_helper(
        resolution="Artificial Intelligence must be regulated",
        pro_llm=pro,
        con_llm=con,
        judge_llm=judge,
    )

    # Verify the results are parsed and returned correctly
    assert result["winner"] == "PRO"
    assert "WINNER: PRO" in result["reasoning"]

    # Verify Judge's prompt included the complete debate transcript
    judge_messages, _ = judge.invocations[0]
    # Invocations are to .prompt(message) -> respond() -> invoke(temp_chat)
    # The prompt message is the last message of the temp_chat
    prompt_sent = judge_messages[-1].content

    assert "Opening Statement: We must act." in prompt_sent
    assert "Opening Statement: Risks are high." in prompt_sent
    assert "Rebuttal: Opponent lacks vision." in prompt_sent
    assert "Rebuttal: Pro has no plan." in prompt_sent
    assert "ProDebater" in prompt_sent
    assert "ConDebater" in prompt_sent


# --- Mechanical Validation: Sealed Bids & Parent Delegation ---


def test_private_channel_sees_parent_history():
    """Verify that players inside a private child room can see parent public history (memory delegation)."""
    alice = MockedChat.from_contents([], name="Alice")
    bob = MockedChat.from_contents([], name="Bob")

    room = ChatRoom(participants=[alice, bob], name="Main Hall")
    with room:
        room.post("Main Announcement: Deal is $84.")

        # Alice enters a private subchannel
        whisper = room.private_channel([alice], name="Alice Whisper")
        with whisper:
            # Build Alice's perspective inside the private channel
            alice_perspective = whisper._build_perspective(alice)

            # Assert she CAN see the parent's announcement! (Memory delegation active)
            announcement_msg = next(
                (m for m in alice_perspective if "Deal is $84" in m.content), None
            )
            assert announcement_msg is not None
            assert announcement_msg.sender.name == "System"


def test_sealed_bid_isolation():
    """Verify that separate private channels prevent bidders from seeing each other's sequential submissions."""
    alpha = MockedChat.from_contents(["Alpha Bid: $90"], name="Alpha")
    beta = MockedChat.from_contents(["Beta Bid: $80"], name="Beta")

    room = ChatRoom(participants=[alpha, beta], name="Contract Room")
    with room:
        # Create private submission channels
        alpha_whisper = room.private_channel([alpha], name="Alpha Sub")
        beta_whisper = room.private_channel([beta], name="Beta Sub")

        # 1. Alpha bids privately first
        with alpha_whisper:
            alpha_whisper.post("Submit your bid.")
            alpha.talk()

        # 2. Now Beta bids privately second
        with beta_whisper:
            beta_whisper.post("Submit your bid.")

            # Build Beta's perspective at her submission time!
            beta_perspective = beta_whisper._build_perspective(beta)

            # Assert Beta CANNOT see Alpha's bid! (Sealed bid isolation active)
            for msg in beta_perspective:
                assert "Alpha Bid: $90" not in msg.content
