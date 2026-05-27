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
    assert "argues AGAINST" not in prompt  # system_prompt NOT leaked to peers
    assert "A debate." in prompt
    assert "argues FOR" in prompt  # alice's personal system_prompt


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
    assert "Topic: AI" in messages_sent[0].content
    assert messages_sent[1].content == "AI is great!"
    assert "AI is risky!" in messages_sent[2].content


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
            assert announcement_msg.sender.name == "Main Hall"


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


# --- Phase D: New Tests ---


def test_reentrant_chatroom_loop():
    """Same ChatRoom private channel entered multiple times in a loop."""
    alice = MockedChat.from_contents(["r1", "r2", "r3"], name="Alice", cycle=True)
    room = ChatRoom(participants=[alice])
    channel = room.private_channel([alice], name="Night")

    with room:
        for i in range(3):
            with channel:
                channel.post(f"Night {i}")
                alice.talk()

    night_posts = [m for m in channel.messages if "Night" in m.content]
    assert len(night_posts) == 3
    alice_msgs = [m for m in channel.messages if m.sender is alice]
    assert len(alice_msgs) == 3


def test_prompt_inside_chatroom_raises():
    """prompt() must not be called inside an active ChatRoom."""
    alice = MockedChat.from_contents(["x"], name="Alice", cycle=True)
    room = ChatRoom(participants=[alice])

    with room:
        with pytest.raises(RuntimeError, match="cannot be called inside"):
            alice.prompt("hello")


def test_roster_does_not_leak_system_prompt():
    """Secret system_prompt must not leak to peers in roster."""
    alice = MockedChat.from_contents(["x"], name="Alice")
    alice.system_prompt = "SECRET: You are a werewolf"

    bob = MockedChat.from_contents(["x"], name="Bob")
    room = ChatRoom(participants=[alice, bob])

    roster = room._build_roster(bob)
    assert "Alice" in roster
    assert "werewolf" not in roster.lower()
    assert "SECRET" not in roster


def test_post_sender_matches_room_name():
    """room.post() sender name matches the room's name, not 'System'."""
    alice = MockedChat.from_contents(["x"], name="Alice")
    room = ChatRoom(participants=[alice], name="Moderator")

    with room:
        msg = room.post("Hello")

    assert msg.sender.name == "Moderator"


def test_nested_channels_three_levels():
    """Channel-in-channel: root -> team -> leader, with correct visibility."""
    alice = MockedChat.from_contents(["cmd"], name="Alice", cycle=True)
    bob = MockedChat.from_contents(["ack"], name="Bob", cycle=True)
    charlie = MockedChat.from_contents(["x"], name="Charlie")

    room = ChatRoom(participants=[alice, bob, charlie])
    team = room.private_channel([alice, bob], name="Team")
    leader = team.private_channel([alice], name="Leader")

    with room:
        room.post("Public")
        with team:
            team.post("Team only")
            with leader:
                leader.post("Leader only")
                p = leader._build_perspective(alice)
                contents = [m.content for m in p]
                assert any("Public" in c for c in contents)
                assert any("Team only" in c for c in contents)
                assert any("Leader only" in c for c in contents)

    charlie_p = room._build_perspective(charlie)
    charlie_contents = [m.content for m in charlie_p]
    assert any("Public" in c for c in charlie_contents)
    assert not any("Team only" in c for c in charlie_contents)
    assert not any("Leader only" in c for c in charlie_contents)


def test_actor_talk_without_message_raises_typeerror():
    """Actor.talk() requires a message argument."""
    game = actors.Actor(name="Game")
    room = ChatRoom(participants=[game])
    with room:
        with pytest.raises(TypeError):
            game.talk()


def test_private_channel_validates_participants():
    """private_channel() rejects participants not in the parent room."""
    alice = MockedChat.from_contents(["x"], name="Alice")
    bob = MockedChat.from_contents(["x"], name="Bob")
    outsider = MockedChat.from_contents(["x"], name="Outsider")
    room = ChatRoom(participants=[alice, bob])

    with pytest.raises(ValueError, match="Unknown: Outsider"):
        room.private_channel([alice, outsider], name="Bad Channel")


def test_perspective_filters_invisible_messages():
    """Messages with is_visible_to_llm=False are not shown in perspective."""
    alice = MockedChat.from_contents(["x"], name="Alice")
    bob = MockedChat.from_contents(["x"], name="Bob")
    room = ChatRoom(participants=[alice, bob])

    visible_msg = Message(sender=alice, content="I can be seen")
    invisible_msg = Message(
        sender=alice, content="I am hidden", is_visible_to_llm=False
    )
    room.history = [visible_msg, invisible_msg]

    perspective = room._build_perspective(bob)
    contents = [m.content for m in perspective]
    assert any("I can be seen" in c for c in contents)
    assert not any("I am hidden" in c for c in contents)


def test_perspective_preserves_avatars():
    """Synthetic actors in perspective should retain original avatars."""
    alice = MockedChat.from_contents(["x"], name="Alice")
    alice.avatar = "🐺"
    bob = MockedChat.from_contents(["x"], name="Bob")
    bob.avatar = "🧙"
    room = ChatRoom(participants=[alice, bob])

    room.history = [
        Message(sender=alice, content="hello"),
        Message(sender=bob, content="world"),
    ]

    perspective = room._build_perspective(alice)
    assert perspective[0].sender.avatar == "🐺"
    assert perspective[1].sender.avatar == "🧙"


# --- LLM-Side Verification: What Each LLM Actually Receives ---


def test_llm_sees_correct_system_prompt_with_roster():
    """Verify the system prompt sent to the LLM contains the roster,
    room instructions, and personal prompt — but NOT peers' system_prompts."""
    alice = MockedChat.from_contents(["hello"], name="Alice")
    alice.system_prompt = "SECRET: You are a werewolf"
    bob = MockedChat.from_contents(["hi"], name="Bob")
    bob.system_prompt = "SECRET: You are a villager"

    room = ChatRoom(
        participants=[alice, bob],
        system_prompt="A game of Werewolf.",
        name="Moderator",
    )

    with room:
        alice.talk()

    # Inspect the system prompt sent to Alice's LLM
    _, kwargs = alice.invocations[0]
    system = kwargs["system"]

    # Alice should see her own roster identity
    assert "You are Alice" in system
    # Alice should see Bob listed as a peer (name only)
    assert "Bob" in system
    # Alice should see the room instructions
    assert "A game of Werewolf" in system
    # Alice should see her OWN secret system_prompt
    assert "SECRET: You are a werewolf" in system
    # Alice must NOT see Bob's secret system_prompt
    assert "SECRET: You are a villager" not in system
    # The narrator note should reference the room name
    assert 'Note on "Moderator"' in system


def test_llm_sees_correct_message_roles_and_prefixes():
    """Verify the message history sent to the LLM has correct roles
    and name prefixes: own messages as assistant, peers as user with [Name]:."""
    alice = MockedChat.from_contents(["I agree!", "Me too!"], name="Alice")
    bob = MockedChat.from_contents(["I disagree!"], name="Bob")
    room = ChatRoom(
        participants=[alice, bob],
        system_prompt="Debate.",
        name="Moderator",
    )

    with room:
        room.post("Topic: AI")
        alice.talk()
        bob.talk()
        alice.talk()  # second turn — Alice sees full history

    # Alice's 2nd invocation should see: post + alice_r1 + bob_r1
    msgs, _ = alice.invocations[1]
    assert len(msgs) == 3

    # 1. Room post → user role, prefixed with narrator name
    assert msgs[0].sender.role == "user"
    assert "[Moderator]:" in msgs[0].content
    assert "Topic: AI" in msgs[0].content

    # 2. Alice's own first message → assistant role, NO prefix
    assert msgs[1].sender.role == "assistant"
    assert msgs[1].content == "I agree!"  # raw content, no [Alice]: prefix

    # 3. Bob's message → user role, [Bob]: prefix
    assert msgs[2].sender.role == "user"
    assert "[Bob]:" in msgs[2].content
    assert "I disagree!" in msgs[2].content


def test_llm_private_channel_isolation():
    """Verify that a villager LLM never receives any messages from the
    werewolf private channel — complete information isolation."""
    alice = MockedChat.from_contents(["wolf strategy", "I'm innocent!"], name="Alice")
    bob = MockedChat.from_contents(["confirmed", "trust me"], name="Bob")
    charlie = MockedChat.from_contents(["suspicious"], name="Charlie")

    moderator = actors.Actor(name="Moderator", role="user", avatar="🧙")

    room = ChatRoom(
        participants=[moderator, alice, bob, charlie],
        system_prompt="Werewolf game.",
        name="Moderator",
    )

    with room:
        # Night: wolves coordinate secretly
        wolf_chat = room.private_channel([alice, bob], name="Wolf Night Chat")
        with wolf_chat:
            wolf_chat.post("Wolves, pick a victim.")
            alice.talk()  # "wolf strategy"
            bob.talk()  # "confirmed"

        # Day: everyone discusses publicly
        moderator.talk("Day breaks!")
        alice.talk()  # "I'm innocent!"
        charlie.talk()  # "suspicious"

    # Charlie's invocation: should see ONLY day messages
    charlie_msgs, charlie_kwargs = charlie.invocations[0]
    charlie_all_content = " ".join(m.content for m in charlie_msgs)

    # Charlie must NOT see any wolf chat content
    assert "wolf strategy" not in charlie_all_content
    assert "confirmed" not in charlie_all_content
    assert "Wolves, pick a victim" not in charlie_all_content
    assert "Wolf Night Chat" not in charlie_all_content

    # Charlie SHOULD see the day messages
    assert "Day breaks!" in charlie_all_content

    # Charlie's system prompt should NOT contain wolf-related info
    charlie_system = charlie_kwargs["system"]
    assert "Wolf Night Chat" not in charlie_system


def test_llm_knows_private_channel_context():
    """Verify that when an LLM talks inside a private channel, the
    projected messages are tagged with the channel name so the LLM
    knows it's in a private context."""
    alice = MockedChat.from_contents(["public hello", "secret plan"], name="Alice")
    bob = MockedChat.from_contents(["ack"], name="Bob")

    room = ChatRoom(
        participants=[alice, bob],
        system_prompt="A negotiation.",
        name="Boardroom",
    )

    with room:
        room.post("Welcome everyone.")
        alice.talk()  # "public hello"

        # Private channel
        whisper = room.private_channel([alice, bob], name="Side Deal")
        with whisper:
            whisper.post("This is private.")
            alice.talk()  # "secret plan" — posted inside the private channel
            bob.talk()  # "ack" — Bob sees both public and private

    # Bob's invocation inside the private channel
    bob_msgs, _ = bob.invocations[0]
    bob_all_content = " ".join(m.content for m in bob_msgs)

    # Bob should see public messages (from the root room)
    assert "Welcome everyone" in bob_all_content

    # Bob should see Alice's public message
    assert "public hello" in bob_all_content

    # Alice's message INSIDE the private channel should be tagged
    # with "(private: Side Deal)" so the LLM knows the context.
    # (Narrator messages are NOT tagged — only peer messages.)
    private_msgs = [m.content for m in bob_msgs if "private: Side Deal" in m.content]
    assert len(private_msgs) > 0
    assert any("secret plan" in m for m in private_msgs)


def test_llm_nested_channel_correct_visibility():
    """End-to-end: 3 LLMs, 3 levels of nesting. Verify each LLM
    receives exactly the messages they're authorized to see."""
    handler = MockedChat.from_contents(["mission briefing"], name="Handler")
    spy_a = MockedChat.from_contents(["intel gathered", "report"], name="SpyA")
    spy_b = MockedChat.from_contents(["cover story"], name="SpyB")

    room = ChatRoom(
        participants=[handler, spy_a, spy_b],
        system_prompt="Espionage operation.",
        name="HQ",
    )

    with room:
        room.post("All agents, report in.")
        spy_b.talk()  # "cover story" — public

        # Level 2: field team (handler + spy_a)
        field_chat = room.private_channel([handler, spy_a], name="Field Team")
        with field_chat:
            field_chat.post("Field team, coordinate.")
            spy_a.talk()  # "intel gathered"

            # Level 3: handler-only briefing
            briefing = field_chat.private_channel([handler], name="Eyes Only")
            with briefing:
                briefing.post("Top secret.")
                handler.talk()  # "mission briefing"

    # Handler should see ALL 3 levels
    handler_msgs, _ = handler.invocations[0]
    handler_content = " ".join(m.content for m in handler_msgs)
    assert "All agents, report in" in handler_content
    assert "cover story" in handler_content
    assert "Field team, coordinate" in handler_content
    assert "intel gathered" in handler_content
    assert "Top secret" in handler_content

    # SpyA should see levels 1 + 2, but NOT level 3
    spy_a_msgs, _ = spy_a.invocations[0]
    spy_a_content = " ".join(m.content for m in spy_a_msgs)
    assert "All agents, report in" in spy_a_content
    assert "cover story" in spy_a_content
    # SpyA should NOT see the Eyes Only briefing
    assert "Top secret" not in spy_a_content
    assert "mission briefing" not in spy_a_content

    # SpyB should see ONLY level 1 (public)
    spy_b_msgs, _ = spy_b.invocations[0]
    spy_b_content = " ".join(m.content for m in spy_b_msgs)
    assert "All agents, report in" in spy_b_content
    # SpyB should NOT see any private channel content
    assert "Field team, coordinate" not in spy_b_content
    assert "intel gathered" not in spy_b_content
    assert "Top secret" not in spy_b_content
    assert "mission briefing" not in spy_b_content
